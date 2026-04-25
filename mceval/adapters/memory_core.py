"""Memory Core adapter — default hosted, self-hosted compatible.

The hosted API and a locally-run Memory Core server expose the same REST
surface, so the same adapter works for both. Switch by passing ``base_url``
or setting ``MEMORY_CORE_URL``.

    # default: hosted
    MemoryCoreAdapter()

    # self-hosted, same adapter
    MemoryCoreAdapter(base_url="http://localhost:8001")
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import httpx

from .base import Memory, Turn

DEFAULT_HOSTED_URL = "https://api.memory-core.dev"


class MemoryCoreAdapter:
    name = "memory-core"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = float(os.getenv("MCEVAL_HTTP_TIMEOUT", "300")),
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("MEMORY_CORE_URL")
            or DEFAULT_HOSTED_URL
        )
        self.api_key = api_key or os.getenv("MEMORY_CORE_API_KEY")

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

        # Per-namespace content→session_id map, authoritative for THIS run.
        # Works around the mcp-memory-service backend's global content hashing:
        # if the same content string was stored earlier under a different
        # namespace, the server may return that old namespace's session tag.
        # For eval correctness we trust what WE stored this run.
        self._content_to_session: dict[str, dict[str, str]] = defaultdict(dict)
        self._map_lock = threading.Lock()

        # Per-namespace pending-store buffer. The mceval runner stores
        # all turns for a question in a tight loop and then issues one
        # search; on the LanceDB backend, single-row table.add() is ~360×
        # slower than a batch add. We buffer here and flush as one
        # POST /memories/batch call before any search() (or reset)
        # touches the same namespace, so read-after-write is preserved.
        # Default is "always buffer"; set MCEVAL_BATCH_STORE=0 to disable.
        self._batch_store_enabled = (
            os.getenv("MCEVAL_BATCH_STORE", "1").lower() not in ("0", "false", "no")
        )
        self._batch_size = int(os.getenv("MCEVAL_BATCH_STORE_SIZE", "500"))
        self._buffers: dict[str, list[dict]] = defaultdict(list)
        self._buf_lock = threading.Lock()

    def reset(self, namespace: str) -> None:
        """Best-effort cleanup: list all memories in the namespace via the
        manifest endpoint and delete them in parallel. The server has no
        bulk-delete endpoint, so we pay one DELETE per memory. Safe to call
        on an empty namespace.
        """
        # Drop any pending buffered stores for this namespace before
        # reset — otherwise we'd both clear the server and immediately
        # re-store stale items the next time we touch this namespace.
        with self._buf_lock:
            self._buffers.pop(namespace, None)
        ids: list[str] = []
        cursor: Optional[str] = None
        while True:
            params: dict[str, object] = {"namespace": namespace, "limit": 500}
            if cursor:
                params["cursor"] = cursor
            r = self._client.get("/api/v1/memories/manifest", params=params)
            if r.status_code != 200:
                break
            body = r.json()
            for e in body.get("entries", []):
                mid = e.get("memory_id") or e.get("id")
                if mid:
                    ids.append(mid)
            cursor = body.get("next_cursor")
            if not cursor:
                break

        if ids:
            def _del(mid: str) -> None:
                try:
                    self._client.delete(f"/api/v1/memories/{mid}", params={"namespace": namespace})
                except Exception:
                    pass

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(_del, ids))

        with self._map_lock:
            self._content_to_session.pop(namespace, None)

    def _build_store_payload(self, namespace: str, turn: Turn) -> dict[str, object]:
        tags = [
            f"session:{turn.session_id}",
            f"role:{turn.role}",
            f"turn:{turn.turn_idx}",
            f"session_idx:{turn.session_idx}",
        ]
        payload: dict[str, object] = {
            "content": turn.content,
            "namespace": namespace,
            "type": "observation",
            "tags": tags,
            "source_ref": f"session/{turn.session_id}/turn/{turn.turn_idx}",
        }
        if turn.timestamp is not None:
            payload["created_at"] = turn.timestamp.isoformat()
        return payload

    def _flush(self, namespace: str) -> None:
        """Drain the per-namespace buffer via one /memories/batch call.
        Caller is responsible for any concurrency control around the
        store loop."""
        with self._buf_lock:
            pending = self._buffers.get(namespace, [])
            self._buffers[namespace] = []
        if not pending:
            return
        # Chunk to respect server-side batch size limit.
        for i in range(0, len(pending), self._batch_size):
            chunk = pending[i : i + self._batch_size]
            r = self._client.post(
                "/api/v1/memories/batch",
                json={"memories": chunk, "max_parallelism": 8},
            )
            r.raise_for_status()

    def store(self, namespace: str, turn: Turn) -> str:
        payload = self._build_store_payload(namespace, turn)
        with self._map_lock:
            self._content_to_session[namespace][turn.content] = turn.session_id

        if not self._batch_store_enabled:
            r = self._client.post("/api/v1/memories/", json=payload)
            r.raise_for_status()
            return r.json().get("id", "")

        # Buffered path. Don't return a real id (the mceval runner only
        # uses the id for adapters that need it; the recall scorer keys
        # on session_id/turn_idx via tags, which are already attached).
        with self._buf_lock:
            self._buffers[namespace].append(payload)
        return ""

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        # Flush pending buffered stores before searching: read-after-write.
        self._flush(namespace)
        body: dict[str, object] = {
            "query": query,
            "namespace": namespace,
            "top_k": top_k,
            "min_score": 0.0,
        }
        if as_of_date is not None:
            body["as_of_date"] = as_of_date.isoformat()
        r = self._client.post("/api/v1/memories/search", json=body)
        r.raise_for_status()
        results = r.json().get("memories", [])

        with self._map_lock:
            content_map = dict(self._content_to_session.get(namespace, {}))

        out: list[Memory] = []
        for m in results:
            content = m.get("content", "")
            tags = m.get("tags", [])
            # Prefer this-run content→session_id over returned tags (dedup-safe).
            sid = content_map.get(content) or _extract_tag(tags, "session:")
            out.append(Memory(
                id=m.get("id", ""),
                content=content,
                score=float(m.get("score", 0.0)),
                session_id=sid,
                turn_idx=_extract_tag_int(tags, "turn:"),
                session_idx=_extract_tag_int(tags, "session_idx:"),
                metadata={"tags": tags},
            ))
        return out

    def close(self) -> None:
        self._client.close()


def _extract_tag(tags: list[str], prefix: str) -> str | None:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def _extract_tag_int(tags: list[str], prefix: str) -> int | None:
    raw = _extract_tag(tags, prefix)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None

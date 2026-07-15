"""mceval adapter for TiMEM (via local shim gateway).

TiMEM consolidates a *session* (a list of messages) into memory on each add().
mceval stores turn-by-turn, so we BUFFER turns per (namespace, session_id) and
flush one add() per session at search time — otherwise every turn would trigger
its own LLM consolidation (catastrophic for ~500-turn questions).

Each mceval question uses a unique namespace (uuid) → we map namespace → TiMEM
user_id, so questions are isolated without needing a server-side reset.

Returns consolidated memory fragments (no original session_id/turn_idx), so this
adapter is usable for the QA-accuracy harness (content only), NOT Recall@k.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict

from timem import Memory as TiMEMClient

from mceval.adapters.base import Memory, Turn


class TiMEMAdapter:
    name = "timem"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, **_):
        self._client = TiMEMClient(
            base_url=base_url or "http://127.0.0.1:8077",
            api_key=api_key or "x",
            timeout=120.0,
        )
        self._char = "a"
        self._lock = threading.Lock()
        # ns -> OrderedDict[session_id -> list[(role, content)]]
        self._buf: dict[str, "OrderedDict[str, list[tuple[str, str]]]"] = {}
        self._flushed: set[str] = set()

    def reset(self, namespace: str) -> None:
        # Unique uuid namespace → fresh user_id server-side; just clear local buffer.
        with self._lock:
            self._buf.pop(namespace, None)
            self._flushed.discard(namespace)

    def store(self, namespace: str, turn: Turn) -> str:
        with self._lock:
            ns = self._buf.setdefault(namespace, OrderedDict())
            sid = turn.session_id or "s0"
            ns.setdefault(sid, []).append((turn.role or "user", turn.content or ""))
        return ""

    def _flush(self, namespace: str) -> None:
        if namespace in self._flushed:
            return
        with self._lock:
            sessions = self._buf.get(namespace, OrderedDict())
            self._flushed.add(namespace)
        for sid, turns in sessions.items():
            messages = [{"role": r, "content": c} for r, c in turns if c]
            if not messages:
                continue
            try:
                self._client.add(
                    messages=messages,
                    user_id=namespace,
                    character_id=self._char,
                    session_id=str(sid),
                )
            except Exception:
                # one bad session shouldn't kill the question; keep going
                continue

    def search(self, namespace: str, query: str, top_k: int, as_of_date=None) -> list[Memory]:
        self._flush(namespace)
        try:
            resp = self._client.search(
                query=query,
                user_id=namespace,
                limit=top_k,
                character_id=self._char,
            )
        except Exception:
            return []
        results = resp.get("results", []) if isinstance(resp, dict) else []
        out: list[Memory] = []
        for i, r in enumerate(results[:top_k]):
            content = r.get("memory") or r.get("content") or ""
            if not content:
                continue
            out.append(Memory(
                id=str(r.get("id", f"{namespace}:{i}")),
                content=content,
                score=float(r.get("score") or 0.0),
                session_id=None,
                turn_idx=None,
                metadata={"layer": r.get("layer")},
            ))
        return out

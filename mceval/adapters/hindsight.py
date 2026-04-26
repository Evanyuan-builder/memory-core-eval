"""Hindsight adapter — peer baseline against Vectorize AI's Hindsight.

Hindsight is a same-family hybrid memory engine (semantic + BM25 + graph +
temporal, RRF fusion, cross-encoder rerank). This adapter wraps its remote
``hindsight-client`` against a user-run server (embedded ``hindsight-all`` or
Hindsight Cloud).

Design notes
------------
* One bank per namespace. Banks are created lazily on first ``store`` and
  dropped in ``reset``.
* ``document_id`` encodes ``(session_id, turn_idx)`` for unambiguous mapping;
  ``metadata`` also carries them as strings (Hindsight requires str values).
* ``recall`` has no ``top_k`` — only ``max_tokens`` + ``budget``. We request a
  generous token budget and slice to ``top_k`` locally.
* ``RecallResult`` has no score field; we synthesize a descending rank-based
  score so downstream consumers can sort identically to Hindsight's order.

Configuration via env:
  HINDSIGHT_URL      base_url (default http://127.0.0.1:8080)
  HINDSIGHT_API_KEY  optional bearer token
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from hindsight_client import Hindsight, RecallResult

from .base import Memory, Turn

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8080"
# Large token budget so we reliably get ≥ top_k candidates back.
RECALL_MAX_TOKENS = 16384


class HindsightAdapter:
    name = "hindsight"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = float(os.getenv("MCEVAL_HTTP_TIMEOUT", "300")),
        budget: str = "mid",
    ) -> None:
        self.base_url = base_url or os.getenv("HINDSIGHT_URL") or DEFAULT_URL
        self.api_key = api_key or os.getenv("HINDSIGHT_API_KEY")
        self.budget = budget
        self._client = Hindsight(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )
        # Track banks we've created this run so we know when to create vs. skip.
        self._known_banks: set[str] = set()

    def reset(self, namespace: str) -> None:
        """Drop and forget the bank. Safe on missing banks."""
        try:
            self._client.delete_bank(bank_id=namespace)
        except Exception as exc:  # 404, already-deleted, etc.
            logger.debug("delete_bank(%s) raised %s; treating as reset", namespace, exc)
        self._known_banks.discard(namespace)

    def store(self, namespace: str, turn: Turn) -> str:
        self._ensure_bank(namespace)
        doc_id = f"s{turn.session_id}__t{turn.turn_idx}"
        meta = {
            "session_id": str(turn.session_id),
            "turn_idx": str(turn.turn_idx),
            "session_idx": str(turn.session_idx),
            "role": turn.role,
        }
        resp = self._client.retain(
            bank_id=namespace,
            content=turn.content,
            timestamp=turn.timestamp,
            document_id=doc_id,
            metadata=meta,
        )
        # retain may return a single operation_id or a list.
        return resp.operation_id or (resp.operation_ids[0] if resp.operation_ids else doc_id)

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        if namespace not in self._known_banks:
            # Never stored anything — nothing to retrieve.
            return []
        kwargs: dict = {
            "bank_id": namespace,
            "query": query,
            "max_tokens": RECALL_MAX_TOKENS,
            "budget": self.budget,
        }
        if as_of_date is not None:
            kwargs["query_timestamp"] = as_of_date.isoformat()
        resp = self._client.recall(**kwargs)
        results: list[RecallResult] = list(resp.results or [])

        out: list[Memory] = []
        n = len(results)
        for rank, r in enumerate(results[:top_k]):
            sid, tidx, sidx, role = _decode(r)
            out.append(Memory(
                id=r.id,
                content=r.text,
                score=float(n - rank),  # rank-descending synthetic score
                session_id=sid,
                turn_idx=tidx,
                session_idx=sidx,
                metadata={
                    "role": role,
                    "document_id": r.document_id,
                    "hindsight_type": r.type,
                    "tags": r.tags or [],
                },
            ))
        return out

    def close(self) -> None:
        self._client.close()

    def _ensure_bank(self, namespace: str) -> None:
        if namespace in self._known_banks:
            return
        try:
            self._client.create_bank(bank_id=namespace)
        except Exception as exc:
            # Idempotent: 409 / AlreadyExists is fine; anything else bubbles up.
            msg = str(exc).lower()
            if "409" not in msg and "exist" not in msg:
                raise
        self._known_banks.add(namespace)


def _decode(r: RecallResult) -> tuple[str | None, int | None, int | None, str | None]:
    """Recover (session_id, turn_idx, session_idx, role) from result metadata.

    Prefer structured ``metadata`` dict; fall back to parsing ``document_id``.
    """
    meta = r.metadata or {}
    sid = meta.get("session_id")
    tidx = _to_int(meta.get("turn_idx"))
    sidx = _to_int(meta.get("session_idx"))
    role = meta.get("role")

    if sid is None and r.document_id:
        # document_id shape is "s<session>__t<turn>"
        doc = r.document_id
        if doc.startswith("s") and "__t" in doc:
            sid_part, _, tidx_part = doc[1:].partition("__t")
            sid = sid or sid_part
            if tidx is None:
                tidx = _to_int(tidx_part)
    return sid, tidx, sidx, role


def _to_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

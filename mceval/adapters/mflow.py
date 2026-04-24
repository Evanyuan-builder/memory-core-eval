"""M-flow adapter — peer baseline against FlowElement-ai/m_flow.

M-flow is a bio-inspired cognitive memory engine for Graph RAG. Its public
surface is three async module-level calls:

    await m_flow.add(text)        # stage a new memory
    await m_flow.memorize()       # build KG + embeddings on staged memories
    await m_flow.query(question)  # graph-routed retrieval

Caveats (read before trusting numbers from this adapter)
--------------------------------------------------------
* **No native namespace isolation.** m_flow's OSS partitioning is keyed on
  face recognition output — not usable from the eval harness. This adapter
  runs every question against the single global m_flow store; ``reset(ns)``
  clears ALL memories (best-effort via ``m_flow.reset`` or ``clear`` if the
  library exposes them, otherwise a warning + full-graph wipe). Running two
  adapters concurrently on the same m_flow daemon WILL cross-contaminate.
* **``memorize()`` lag.** m_flow separates ingest from index build. This
  adapter calls ``memorize`` lazily on the first ``search`` after any stores
  — so storing 500 turns is ~free, the cost lands on the first search.
* **Return-shape guesswork.** The README is marketing-flavoured and does
  not fix the shape of ``query`` results. This adapter tries common field
  names (``content``/``text``, ``score``, ``metadata``) and falls back to
  ``repr(result)`` if nothing matches. If your m_flow version changes the
  contract, expect TypeErrors at search time.

If you need trustworthy isolation, run m_flow in ephemeral docker containers
— one per question — and point this adapter at a fresh URL each time. That
flow is out of scope for the harness right now.

Install
-------
    pip install mflow-ai
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from .base import Memory, Turn

logger = logging.getLogger(__name__)


def _run(coro):
    """Bridge m_flow's async API into the sync MemoryAdapter protocol.

    Uses a single event loop per process so connection state inside m_flow
    (if any) is preserved across calls.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class MflowAdapter:
    name = "m-flow"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = float(os.getenv("MCEVAL_HTTP_TIMEOUT", "300")),
    ) -> None:
        try:
            import m_flow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mflow-ai is not installed. Install with: pip install mflow-ai"
            ) from e
        self._m_flow = __import__("m_flow")
        # Per-namespace pending-add counter. When >0 at search time, we call
        # memorize() first. Tracked by namespace even though m_flow itself
        # has no namespace — this keeps the lazy-memorize behaviour tidy.
        self._pending: dict[str, int] = {}
        # Best-effort record of "memorized" state so we can skip redundant
        # builds between stores and the next search.
        self._dirty: set[str] = set()

    def reset(self, namespace: str) -> None:
        """Best-effort wipe. Prefer a library-level reset if exposed."""
        mf = self._m_flow
        for fn_name in ("reset", "clear", "wipe", "drop_all"):
            fn = getattr(mf, fn_name, None)
            if fn is None:
                continue
            try:
                if asyncio.iscoroutinefunction(fn):
                    _run(fn())
                else:
                    fn()
                break
            except Exception as e:
                logger.debug("m_flow.%s() raised %s — trying next", fn_name, e)
        else:
            logger.warning(
                "m_flow exposes no reset/clear/wipe/drop_all — namespace %s "
                "not actually cleared. Eval correctness compromised.",
                namespace,
            )
        self._pending.pop(namespace, None)
        self._dirty.discard(namespace)

    def store(self, namespace: str, turn: Turn) -> str:
        """Stage the turn via m_flow.add. Index build is deferred to search."""
        mf = self._m_flow
        add = getattr(mf, "add", None)
        if add is None:
            raise RuntimeError("m_flow.add not found — incompatible m_flow version")
        # Pass extras as best-effort kwargs; m_flow may accept or ignore them.
        kwargs: dict[str, Any] = {}
        if turn.timestamp is not None:
            kwargs["timestamp"] = turn.timestamp.isoformat()
        try:
            if asyncio.iscoroutinefunction(add):
                _run(add(turn.content, **kwargs))
            else:
                add(turn.content, **kwargs)
        except TypeError:
            # Drop kwargs if the signature doesn't accept them.
            if asyncio.iscoroutinefunction(add):
                _run(add(turn.content))
            else:
                add(turn.content)

        self._pending[namespace] = self._pending.get(namespace, 0) + 1
        self._dirty.add(namespace)
        # m_flow.add doesn't surface a stable id; fabricate one from the turn
        # coordinates so the harness has something to echo back.
        return f"{turn.session_id}__{turn.turn_idx}"

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        mf = self._m_flow
        # Build any staged memories into the graph/index before querying.
        if namespace in self._dirty:
            memorize = getattr(mf, "memorize", None)
            if memorize is not None:
                try:
                    if asyncio.iscoroutinefunction(memorize):
                        _run(memorize())
                    else:
                        memorize()
                except Exception as e:
                    logger.warning("m_flow.memorize() failed: %s", e)
            self._dirty.discard(namespace)
            self._pending[namespace] = 0

        query_fn = getattr(mf, "query", None)
        if query_fn is None:
            raise RuntimeError("m_flow.query not found — incompatible m_flow version")

        try:
            if asyncio.iscoroutinefunction(query_fn):
                raw = _run(query_fn(query))
            else:
                raw = query_fn(query)
        except Exception as e:
            logger.warning("m_flow.query raised %s — returning empty result", e)
            return []

        results = _as_list(raw)
        out: list[Memory] = []
        n = len(results)
        for rank, r in enumerate(results[:top_k]):
            content, score, meta = _extract(r)
            out.append(Memory(
                id=str(meta.get("id") or f"mflow-{rank}"),
                content=content,
                score=score if score is not None else float(n - rank),
                session_id=meta.get("session_id"),
                turn_idx=_to_int(meta.get("turn_idx")),
                session_idx=_to_int(meta.get("session_idx")),
                metadata=meta,
            ))
        return out


def _as_list(raw: Any) -> list[Any]:
    """m_flow.query may return a list, a dict with ``results``, or a wrapper
    object. Peel those variants down to a plain list.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("results", "memories", "items", "episodes"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return []
    results_attr = getattr(raw, "results", None)
    if isinstance(results_attr, list):
        return results_attr
    return []


def _extract(r: Any) -> tuple[str, float | None, dict[str, Any]]:
    if isinstance(r, str):
        return r, None, {}
    if isinstance(r, dict):
        content = r.get("content") or r.get("text") or r.get("body") or ""
        score = r.get("score") if isinstance(r.get("score"), (int, float)) else None
        meta = r.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        return str(content), (float(score) if score is not None else None), meta
    # Object-style
    content = (
        getattr(r, "content", None)
        or getattr(r, "text", None)
        or getattr(r, "body", None)
        or ""
    )
    score_val = getattr(r, "score", None)
    score = float(score_val) if isinstance(score_val, (int, float)) else None
    meta = getattr(r, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    return str(content), score, meta


def _to_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

"""BM25 baseline — pure in-memory, one of the LongMemEval paper baselines.

Reference implementation that demonstrates the MemoryAdapter contract with no
network, no model dependencies. Reported as ``BM25`` on the leaderboard.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import datetime

from rank_bm25 import BM25Okapi

from .base import Memory, Turn

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25BaselineAdapter:
    name = "bm25"

    def __init__(self) -> None:
        self._corpus: dict[str, list[tuple[str, str, Turn]]] = defaultdict(list)
        self._index: dict[str, BM25Okapi] = {}
        self._dirty: dict[str, bool] = defaultdict(bool)

    def reset(self, namespace: str) -> None:
        self._corpus.pop(namespace, None)
        self._index.pop(namespace, None)
        self._dirty[namespace] = False

    def store(self, namespace: str, turn: Turn) -> str:
        mem_id = uuid.uuid4().hex
        self._corpus[namespace].append((mem_id, turn.content, turn))
        self._dirty[namespace] = True
        return mem_id

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        # BM25 baseline is time-agnostic by design; ignore as_of_date.
        del as_of_date
        if self._dirty.get(namespace, False):
            self._rebuild(namespace)
        idx = self._index.get(namespace)
        entries = self._corpus.get(namespace, [])
        if idx is None or not entries:
            return []

        scores = idx.get_scores(_tokenize(query))
        ranked = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[Memory] = []
        for i in ranked:
            mem_id, content, turn = entries[i]
            results.append(Memory(
                id=mem_id,
                content=content,
                score=float(scores[i]),
                session_id=turn.session_id,
                turn_idx=turn.turn_idx,
                session_idx=turn.session_idx,
                metadata={"role": turn.role},
            ))
        return results

    def _rebuild(self, namespace: str) -> None:
        entries = self._corpus.get(namespace, [])
        if not entries:
            self._index.pop(namespace, None)
        else:
            tokenized = [_tokenize(c) for _, c, _ in entries]
            self._index[namespace] = BM25Okapi(tokenized)
        self._dirty[namespace] = False

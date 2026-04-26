"""BM25 + Dense via Reciprocal Rank Fusion (RRF).

This is the LongMemEval paper's strongest baseline (Recall@10 ≈ 95.2% on
session-level retrieval). Documents are indexed into both BM25 and a dense
retriever. At query time, each retriever produces a ranked list; RRF fuses
them via ``score(d) = Σ 1 / (k_rrf + rank_r(d))`` with ``k_rrf = 60``.

Identity is ``(session_id, turn_idx)`` — the same turn retrieved by both
retrievers accumulates two rank contributions.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from .base import Memory, Turn
from .bm25_baseline import BM25BaselineAdapter
from .dense_baseline import DenseBaselineAdapter


class HybridRRFBaselineAdapter:
    name = "hybrid-rrf"

    def __init__(
        self,
        k_rrf: int = 60,
        retrieval_k: int = 60,
        dense_model: Optional[str] = None,
    ) -> None:
        self.k_rrf = k_rrf
        self.retrieval_k = retrieval_k
        self._bm25 = BM25BaselineAdapter()
        self._dense = (
            DenseBaselineAdapter(model=dense_model) if dense_model
            else DenseBaselineAdapter()
        )

    def reset(self, namespace: str) -> None:
        self._bm25.reset(namespace)
        self._dense.reset(namespace)

    def store(self, namespace: str, turn: Turn) -> str:
        mem_id = self._bm25.store(namespace, turn)
        self._dense.store(namespace, turn)
        return mem_id

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        bm25_hits = self._bm25.search(namespace, query, self.retrieval_k, as_of_date=as_of_date)
        dense_hits = self._dense.search(namespace, query, self.retrieval_k, as_of_date=as_of_date)

        scores: dict[str, float] = defaultdict(float)
        seen: dict[str, Memory] = {}
        for hits in (bm25_hits, dense_hits):
            for rank, m in enumerate(hits, start=1):
                key = f"{m.session_id}:{m.turn_idx}"
                scores[key] += 1.0 / (self.k_rrf + rank)
                seen.setdefault(key, m)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        out: list[Memory] = []
        for key, fused in ranked:
            m = seen[key]
            out.append(Memory(
                id=m.id,
                content=m.content,
                score=fused,
                session_id=m.session_id,
                turn_idx=m.turn_idx,
                session_idx=m.session_idx,
                metadata={**m.metadata, "fused": True},
            ))
        return out

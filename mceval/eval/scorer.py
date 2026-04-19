"""Per-question scoring: recall@k on session IDs.

The harness trusts each adapter's Memory.session_id. If your adapter stores
Turn objects but loses session_id on retrieval, your contract test will fail —
fix the adapter, not the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mceval.adapters.base import Memory


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    abstention: bool
    n_indexed: int
    n_retrieved: int
    recall: dict[int, bool]              # k -> hit
    evidence_sessions: list[str]
    retrieved_sessions: list[str]        # top-k session_ids in order
    hypothesis: str = ""                 # best retrieved content (for QA judge)
    error: str | None = None
    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)


def score(
    question_id: str,
    question_type: str,
    abstention: bool,
    evidence_sessions: set[str],
    retrieved: list[Memory],
    n_indexed: int,
    top_k_values: list[int],
    elapsed_s: float = 0.0,
) -> QuestionResult:
    retrieved_sids: list[str] = [m.session_id for m in retrieved if m.session_id]
    recall = {
        k: bool(set(retrieved_sids[:k]) & evidence_sessions)
        for k in top_k_values
    }
    return QuestionResult(
        question_id=question_id,
        question_type=question_type,
        abstention=abstention,
        n_indexed=n_indexed,
        n_retrieved=len(retrieved),
        recall=recall,
        evidence_sessions=sorted(evidence_sessions),
        retrieved_sessions=retrieved_sids,
        hypothesis=retrieved[0].content if retrieved else "",
        elapsed_s=elapsed_s,
    )

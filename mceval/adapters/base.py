"""MemoryAdapter Protocol — the contract every evaluated system must satisfy.

Design notes
------------
Namespace is an argument, not part of Turn. The eval harness uses one namespace
per question to isolate state (so question-2 can't retrieve question-1's turns).
Adapters are free to map namespaces onto whatever primitive they have (a table,
a prefix, a user-id, an index).

The Protocol is intentionally minimal. Clever behavior — temporal reranking,
multi-vector indexing, query rewriting, consolidation — lives inside adapters,
because it is part of *what is being evaluated*, not a neutral harness concern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class Turn:
    """A single conversation turn to be stored in memory."""

    content: str
    role: str                            # "user" | "assistant" | "system"
    session_id: str
    turn_idx: int                        # 0-based index within the session
    session_idx: int = 0                 # chronological session order in haystack
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """A retrieval result.

    The evaluator matches (session_id, turn_idx) against the dataset's gold
    evidence turns to compute Recall@k. Adapters MUST populate session_id and
    turn_idx when the stored content came from a Turn.
    """

    id: str
    content: str
    score: float
    session_id: str | None = None
    turn_idx: int | None = None
    session_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryAdapter(Protocol):
    """The three-method contract. Any object satisfying this is evaluatable."""

    name: str

    def reset(self, namespace: str) -> None:
        """Clear all memories in *namespace*. Must be idempotent."""
        ...

    def store(self, namespace: str, turn: Turn) -> str:
        """Index *turn* under *namespace*. Return the adapter's memory id."""
        ...

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        """Return the top_k most relevant memories in *namespace* for *query*.

        ``as_of_date`` is the reference point for resolving relative temporal
        expressions in *query* ("N days ago", "last Tuesday", "yesterday").
        Adapters that don't model time should ignore it.

        Ordering is the adapter's responsibility. The harness only reads the
        returned list as-is; it does not re-rank.
        """
        ...

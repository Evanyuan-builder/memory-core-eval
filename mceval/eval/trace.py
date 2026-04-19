"""JSONL trace writer — one row per question, flushed incrementally.

A trace row captures everything needed to audit a scored question after the
fact: what was stored, what was queried, what the adapter returned, and the
verdict. Traces are public-by-default for ``traces/sample_50/``; for full-N
runs they're attached to GitHub releases.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mceval.adapters.base import Memory, Turn

from .scorer import QuestionResult


class TraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w")

    def write(
        self,
        question_id: str,
        question: str,
        namespace: str,
        stored: list[Turn],
        retrieved: list[Memory],
        result: QuestionResult,
    ) -> None:
        row: dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "namespace": namespace,
            "stored": [_turn_to_dict(t) for t in stored],
            "retrieved": [_mem_to_dict(m) for m in retrieved],
            "result": _result_to_dict(result),
        }
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _turn_to_dict(t: Turn) -> dict[str, Any]:
    d = asdict(t)
    if t.timestamp:
        d["timestamp"] = t.timestamp.isoformat()
    return d


def _mem_to_dict(m: Memory) -> dict[str, Any]:
    return asdict(m)


def _result_to_dict(r: QuestionResult) -> dict[str, Any]:
    d = asdict(r)
    d["recall"] = {str(k): v for k, v in r.recall.items()}
    return d

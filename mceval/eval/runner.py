"""End-to-end eval orchestration.

Evaluates an adapter over a LongMemEval-shaped dataset with parallel question
workers. Each worker owns an isolated namespace so adapters don't need to
handle cross-question concurrency — only per-adapter, per-namespace.

Within a single question, stores are sequential (keeps adapters simple and
results deterministic). Question parallelism is governed by ``workers``.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from mceval.adapters.base import MemoryAdapter, Memory, Turn
from mceval.datasets.longmemeval import (
    evidence_session_ids,
    is_abstention,
    iter_turns,
)

from .metrics import compute_metrics
from .scorer import QuestionResult, score
from .trace import TraceWriter

DEFAULT_TOP_K_VALUES = [1, 5, 10]


@dataclass
class EvalOutput:
    meta: dict[str, Any]
    metrics: dict[str, Any]
    results: list[QuestionResult] = field(default_factory=list)


def _evaluate_one(
    adapter: MemoryAdapter,
    item: dict,
    top_k_values: list[int],
    trace_writer: Optional[TraceWriter] = None,
) -> QuestionResult:
    qid = item["question_id"]
    question = item["question"]
    q_type = item.get("question_type", "unknown")
    abstention = is_abstention(item)
    ev_sids = evidence_session_ids(item)
    ns = f"mceval-{uuid.uuid4().hex[:12]}"

    t0 = time.perf_counter()
    adapter.reset(ns)

    stored: list[Turn] = []
    for sid, s_idx, t_idx, role, content in iter_turns(item):
        turn = Turn(
            content=content,
            role=role,
            session_id=sid,
            turn_idx=t_idx,
            session_idx=s_idx,
        )
        try:
            adapter.store(ns, turn)
            stored.append(turn)
        except Exception:
            continue

    if not stored:
        result = QuestionResult(
            question_id=qid,
            question_type=q_type,
            abstention=abstention,
            n_indexed=0,
            n_retrieved=0,
            recall={k: False for k in top_k_values},
            evidence_sessions=sorted(ev_sids),
            retrieved_sessions=[],
            error="nothing_indexed",
            elapsed_s=time.perf_counter() - t0,
        )
    else:
        retrieved: list[Memory] = adapter.search(ns, question, top_k=max(top_k_values))
        result = score(
            question_id=qid,
            question_type=q_type,
            abstention=abstention,
            evidence_sessions=ev_sids,
            retrieved=retrieved,
            n_indexed=len(stored),
            top_k_values=top_k_values,
            elapsed_s=time.perf_counter() - t0,
        )
        if trace_writer is not None:
            trace_writer.write(qid, question, ns, stored, retrieved, result)

    try:
        adapter.reset(ns)
    except Exception:
        pass

    return result


def run_eval(
    adapter: MemoryAdapter,
    dataset: list[dict],
    top_k_values: Optional[list[int]] = None,
    workers: int = 4,
    on_progress: Optional[Callable[[int, int, QuestionResult], None]] = None,
    trace_writer: Optional[TraceWriter] = None,
) -> EvalOutput:
    top_k_values = top_k_values or DEFAULT_TOP_K_VALUES
    t_start = time.perf_counter()
    n = len(dataset)

    completed: dict[int, QuestionResult] = {}

    def _task(idx_item: tuple[int, dict]) -> tuple[int, QuestionResult]:
        idx, item = idx_item
        return idx, _evaluate_one(adapter, item, top_k_values, trace_writer)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, (i, item)): i for i, item in enumerate(dataset)}
        done = 0
        for fut in as_completed(futures):
            idx, res = fut.result()
            completed[idx] = res
            done += 1
            if on_progress:
                on_progress(done, n, res)

    results = [completed[i] for i in range(n)]
    metrics = compute_metrics(results, top_k_values)
    elapsed = time.perf_counter() - t_start

    return EvalOutput(
        meta={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adapter": getattr(adapter, "name", type(adapter).__name__),
            "dataset": "longmemeval-oracle",
            "granularity": "turn-level",
            "n_questions": n,
            "top_k_values": top_k_values,
            "workers": workers,
            "elapsed_s": round(elapsed, 1),
        },
        metrics=metrics,
        results=results,
    )

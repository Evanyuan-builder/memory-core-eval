"""Aggregate metrics across questions.

Recall@k is computed only over answerable questions. Abstention questions
(``_abs`` suffix) are tracked separately as ``abstention.precision@k`` — a
correct abstention means the system retrieved *no* evidence-matching session,
which by design is what should happen when the answer is not in the haystack.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .scorer import QuestionResult


def compute_metrics(
    results: list[QuestionResult],
    top_k_values: list[int],
) -> dict[str, Any]:
    total = len(results)
    overall: dict[int, int] = defaultdict(int)
    by_type: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    abstention_correct: dict[int, int] = defaultdict(int)

    n_ans = n_abs = 0
    for r in results:
        if r.error:
            continue
        if r.abstention:
            n_abs += 1
            for k, hit in r.recall.items():
                if not hit:
                    abstention_correct[int(k)] += 1
            continue
        n_ans += 1
        for k, hit in r.recall.items():
            k = int(k)
            overall[k] += int(hit)
            by_type[r.question_type][k].append(hit)

    denom = n_ans or 1
    abs_denom = n_abs or 1

    metrics: dict[str, Any] = {
        "n_total": total,
        "n_answerable": n_ans,
        "n_abstention": n_abs,
        "overall": {
            f"recall@{k}": round(100.0 * overall[k] / denom, 1)
            for k in top_k_values
        },
        "abstention": {
            f"precision@{k}": round(100.0 * abstention_correct[k] / abs_denom, 1)
            for k in top_k_values
        } if n_abs else {},
        "by_type": {},
    }

    for qtype, kmap in sorted(by_type.items()):
        n = len(kmap[top_k_values[0]])
        metrics["by_type"][qtype] = {
            "n": n,
            **{
                f"recall@{k}": round(100.0 * sum(kmap[k]) / (n or 1), 1)
                for k in top_k_values
            },
        }
    return metrics

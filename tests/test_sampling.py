"""Sampling behavior — stratified, seeded, and legacy head-of-list modes."""
from __future__ import annotations

from collections import Counter

import pytest

from mceval.datasets.longmemeval import _stratified_sample


def _fake_dataset() -> list[dict]:
    """Mimic LongMemEval type-clustered ordering (all of one type, then the next)."""
    data = []
    type_counts = {"temporal": 60, "multi": 40, "single-user": 30, "knowledge": 20}
    for qtype, n in type_counts.items():
        for i in range(n):
            data.append({"question_id": f"{qtype}_{i:03d}", "question_type": qtype})
    return data


def test_stratified_preserves_proportions():
    data = _fake_dataset()          # total = 150
    sample = _stratified_sample(data, 30, seed=0)
    assert len(sample) == 30

    counts = Counter(x["question_type"] for x in sample)
    # expected: temporal 12, multi 8, single-user 6, knowledge 4 (exactly)
    assert counts["temporal"] == 12
    assert counts["multi"] == 8
    assert counts["single-user"] == 6
    assert counts["knowledge"] == 4


def test_stratified_deterministic_by_seed():
    data = _fake_dataset()
    a = [x["question_id"] for x in _stratified_sample(data, 30, seed=42)]
    b = [x["question_id"] for x in _stratified_sample(data, 30, seed=42)]
    assert a == b


def test_stratified_seed_changes_selection():
    data = _fake_dataset()
    a = set(x["question_id"] for x in _stratified_sample(data, 30, seed=0))
    b = set(x["question_id"] for x in _stratified_sample(data, 30, seed=1))
    assert a != b, "different seeds should pick different items within types"


def test_stratified_handles_rounding_exactly():
    data = _fake_dataset()
    # 150 items across 4 types, request 7 — will require largest-remainder rounding
    sample = _stratified_sample(data, 7, seed=0)
    assert len(sample) == 7


@pytest.mark.parametrize("n", [1, 10, 50, 149, 150])
def test_stratified_exact_size(n):
    data = _fake_dataset()
    assert len(_stratified_sample(data, n, seed=0)) == n

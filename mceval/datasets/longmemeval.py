"""LongMemEval dataset loader.

Supports the three haystack splits shipped by ``xiaowu0162/longmemeval-cleaned``:

- ``oracle`` — keeps only evidence sessions (turn-level needle-in-haystack is
  low; adapters tend to saturate Recall@10 here).
- ``s`` — session-level haystack (~50 sessions per question). This is the
  LongMemEval-S number the paper reports (BM25 R@10 ≈ 86.2%, hybrid ≈ 95.2%).
- ``m`` — long-horizon haystack (~500 sessions, ~5k turns per question).
  Multi-GB download; plan for hours of wall-time on laptop-class hardware.

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Paper:   https://arxiv.org/abs/2410.10813
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"

SPLIT_FILENAMES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

DEFAULT_CACHE_DIR = Path(
    os.getenv("MCEVAL_CACHE_DIR")
    or Path.home() / ".cache" / "memory-core-eval"
)


def load_longmemeval(
    split: str = "oracle",
    sample: int | None = None,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
    revision: str | None = None,
    seed: int | None = None,
    stratified: bool = False,
) -> list[dict]:
    """Download (once) and return a LongMemEval split.

    Parameters
    ----------
    split
        One of ``"oracle"``, ``"s"``, ``"m"``. See module docstring.
    sample
        If set, return N items. Without ``seed`` or ``stratified`` this takes
        ``data[:N]`` — which is BIASED because the dataset is clustered by
        question type. For any N < len(data) you almost certainly want
        ``seed`` or ``stratified``.
    cache_dir
        Where to cache the dataset file. Defaults to ``~/.cache/memory-core-eval``.
    expected_sha256
        If provided, verify the downloaded file matches this hex digest.
        Raises ``ValueError`` on mismatch.
    revision
        Pin a specific HuggingFace revision (commit hash / tag). Recommended
        for reproducibility.
    seed
        If set, deterministically shuffle before sampling. Ignored if
        ``stratified`` is True (stratified uses seed internally).
    stratified
        If True, sample proportionally across ``question_type`` so the mix
        mirrors the full dataset. Uses ``seed`` (default 0) for determinism.
    """
    split = split.lower()
    if split not in SPLIT_FILENAMES:
        raise ValueError(
            f"unknown LongMemEval split: {split!r}. "
            f"choose one of {sorted(SPLIT_FILENAMES)}"
        )
    filename = SPLIT_FILENAMES[split]

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / filename

    if not cache_file.exists():
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir),
            revision=revision,
        )
        if Path(downloaded).resolve() != cache_file.resolve():
            cache_file.write_bytes(Path(downloaded).read_bytes())

    if expected_sha256:
        actual = hashlib.sha256(cache_file.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Dataset sha256 mismatch. Expected {expected_sha256}, got {actual}. "
                "Delete the cache and retry, or check the expected hash."
            )

    data = json.loads(cache_file.read_text())

    if sample is None or sample >= len(data):
        return data

    if stratified:
        return _stratified_sample(data, sample, seed or 0)

    if seed is not None:
        rng = random.Random(seed)
        shuffled = data[:]
        rng.shuffle(shuffled)
        return shuffled[:sample]

    return data[:sample]


def load_longmemeval_oracle(**kwargs) -> list[dict]:
    """Back-compat wrapper — equivalent to ``load_longmemeval(split="oracle", ...)``."""
    return load_longmemeval(split="oracle", **kwargs)


def _stratified_sample(data: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional per-type sample using largest-remainder rounding to hit n
    exactly. Within each type, items are picked deterministically by ``seed``.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_type[item.get("question_type", "unknown")].append(item)

    total = len(data)
    raw = {t: n * len(items) / total for t, items in by_type.items()}
    floor_alloc = {t: int(v) for t, v in raw.items()}
    remainder = n - sum(floor_alloc.values())
    ranked = sorted(raw.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
    for t, _ in ranked[:remainder]:
        floor_alloc[t] += 1

    out: list[dict] = []
    for t, items in by_type.items():
        k = min(floor_alloc[t], len(items))
        if k <= 0:
            continue
        picked = rng.sample(items, k)
        out.extend(picked)

    out.sort(key=lambda x: x.get("question_id", ""))
    return out


def iter_turns(item: dict) -> Iterable[tuple[str, int, int, str, str]]:
    """Yield (session_id, session_idx, turn_idx, role, content) for each turn
    in a LongMemEval question's haystack.
    """
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(sessions):
        sid = str(session_ids[s_idx]) if s_idx < len(session_ids) else str(s_idx)
        for t_idx, turn in enumerate(session):
            content = (turn.get("content") or "").strip()
            if content:
                yield sid, s_idx, t_idx, turn.get("role", "user"), content


def evidence_session_ids(item: dict) -> set[str]:
    """Extract the set of gold evidence session IDs for a question.

    Prefers ``answer_session_ids`` (authoritative across all splits). Falls
    back to the legacy ``evidence[].session_id`` shape for older snapshots.
    Abstention questions (``_abs`` suffix) have no evidence by design.
    """
    if is_abstention(item):
        return set()

    sids = {str(sid) for sid in item.get("answer_session_ids") or [] if sid}
    if sids:
        return sids

    for ev in item.get("evidence") or []:
        sid = ev.get("session_id") or ev.get("evidence_session_id")
        if sid:
            sids.add(str(sid))
    return sids


def is_abstention(item: dict) -> bool:
    return item.get("question_id", "").endswith("_abs")

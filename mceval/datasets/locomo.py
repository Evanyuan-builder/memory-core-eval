"""LoCoMo benchmark dataset loader.

LoCoMo (Maharana et al. 2024) is a long-range conversational memory benchmark:
10 conversations × ~30 sessions × ~600 turns, each paired with ~200 QA pairs
whose ``evidence`` points to specific dialog turns via ``dia_id`` of the form
``"D<session_num>:<turn_num>"`` (both 1-indexed). 5 categories (integers 1–5).

This loader flattens the dataset so each LoCoMo QA becomes one item with the
LongMemEval item shape, letting the existing eval runner (``iter_turns``,
``evidence_session_ids``, scorer) operate unchanged. Recall here is thus
**session-level** — looser than the paper's dia_id-level metric, but
apples-to-apples with LongMemEval numbers produced by the same runner.

Same-sample QA pairs share the same haystack. The runner re-ingests per
question (one namespace per question); for full 1986-question runs this is
wasteful but correct. Start with ``--sample`` + ``--stratified`` for a smoke.

Source: https://github.com/snap-research/locomo (``data/locomo10.json``)
Paper:  https://arxiv.org/abs/2402.17753
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from datetime import datetime
from pathlib import Path

from .longmemeval import DEFAULT_CACHE_DIR, _stratified_sample

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_CACHE_FILENAME = "locomo10.json"

_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")
_LOCOMO_DT_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s+(am|pm)\s+on\s+(\d{1,2})\s+(\w+),\s+(\d{4})\s*$",
    re.IGNORECASE,
)
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1)}
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_locomo(
    sample: int | None = None,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
    seed: int | None = None,
    stratified: bool = False,
) -> list[dict]:
    """Download (once) and return LoCoMo flattened to LongMemEval-shape items.

    Parameters mirror ``load_longmemeval``. ``stratified`` samples
    proportionally across ``question_type`` (i.e. ``category_1`` … ``category_5``).
    QA pairs whose evidence resolves to zero sessions (4/1986 in the released
    file) are dropped so session-level recall stays well-defined.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / LOCOMO_CACHE_FILENAME

    if not cache_file.exists():
        with urllib.request.urlopen(LOCOMO_URL, timeout=60) as resp:
            cache_file.write_bytes(resp.read())

    if expected_sha256:
        actual = hashlib.sha256(cache_file.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"LoCoMo sha256 mismatch. Expected {expected_sha256}, got {actual}. "
                "Delete the cache and retry, or check the expected hash."
            )

    raw = json.loads(cache_file.read_text())
    items: list[dict] = []
    for convo in raw:
        items.extend(_convo_to_items(convo))

    if sample is None or sample >= len(items):
        return items

    if stratified:
        return _stratified_sample(items, sample, seed or 0)

    if seed is not None:
        rng = random.Random(seed)
        shuffled = items[:]
        rng.shuffle(shuffled)
        return shuffled[:sample]

    return items[:sample]


def _convo_to_items(convo: dict) -> list[dict]:
    sample_id = convo["sample_id"]
    conversation = convo["conversation"]

    session_keys = sorted(
        [k for k in conversation
         if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    haystack_sessions: list[list[dict]] = []
    haystack_session_ids: list[str] = []
    haystack_dates: list[str | None] = []
    session_num_to_sid: dict[int, str] = {}

    for sk in session_keys:
        num = int(sk.split("_")[1])
        sid = f"{sample_id}:{sk}"
        session_num_to_sid[num] = sid
        turns = [
            {"role": t.get("speaker") or "user", "content": t.get("text") or ""}
            for t in conversation[sk]
        ]
        haystack_sessions.append(turns)
        haystack_session_ids.append(sid)
        haystack_dates.append(_format_locomo_dt(conversation.get(f"{sk}_date_time")))

    items: list[dict] = []
    for q_idx, qa in enumerate(convo.get("qa", [])):
        session_ids_set: set[str] = set()
        for ev in qa.get("evidence") or []:
            parsed = _parse_dia_id(ev)
            if parsed is None:
                continue
            snum, _tnum = parsed
            sid = session_num_to_sid.get(snum)
            if sid is not None:
                session_ids_set.add(sid)

        if not session_ids_set:
            # No parseable evidence — session-level recall is undefined. Skip.
            continue

        items.append({
            "question_id": f"{sample_id}:q{q_idx}",
            "question": qa.get("question") or "",
            "answer": qa.get("answer") or "",
            "question_type": f"category_{qa.get('category')}",
            "question_date": None,
            "haystack_sessions": haystack_sessions,
            "haystack_session_ids": haystack_session_ids,
            "haystack_dates": haystack_dates,
            "answer_session_ids": sorted(session_ids_set),
        })
    return items


def _parse_dia_id(s: object) -> tuple[int, int] | None:
    if not isinstance(s, str):
        return None
    m = _DIA_ID_RE.match(s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _format_locomo_dt(raw: str | None) -> str | None:
    """``"1:56 pm on 8 May, 2023"`` → ``"2023/05/08 (Mon) 13:56"`` so the
    existing ``parse_longmemeval_date`` accepts it unchanged.
    """
    if not raw:
        return None
    m = _LOCOMO_DT_RE.match(raw)
    if not m:
        return None
    hh, mm, ampm, dd, month_str, yyyy = m.groups()
    month = _MONTHS.get(month_str.lower())
    if month is None:
        return None
    hour = int(hh) % 12 + (12 if ampm.lower() == "pm" else 0)
    try:
        dt = datetime(int(yyyy), month, int(dd), hour, int(mm))
    except ValueError:
        return None
    return (
        f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d} "
        f"({_WEEKDAYS[dt.weekday()]}) {dt.hour:02d}:{dt.minute:02d}"
    )

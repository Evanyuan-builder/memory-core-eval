"""A/B diagnosis: memory-core vs hybrid-rrf on specific gap questions.

For each target question, stores the same turns in both adapters, issues
the same query, and reports where the gold session lands in top-K of each.

If memory-core has gold beyond top-10 but within top-K, the issue is in
ranking (composite / CE rerank ordering). If memory-core's top-K doesn't
contain gold at all while hybrid-rrf does, the issue is upstream of
ranking — retrieval candidate pool / overfetch cap.

Usage::

    MEMORY_CORE_URL=http://127.0.0.1:8001 \\
        python -m mceval.diagnose.ab conv-41:q19 conv-42:q186 conv-47:q27
"""
from __future__ import annotations

import os
import sys
import uuid

from mceval.adapters.hybrid_rrf_baseline import HybridRRFBaselineAdapter
from mceval.adapters.memory_core import MemoryCoreAdapter
from mceval.adapters.base import Turn
from mceval.datasets.locomo import load_locomo
from mceval.datasets.longmemeval import parse_longmemeval_date

DEFAULT_TARGETS = [
    "conv-30:q49",   # cat_4, 369 turns, "What did Gina want her customers to feel?"
    "conv-26:q41",   # cat_2, 419 turns, "When did Caroline join a new activist group?"
    "conv-26:q168",  # cat_5, 419 turns, "What are the new shoes Caroline got used for?"
]
TOP_K = 100


def iter_turns_locomo(item):
    for s_idx, (sid, turns, session_date_raw) in enumerate(zip(
        item["haystack_session_ids"],
        item["haystack_sessions"],
        item["haystack_dates"],
    )):
        session_date = parse_longmemeval_date(session_date_raw)
        for t_idx, turn in enumerate(turns):
            yield (
                sid, s_idx, t_idx,
                turn.get("role", "user"),
                turn["content"],
                session_date,
            )


def run_one(item, mc, hy):
    qid = item["question_id"]
    question = item["question"]
    gold_sids = set(item["answer_session_ids"])
    q_date = parse_longmemeval_date(item.get("question_date"))
    haystack_n = sum(len(s) for s in item["haystack_sessions"])

    print(f"\n{'='*78}")
    print(f"== {qid} ({item['question_type']}) | haystack={haystack_n} turns ==")
    print(f"  Q: {question}")
    print(f"  A: {item['answer']!r}")
    print(f"  gold: {sorted(gold_sids)}")

    ns = f"diag-{uuid.uuid4().hex[:10]}"

    for ad_name, ad in [("memory-core", mc), ("hybrid-rrf", hy)]:
        ad.reset(ns)
    for sid, s_idx, t_idx, role, content, sdate in iter_turns_locomo(item):
        t = Turn(content=content, role=role, session_id=sid,
                 turn_idx=t_idx, session_idx=s_idx, timestamp=sdate)
        try:
            mc.store(ns, t)
        except Exception as e:
            print(f"  mc.store err: {e}")
        try:
            hy.store(ns, t)
        except Exception as e:
            print(f"  hy.store err: {e}")

    mc_hits = mc.search(ns, question, top_k=TOP_K, as_of_date=q_date)
    hy_hits = hy.search(ns, question, top_k=TOP_K, as_of_date=q_date)

    def gold_positions(hits):
        return [i + 1 for i, m in enumerate(hits) if m.session_id in gold_sids]

    mc_pos = gold_positions(mc_hits)
    hy_pos = gold_positions(hy_hits)

    print(f"\n  memory-core top-{TOP_K} gold positions: {mc_pos or 'NONE'}")
    print(f"  hybrid-rrf  top-{TOP_K} gold positions: {hy_pos or 'NONE'}")

    # Extra diagnosis: are gold turns even retrieved within top-20 by memory-core?
    if not mc_pos:
        print(f"  → memory-core did NOT surface gold within top-{TOP_K}")
        print(f"    (retrieval-stage loss — gold dropped before composite ranking)")
    elif min(mc_pos) > 10:
        print(f"  → memory-core HAS gold at rank {min(mc_pos)}, beyond top-10")
        print(f"    (composite-rerank loss — gold retrieved but ranked too low)")
    else:
        print(f"  → memory-core hits gold at rank {min(mc_pos)} (R@10 ok)")

    # Top-5 compact view side-by-side, mark gold
    print(f"\n  --- compact top-10 ---")
    print(f"  {'rank':<6}{'memory-core (session:turn)':<44}{'hybrid-rrf (session:turn)':<40}")
    for i in range(10):
        mcm = mc_hits[i] if i < len(mc_hits) else None
        hym = hy_hits[i] if i < len(hy_hits) else None
        mc_s = (f"[{mcm.session_id}:{mcm.turn_idx}]"
                + ("★" if mcm.session_id in gold_sids else "")) if mcm else "-"
        hy_s = (f"[{hym.session_id}:{hym.turn_idx}]"
                + ("★" if hym.session_id in gold_sids else "")) if hym else "-"
        print(f"  {i+1:<6}{mc_s:<44}{hy_s:<40}")

    mc.reset(ns)
    hy.reset(ns)
    return {"qid": qid, "mc_pos": mc_pos, "hy_pos": hy_pos}


def main():
    targets = sys.argv[1:] or DEFAULT_TARGETS
    sel = load_locomo(sample=100, seed=0, stratified=True)
    by_qid = {q["question_id"]: q for q in sel}

    mc = MemoryCoreAdapter(base_url=os.environ["MEMORY_CORE_URL"])
    hy = HybridRRFBaselineAdapter()

    results = []
    for qid in targets:
        if qid not in by_qid:
            print(f"[skip] {qid} not in stratified n=100 seed=0 sample")
            continue
        results.append(run_one(by_qid[qid], mc, hy))

    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    for r in results:
        mc_top10 = any(p <= 10 for p in r["mc_pos"])
        hy_top10 = any(p <= 10 for p in r["hy_pos"])
        mc_top20 = bool(r["mc_pos"])
        if mc_top10:
            diag = "R@10 ok"
        elif mc_top20:
            diag = f"COMPOSITE-LOSS (gold at mc rank {min(r['mc_pos'])})"
        else:
            diag = "RETRIEVAL-LOSS (gold not in mc top-20)"
        mc_str = str(r["mc_pos"]) if r["mc_pos"] else "miss"
        hy_str = str(r["hy_pos"]) if r["hy_pos"] else "miss"
        print(f"  {r['qid']:<20} mc={mc_str:<22} hy={hy_str:<22} {diag}")


if __name__ == "__main__":
    main()

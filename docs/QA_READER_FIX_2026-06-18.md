# QA-accuracy harness: reader fix + apples-to-apples (2026-06-18)

Follow-up to the 2026-06-14 finding that QA accuracy was **reader-bound, not
retrieval-bound**. Two things done here:

1. **Fixed the reader** (`qa_runner.py`): snippets are now prefixed with
   `[speaker · date]` (built from each question's own turns at ingest), and the
   system prompt no longer pushes the model to bail to "I don't know." The
   original conservative reader is preserved as `--reader-prompt v1`; the new one
   is `v2` (default). `v1` reproduces the pre-2026-06-18 matrix exactly.
2. **TiMEM apples-to-apples**: TiMEM's retrieved memories run through the *same*
   reader+judge as everyone else, isolating "memory quality" from "TiMEM's own
   pipeline + own grader" (their published LoCoMo 75.30 / LME-S 76.88).

## Results — QA accuracy (deepseek-v4-flash reader+judge, n=100 stratified seed=0)

> Our own LLM judge, NOT the official LongMemEval GPT-4 grader. Numbers are for
> same-axis *relative* comparison only.

| retriever | LoCoMo v1 | LoCoMo v2 | LME-S v1 | LME-S v2 |
|---|---:|---:|---:|---:|
| memory-core | 26 | **41** | 49 | **60** |
| hybrid-rrf | 25 | 36 | 44 | 58 |
| bm25 | 23 | 32 | 45 | 58 |
| timem | 29 | — | — (wedged) | — |

v1 = conservative reader + bare snippets. v2 = improved reader + `[speaker·date]`
enrichment (turn-level retrievers only; TiMEM returns consolidated fragments that
can't be enriched, which is itself a property of the system).

## What the numbers say

- **QA was reader-bound.** Same retrieval, +14–15pt on answerable just by giving
  the reader speaker/date and letting it reason. The old 26/49 was a floor
  imposed by a handicapped reader, not memory-core's ceiling.
- **memory-core leads on both** (41 LoCoMo, 60 LME-S), but the *shape* differs:
  - **LoCoMo**: retrieval gap is real and material (41 vs 36 vs 32). Speaker-centric
    multi-session conversation rewards better retrieval.
  - **LME-S**: gap collapses to ≤2pt (60 / 58 / 58). Recall is saturated
    (memory-core R@10 = 100 here), so all three surface the evidence and the
    reader becomes the bottleneck — every retriever hits the same reader ceiling.
- **The remaining ceiling is reader reasoning, not retrieval.** memory-core LME-S
  by-type: single-session-user 92.9 / assistant 90.9 / knowledge-update 80, but
  **multi-session 48.1 / temporal-reasoning 40.7** — the aggregation/time-math
  types deepseek-flash struggles with, despite recall=100.
- **Loosening the prompt traded abstention for answerable**: memory-core LME-S
  abstention 100→66.7 (n=6, i.e. 2 questions), answerable 45.7→59.6. Net +11. A
  product would tune the abstention threshold separately.

## TiMEM apples-to-apples conclusion

Under the identical bare reader (v1), **TiMEM LoCoMo = 29 vs memory-core = 26** —
parity, not the ~46pt advantage their published 75.30 implies. That gap is almost
entirely TiMEM's reader pipeline + their own grader, **not** memory quality. The
probe showed TiMEM's consolidation actively *loses* detail needed for
multi-session aggregation (e.g. "total distance across four trips" gold 3,000 mi
→ consolidated to "1,800 miles across three trips").

**TiMEM LME-S did not complete**: the run wedged at the tail after ~4h. The
consolidation-heavy LME-S questions (large multi-session) are exactly the
high-volume re-ingest throughput wall flagged for this class of backend on
2026-04-23. To fill the cell, re-run with a hard per-question timeout + lower
concurrency. The conclusion above does not depend on it.

## Reproduce

```bash
# v2 (default, improved reader)
LLM_API_KEY=... PYTHONPATH=. HF_HUB_OFFLINE=1 \
  python -m mceval.eval.qa_runner --adapter memory-core --dataset locomo \
  --sample 100 --stratified --seed 0 --workers 2 --base-url http://127.0.0.1:8001 \
  --out-dir baselines_qa
# v1 (reproduce the old conservative/bare matrix)
  ... --reader-prompt v1
```

Strong stack must be up on :8001 (LANCEDB_PRIMARY=true + bge-large, HF offline).

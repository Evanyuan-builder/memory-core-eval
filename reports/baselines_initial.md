# Initial Baselines — LongMemEval Oracle Split (n=100 stratified)

**Date:** 2026-04-19
**Harness:** `memory-core-eval` v0.1.0
**Dataset:** `xiaowu0162/longmemeval-cleaned` · oracle split · turn-level
**Sample:** 100 questions, stratified by `question_type`, `seed=0`
**Hardware:** MacBook, Python 3.12, CPU-only (sentence-transformers `all-MiniLM-L6-v2`)

---

## Head-to-head

| Adapter | R@1 | R@5 | R@10 | Abstention P@10 | Time |
|---|---|---|---|---|---|
| BM25 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2s |
| Dense MiniLM | 100.0% | 100.0% | 100.0% | 100.0% | 18s |
| BM25 + Dense RRF | 100.0% | 100.0% | 100.0% | 100.0% | 18s |
| **Memory Core v1.1.0** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 934s |

All four adapters saturate the oracle split on n=100 stratified. Memory Core
matches the classical baselines on recall while going through the full
write→retrieve API path (hence the ~15m runtime vs in-memory BM25/Dense).

Reproduce:

```bash
pip install 'memory-core-eval[dense]'
mceval run --adapter bm25       --sample 100 --stratified --seed 0
mceval run --adapter dense      --sample 100 --stratified --seed 0
mceval run --adapter hybrid-rrf --sample 100 --stratified --seed 0
mceval run --adapter memory-core --base-url http://127.0.0.1:8001 \
           --sample 100 --stratified --seed 0
```

---

## Caveat — oracle split under-discriminates

Per the LongMemEval paper, **oracle split** keeps only sessions that contain
evidence for each question. With turn-level indexing, the needle-in-haystack
ratio is low enough that even pure BM25 saturates Recall@10 on this
well-specified benchmark slice.

This is **not** a weakness of the harness — it is a property of the oracle
split. To differentiate systems meaningfully we need the harder splits:

- **LongMemEval-S** — session-level haystack (paper reports BM25 @ 86.2%, hybrid @ 95.2%)
- **LongMemEval-M/L** — longer horizons with distractor sessions

Adding these splits is tracked as a follow-up.

---

## Per-type breakdown (100 stratified questions)

| Question type | n | BM25 R@1 | Dense R@1 | Hybrid R@1 | MC R@1 |
|---|---|---|---|---|---|
| knowledge-update | 15 | 100.0% | 100.0% | 100.0% | 100.0% |
| multi-session | 25 | 100.0% | 100.0% | 100.0% | 100.0% |
| single-session-assistant | 11 | 100.0% | 100.0% | 100.0% | 100.0% |
| single-session-preference | 6 | 100.0% | 100.0% | 100.0% | 100.0% |
| single-session-user | 13 | 100.0% | 100.0% | 100.0% | 100.0% |
| temporal-reasoning | 27 | 100.0% | 100.0% | 100.0% | 100.0% |
| abstention (✗=correct) | 3 | 100.0% | 100.0% | 100.0% | 100.0% |

---

## Artifacts

Full per-question results are in `baselines/*.json`. Full traces available on
request (linked from GitHub release when the n=500 audit bundle is published).

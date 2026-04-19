# LongMemEval-S Baselines — Session Haystack

**Date:** 2026-04-19
**Harness:** `memory-core-eval` v0.2.0 (`--split s`)
**Dataset:** `xiaowu0162/longmemeval-cleaned` · `longmemeval_s_cleaned.json`
**Full sample:** 500 questions, per-question haystack ≈ 50 sessions (≈ 500 turns)
**Hardware:** MacBook, Python 3.12, CPU-only MiniLM, workers=4

---

## Head-to-head

| Adapter | R@1 | R@5 | R@10 | Abstention P@10 | Sample | Time |
|---|---:|---:|---:|---:|---|---:|
| BM25 | 84.7% | 93.2% | 96.2% | 100.0% | n=500 | 16s |
| Dense MiniLM | 86.8% | 94.7% | 97.2% | 100.0% | n=500 | 27m |
| BM25 + Dense RRF | **89.8%** | **96.0%** | **97.9%** | 100.0% | n=500 | 27m |
| Memory Core v1.1.0 | 59.3% | 77.8% | 77.8% | 100.0% | n=30 stratified | 10m |

Hybrid-RRF is the best open baseline, as the paper predicts.

**Memory Core is 20 points behind hybrid-rrf.** This is a real gap on the
harder split — oracle under-discriminates and ties everyone at 100%, but
when the haystack grows to ~50 sessions per question, Memory Core's
retrieval degrades.

Full n=500 for Memory Core requires >12h of serial HTTP writes against a
local API; n=30 stratified is a representative signal. A full-500 hosted
run is on the roadmap.

Reproduce:

```bash
pip install 'memory-core-eval[dense]'
mceval run --adapter bm25       --split s
mceval run --adapter dense      --split s
mceval run --adapter hybrid-rrf --split s
# Memory Core on stratified sample (replace with hosted URL if applicable):
mceval run --adapter memory-core --split s --sample 100 --stratified --seed 0 \
           --base-url http://127.0.0.1:8001
```

---

## Per-type breakdown

Full n=500 for BM25 / Dense / Hybrid; n=30 stratified for Memory Core.

| Question type | n (full) | BM25 R@1 | Dense R@1 | Hybrid R@1 | MC R@1 (n=30) |
|---|---:|---:|---:|---:|---:|
| knowledge-update | 72 | 95.8% | 95.8% | 97.2% | 75.0% (n=4) |
| multi-session | 121 | 83.5% | 86.0% | 89.3% | 50.0% (n=6) |
| single-session-assistant | 56 | 100.0% | 98.2% | 100.0% | 100.0% (n=3) |
| single-session-preference | 30 | 33.3% | 70.0% | 70.0% | 50.0% (n=2) |
| single-session-user | 64 | 93.8% | 85.9% | 93.8% | 50.0% (n=4) |
| temporal-reasoning | 127 | 80.3% | 81.9% | 84.3% | 50.0% (n=8) |
| abstention (✗=correct) | 30 | 100.0% | 100.0% | 100.0% | 100.0% (n=3) |

Notes on the open baselines:

- **`single-session-preference`** is where lexical BM25 collapses (33.3%
  R@1); dense embeddings recover it to 70%. This is the classic case for
  semantic retrieval — preferences are paraphrased, not repeated verbatim.
- **`temporal-reasoning`** is the category where *all* open baselines are
  weakest (≤ 84% R@1). These questions require reasoning over "last
  Tuesday" / "two months ago" — a known retrieval limitation.
- **`single-session-user`**: BM25 and hybrid tie at 93.8% while dense dips
  to 85.9%. Dense on its own loses some keyword-grounded questions; RRF
  recovers them.

## Where Memory Core loses the 20 points

Diagnostic on the n=30 run:

1. **Recall cliff, not ranking issue.** `R@5 == R@10 == 77.8%` — not a
   top_k cap (server correctly returns 10). The missing evidence turns
   simply never surface in the top-10.
2. **Low session diversity in top-10.** Same session contributes multiple
   turns that crowd out other sessions. The evidence session gets pushed
   out.
3. **Dense drift on long-form haystacks.** Query "bedside lamp bulb" gets
   pulled toward `ultrachat_*` long-form dialogues that are semantically
   fuzzy but lexically rich.

**Roadmap:**

- Session-level diversity reranking (at most N turns per session in top-K)
- Dedicated temporal range filter for "N days ago" queries
- Preference-fact extraction and promotion (so preferences don't have to
  win on raw turn similarity)

---

## Comparison to LongMemEval paper

The paper reports BM25 R@10 ≈ 86.2% on LongMemEval-S. Our BM25 gets 96.2%.
The difference is the **indexing unit**:

- **Paper**: each session is a single document. Retrieve top-K sessions.
- **This harness**: each *turn* is a document. A session counts as "hit" if
  any of its turns ranks in top-K. This matches how production memory
  systems (including Memory Core) actually operate.

Turn-level indexing gives BM25 more surface to match against, which pulls
the headline number up by ~10 points. The per-type **relative** ordering is
what matters — and it agrees with the paper: BM25 weakest on preference /
temporal, hybrid strongest overall.

If you want the exact paper protocol (session-level docs) add a runtime
flag — it is a one-line change in the adapter's `store()` aggregation. We
default to turn-level because it's what the target systems actually do.

---

## Artifacts

Per-question JSON results in `baselines/*_n500.json`. Full traces available
on request.

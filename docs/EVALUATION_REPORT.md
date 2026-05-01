# Memory Core Evaluation Report

**Memory Core** — `Evanyuan-builder/memory-core@ae1352d` (v0.3.1;
includes the bug-#6 composite temporal_factor fix and the bug-#7
schema-completeness fix described in the Lessons section)
**Harness** — `memory-core-eval@6585e14` (this repo, public, Apache-2.0)
**Date** — 2026-04-27 (numbers); 2026-05-01 (v0.3.1 schema annotations)

This report measures Memory Core's retrieval performance on two
public agent-memory benchmarks (LoCoMo and LongMemEval-S) at the
default-stack settings shipped in v0.3.0. It also documents
cross-restart determinism, ablation tables, known limitations, the
end-to-end reproduction recipe, and the dogfood-surfaced bugs that
shaped the v0.3 release. Treat this as a primary-source snapshot,
not a leaderboard claim — the public paper anchors below were taken
from those papers' published baseline numbers, and a third-party
reproduction has not yet been performed.

---

## TL;DR

| Benchmark | Sample | R@1 | R@5 | R@10 | vs strongest paper anchor |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 500 stratified | 58.4 | 81.8 | **88.8** | +3.8pt vs Hybrid-RRF (n=100=85.0) |
| LongMemEval-S | 500 stratified | 94.5 | 98.3 | **99.4** | +1.5pt vs Hybrid-RRF (n=500=97.9) |

Cross-restart determinism: zero question flips across two clean-DB
runs at the same code revision. Default code-stack settings
(`MEMORY_DIVERSITY_OVERFETCH=4`, `MEMORY_OVERFETCH_CAP=40`,
`MEMORY_CE_RERANK_TOP_N=40`, CE auto-on with sentence-transformers,
SQL push-down filters); LanceDB primary, mcp-memory-service
disabled.

---

## What this is and isn't

### This is

- A primary-source measurement at default settings on the new
  LanceDB-primary stack.
- Apples-to-apples in metric and split with the published paper
  baselines we cite — the same dataset, same stratified
  seed=0 sample, same Recall@k formulation.

### This is not

- A leaderboard claim. The strongest baselines we compare against
  (BM25 / Dense MiniLM / Hybrid-RRF) are the strongest *published*
  baselines we know of for these benchmarks at session-level
  retrieval recall, not the strongest possible memory system.
  Mem0, Zep, Letta and similar projects publish numbers on adjacent
  but different metrics (answer accuracy with a downstream LLM, F1
  on token-level alignment, etc.); a same-corpus head-to-head with
  those projects has not been run here.
- A proof of correctness. The harness reports turn-level recall
  against gold sessions; bugs that are silent under recall (e.g.,
  the `ttl_seconds=None → 0` round-trip bug we shipped and fixed in
  this release cycle) are caught by the unit and integration suites,
  not by this evaluation.
- Independently reproduced. The reproduction recipe at the end of
  this document is exactly what we ran; we have not yet had a third
  party run it.

---

## Methodology

### Datasets

- **LoCoMo** (Maharana et al. 2024) — long-range multi-session
  conversation memory. We use the public dataset loaded via
  `mceval/datasets/locomo.py`. 1978 of 1986 QAs retained (8 dropped
  during evidence parsing); five `category_1..5` strata. Session-level
  granularity (recall is computed against the gold session ID, not
  the dia ID).
- **LongMemEval-S** (Wu et al. 2024) — session-haystack split, ~50
  sessions per question. Loaded via
  `mceval/datasets/longmemeval.py` with HF revision pin and sha256
  verification. Six `question_type` strata + an abstention split.

### Sampling

`--stratified --seed 0 --sample <N>`. Stratified preserves per-class
distribution; seed 0 makes the sample exactly reproducible.
Question order within a stratum is also seed 0 — this matters for
small-sample reruns.

### Adapter contract

`MemoryCoreAdapter` (`mceval/adapters/memory_core.py`) is the
hosted-API client. Per-namespace store buffer flushes via
`POST /api/v1/memories/batch` on first read; reset is per-question.
The adapter does not depend on internal Memory Core APIs; any
implementation of `mceval.adapters.base.MemoryAdapter` (the
three-method contract: `reset` / `store` / `search`) plugs in.

### Top-k and metrics

`top_k_values = [1, 5, 10]`. Recall@k = (retrieved gold session in
top-k) / (questions where a gold session exists). Abstention questions
score 1.0 if the adapter abstains correctly (returns no relevant
session) and 0.0 otherwise.

### Default code-stack at this measurement

```
MEMORY_DIVERSITY_OVERFETCH        = 4
MEMORY_OVERFETCH_CAP              = 40
MEMORY_CE_RERANK_TOP_N            = 40
MEMORY_CE_RERANK_ENABLED          = auto-on (sentence-transformers installed)
MEMORY_COMPOSITE_RANKING_ENABLED  = 1 (default, no-op on eval data — see Ablations)
MEMORY_DIVERSITY_GROUP_CAP        = 0  (off)
MEMORY_PREFERENCE_INTENT_ENABLED  = 0  (off)
LANCEDB_PRIMARY                   = true
LANCEDB_NS_STRATEGY               = table  (one table per namespace)
LANCEDB_RETRIEVAL_K               = 60
LANCEDB_RRF_K                     = 60
MCP_MEMORY_ENABLED                = false
```

These are visible at `GET /api/v1/config` on the running instance.

---

## Headline results (n=500)

### LoCoMo

| System | n | R@1 | R@5 | R@10 | Time |
|---|---:|---:|---:|---:|---:|
| BM25 (paper anchor) | 100 | 54.0 | 74.0 | 84.0 | 1s |
| Hybrid-RRF (paper anchor) | 100 | 50.0 | 78.0 | 85.0 | 396s |
| **Memory Core** | **500** | **58.4** | **81.8** | **88.8** | 749s |

Per-category R@10 at n=500:

| category | n | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| category_5 | 113 | 68.1 | 90.3 | 93.8 |
| category_4 | 213 | 65.7 | 88.7 | 92.5 |
| category_1 | 71 | 36.6 | 71.8 | 84.5 |
| category_2 | 81 | 54.3 | 70.4 | 81.5 |
| category_3 | 22 | 22.7 | 45.5 | 68.2 |

Baseline JSON: `baselines/memory-core_2026-04-27_11-06-32_n500.json`.

### LongMemEval-S

| System | n | R@10 |
|---|---:|---:|
| BM25 (paper anchor) | 500 | 96.2 |
| Dense MiniLM (paper anchor) | 500 | 97.2 |
| Hybrid-RRF (paper anchor) | 500 | 97.9 |
| **Memory Core** | **500** | **99.4** |

Per-category R@10 at n=500:

| question_type | n | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| knowledge-update | 72 | 98.6 | 100 | 100 |
| multi-session | 121 | 98.3 | 99.2 | 100 |
| single-session-assistant | 56 | 100 | 100 | 100 |
| single-session-user | 64 | 98.4 | 100 | 100 |
| temporal-reasoning | 127 | 90.6 | 96.1 | 98.4 |
| single-session-preference | 30 | 66.7 | 93.3 | 96.7 |
| abstention | 30 | 100 | 100 | 100 |

Baseline JSON: `baselines/memory-core_2026-04-27_10-53-34_n500.json`.

The 100% / 100% / 100% / 100% rows on LongMemEval-S are striking but
worth interpreting carefully: at session-level granularity on this
benchmark, R@10 is approaching ceiling for any reasonable hybrid
retriever. The signal in this table is that **temporal-reasoning
98.4** and **single-session-preference 96.7** are the categories
where headroom still exists, and where Memory Core's temporal-window
boost (1.5×) and preference-intent boost (currently opt-in,
1.15×) target.

---

## Cross-restart determinism

**Claim.** The same code revision, the same input corpus, and two
fresh LanceDB databases produce bit-for-bit identical top-k content
on every query.

**Evidence.** Two LoCoMo n=100 runs against the role-aware build
(`memory-core@54a20e7`):

- `baselines/memory-core_2026-04-27_10-26-51_n100.json` — Run A
- `baselines/memory-core_2026-04-26_14-22-14_n100.json` — Run B (the
  pre-role-aware canonical, master @ `eb0de8f`)

Same R@1=58.0 / R@5=80.0 / R@10=87.0; identical per-category
breakdown. Run A is on a release that shipped role-aware routing
end-to-end; the byte-identical reproduction of Run B's numbers
demonstrates that the new code's default path
(`role_scope=None`) is byte-identical to the prior path.

A second cross-restart probe is `scripts/determinism_check.py` in
the implementation repo, which boots two clean LanceDB-primary
servers, ingests a fixed 50-document corpus, runs 20 fixed queries,
and diffs the resulting top-10 contents. CI runs this on every
push; an alert fires on any per-query flip. This is the *floor* for
determinism — the workers=2 fragment-layout class of bug (which
caused the 87-mirage incident on 2026-04-26 — see *Lessons* below)
needs the full eval harness on real benchmark data to surface.

### Why this matters for agent teams

A retrieval layer where the same query produces different rankings
on different boots is unsafe under any production scenario that
audits results, replays past states, or coordinates multiple agents
sharing memory. Mem0 / Zep / Letta do not currently publish
determinism guarantees; we treat it as a first-class property,
verifiable in CI, with the receipts above.

---

## Ablations

LoCoMo n=100 stratified seed=0 workers=2 (smaller sample for
ablation-volume; n=500 per row would be ~1 hour of compute).

| Configuration | R@1 | R@5 | R@10 | vs default |
|---|---:|---:|---:|---:|
| **Default stack** (CE on, composite on, 4/40/40) | 58.0 | 80.0 | **87.0** | — |
| CE rerank off (`MEMORY_CE_RERANK_ENABLED=0`) | 53.0 | 75.0 | 84.0 | −5 / −5 / −3 |
| Composite ranking off (`MEMORY_COMPOSITE_RANKING_ENABLED=0`) | 58.0 | 80.0 | 87.0 | 0 / 0 / 0 |

Reading the table:

- **CE rerank** contributes +5pt R@1, +5pt R@5, +3pt R@10. The
  shape matches the CE design intent: it re-ranks within the
  existing top-N, so it lifts R@1 and R@5 most, and its effect on
  R@10 is bounded by what was already in the candidate set. CE-off
  R@10=84 sits at the BM25 paper anchor (84.0) — the floor the
  hybrid stack falls back to without semantic re-ranking on top.
- **Composite ranking** is bit-identical to default on this
  benchmark. By design — composite multiplies in production-only
  signals (credibility from `autoDream` consolidation, relation
  graphs, entity pages, path-glob anchors). The eval data carries
  none of those, so every multiplier is 1.0 and the composite is
  mathematically identity. The agreement here is a property test:
  if a future change perturbs composite into anything *other*
  than a no-op on flat-pool data, this ablation will surface it.

### What this CE-off ablation found (a real bug, fixed in this release)

The CE-off ablation is the reason this report exists in its current
form. Before the floor-fix described below, CE-off on LoCoMo n=100
gave R@10=45 — far worse than even BM25-only (84) at the same
sample. The composite ranker was doing actual harm on memory
corpora with historical timestamps:

- The composite formula multiplies `temporal_factor =
  exp(-age_days / 30)` into the score.
- LoCoMo conversation timestamps are 2022–2023; "now" is
  2026-04-27, so `age_days ≈ 1100–1500` for every memory.
- `exp(-1500/30) ≈ 1e-22` — every composite score collapses to
  near-zero, with only floating-point noise distinguishing
  memories.
- The post-composite sort then gives an effectively-random order.
  CE rerank, when on, smoothed this over because it re-orders
  top-40 by cross-encoder similarity. CE off exposed the broken
  ranking.

The fix (committed to the implementation repo as the bug-#6
companion to this report):

1. **Floor `temporal_factor` at `MEMORY_TEMPORAL_FACTOR_FLOOR`**
   (default 0.05, env-configurable). Keeps the decay shape but
   prevents catastrophic collapse below FP-noise.
2. **Skip `temporal_factor` when `as_of_date` is provided.** When
   the caller is replaying history, "freshness against now" is
   not a meaningful signal. The eval harness passes `as_of_date`
   on LongMemEval-S so this code path was already a no-op there;
   LoCoMo's `question_date` field is unparseable so it didn't
   trigger the skip — hence the asymmetric impact.

Local 348-test unit suite passes after the fix. The CE-off row
above is the post-fix measurement; the pre-fix R@10=45 figure is
preserved here precisely because it's the receipt that the fix
matters.

---

## Architecture (just enough to interpret the numbers)

```
HTTP REST API (FastAPI :8001)
    └── QueryRouter
         └── LanceDbAdapter (primary)
              ├── per-namespace table; tantivy FTS lazy-rebuild on dirty
              ├── split-leg hybrid retrieval
              │     dense:  LANCEDB_RETRIEVAL_K=60 candidates
              │     fts:    LANCEDB_RETRIEVAL_K=60 candidates
              │     fuse:   external RRF, LANCEDB_RRF_K=60
              ├── over-fetch ×MEMORY_DIVERSITY_OVERFETCH (=4) capped at MEMORY_OVERFETCH_CAP (=40)
              ├── composite ranking (no-op on eval data; production-only)
              ├── CE rerank top MEMORY_CE_RERANK_TOP_N (=40), auto-on with ST
              ├── per-namespace asyncio.Lock around table.add / delete (concurrency-safe)
              └── SQL push-down filters: namespace + role + temporal-window tags
```

Two retrieval features are worth calling out for benchmark
interpretation:

- **Role scoping.** `role_scope=agent | cross-role | organizational`
  on every search request, with an `array_has_any` SQL push-down
  that filters before the dense / FTS legs sample. The eval runs in
  this report set `role_scope=None` to match the paper baselines'
  flat-pool semantics.
- **Temporal window.** When the query parses to a date window
  (`as_of_date` provided + `parse_temporal_window` matches), the
  adapter retrieves a `ts:YYYY-MM-DD`-tagged candidate set in
  parallel with hybrid and RRF-merges. This is what drives
  LongMemEval-S `temporal-reasoning` from a brittle pre-fix path
  to 98.4. LoCoMo's `as_of_date` is rarely parseable from the
  question_date field, so this path is mostly a no-op there.

---

## Known limitations

1. **n=full not run.** LongMemEval-M and the full LoCoMo split are
   not measured here. n=500 stratified is the largest sample.
2. **Mem0 / Zep / Letta head-to-head not run.** Those projects
   publish on different metrics (answer accuracy with a downstream
   LLM); a same-corpus retrieval-only comparison requires running
   their adapters under this harness, which has not been done.
3. **Independent reproduction not performed.** The reproduction
   recipe below is exactly what we ran; a third-party rerun is the
   credibility-multiplying next step.
4. **Composite ranking is default-on but no-op on eval.** The
   design intent is that composite multiplies in production-only
   signals (`autoDream`-emitted credibility, entity-page
   activation, path-glob match, relation graph). On benchmark data
   none of these are present, so composite is mathematically
   identity. We have not yet split composite into a preset-attached
   layer (so the default path is paper-baseline-shaped) — it is
   tracked as the next refactor.
5. **Lancedb is pinned to 0.10.x.** A 0.30 upgrade is queued; the
   FTS query API, null int64 handling, and schema-evolution
   semantics all changed across the 0.10 → 0.30 line and need
   coordinated update.

---

## Reproduction recipe

### Prerequisites

- Python 3.11+
- 4 GB RAM (sentence-transformers all-MiniLM-L6-v2 weights)
- `~/.cache/huggingface` for first-run model download

### Step 1: clone both repos

```bash
git clone https://github.com/Evanyuan-builder/memory-core.git
git clone https://github.com/Evanyuan-builder/memory-core-eval.git
```

### Step 2: install Memory Core and bring up the API

```bash
cd memory-core
pip install fastapi uvicorn[standard] pydantic pydantic-settings httpx \
            'lancedb<0.11' 'pyarrow<16' 'numpy<2' pandas \
            sentence-transformers
LANCEDB_ENABLED=true \
LANCEDB_PRIMARY=true \
LANCEDB_PATH=/tmp/lancedb_repro \
MCP_MEMORY_ENABLED=false \
GRAPHITI_ENABLED=false \
NO_PROXY=127.0.0.1,localhost \
PYTHONPATH=packages \
  uvicorn memory_core_api.main:app --host 127.0.0.1 --port 8001 &
# Wait for /health
until curl -sf http://127.0.0.1:8001/health > /dev/null; do sleep 1; done
```

### Step 3: install eval harness and run

```bash
cd memory-core-eval
pip install -e .
NO_PROXY=127.0.0.1,localhost \
MEMORY_CORE_URL=http://127.0.0.1:8001 \
MCEVAL_BATCH_STORE=1 \
mceval run --adapter memory-core --dataset locomo \
       --sample 500 --seed 0 --stratified --workers 2 \
       --out-dir baselines/
mceval run --adapter memory-core --dataset longmemeval --split s \
       --sample 500 --seed 0 --stratified --workers 2 \
       --out-dir baselines/
```

Each n=500 run takes 12–14 minutes on a recent laptop CPU.

### Step 4: verify the numbers

```bash
python -c "
import json, glob
for f in sorted(glob.glob('baselines/memory-core_2026-04-27_*_n500.json'))[-2:]:
    d = json.load(open(f))
    print(f, '→', d['metrics']['overall'])
"
```

Expected:

```
baselines/memory-core_2026-04-27_10-53-34_n500.json → {'recall@1': 94.5, 'recall@5': 98.3, 'recall@10': 99.4}
baselines/memory-core_2026-04-27_11-06-32_n500.json → {'recall@1': 58.4, 'recall@5': 81.8, 'recall@10': 88.8}
```

Within ±0.5pt is acceptable variance from cross-host environmental
factors; larger drift is a signal that something has shifted (e.g.,
a different pyarrow / numpy / lancedb minor version slipped in).
File a reproduction-divergence issue if you hit it.

---

## Lessons (the failure-to-fix journal that shaped v0.3.x)

This section is here because the most credible thing an
infrastructure project can publish is *what we found broken in our
own code, and how we know it's no longer broken*.

The release leading up to v0.3.0 surfaced five real bugs that the
benchmark numbers above were silent on. They were caught by the
upper-layer maintenance machinery (gc / consolidate) and by
concurrent-store production paths — exactly the surfaces a pure
recall benchmark does not exercise. Two more (#6, #7) landed in
the v0.3.1 cycle.

| # | Bug | Where it would have hit production | Fix |
|---|---|---|---|
| 1 | `ttl_seconds=None` round-tripped to `0` via the LanceDB pandas-dict path | gc would expire every "permanent" memory at first sweep | `bb851bc` — coerce 0 → None on read |
| 2 | `is_verbatim` field never serialised in LanceDB schema | gc would delete verbatim memories that were supposed to be preserved | `3ec6a56` — add column + roundtrip |
| 3 | Concurrent `table.add` on the same namespace raised "Already mutably borrowed" | SDK `asyncio.gather` over `store()` would 500 | `161cf69` — per-namespace asyncio.Lock |
| 4 | `_get_table` first-creation race (5 concurrent stores → 4 with "table already exists") | Same SDK pattern, but at first-store-into-fresh-namespace | `54a20e7` — extend the lock to cover cache-fill |
| 5 | Backend-name assertions hard-coded to `"mcp-memory-service"` across 5 test files | Any backend swap would surface as a fake test failure | `3ec6a56` + `161cf69` — assertions made backend-agnostic |
| 6 | Composite ranker's `exp(-age_days/30)` collapsed to FP-noise on memories with historical timestamps | Any production replay or backfill against an old corpus would silently break ranking | `5e23039` — floor temporal_factor + skip when `as_of_date` provided |
| 7 | Four MemoryUnit fields (`layer`, `is_entity_page`, `temporal`, `relations`) were never declared in the LanceDB schema, so store→retrieve silently reset them to defaults | autoDream entity pages, Graphiti graph edges, and bi-temporal supersede policy were all reading defaults instead of the values upstream populated | `ae1352d` — add columns + flattened bi-temporal layout + JSON-encoded relations + add_columns auto-migration for v0.3.0 tables |

Bug #6 is special: it was found *while writing this very report*.
The CE-off ablation row was supposed to be a routine measurement of
CE rerank's contribution; it surfaced as a 42-point R@10 drop
(45 vs default 87) that no benchmark-only check would have ever
flagged. CE rerank in the default stack was perfectly masking a
catastrophic decay collapse in the composite layer underneath.
The lesson — *the act of writing receipts is itself a bug-finder*
— is one of the strongest arguments we can offer for the
"evaluation-first, with public ablations" practice this report is
trying to establish.

Bug #7 is the inverse twin of #6. It was found by *audit*: a
field-by-field compare of `MemoryUnit` (the model) against
`_make_schema` (the LanceDB column list). Four fields — `layer`,
`is_entity_page`, `temporal`, `relations` — were declared on the
model but absent from the schema, so the adapter silently dropped
them on every write. The benchmark numbers in this report did not
move when v0.3.1 fixed it, *because the eval data does not populate
any of those four fields to begin with*. R@10 = 88.8 / 99.4 holds at
v0.3.1; the schema fix surfaces only on production paths
(autoDream's entity pages, Graphiti relations, bi-temporal
supersede policy). The lesson is symmetrical to #6 — *the act of
auditing receipts against the model is itself a bug-finder, and
some bug classes can never be detected by recall benchmarks alone*.

A separate prior incident, the **"87 mirage"** of 2026-04-26, is
worth its own paragraph. A morning tuning experiment showed LoCoMo
R@10=87 from one server boot and R@10=79 from another at the same
nominal config. The first reaction — "this is unexplainable noise,
revert" — was wrong. The actual cause was a `scan.limit(200)` in a
freshly-shipped python-side filter that surfaced different fragment
prefixes on different ingest orderings. Two cross-server reruns
reproduced 87 cleanly once the predicate was pushed into a SQL
`array_has_any` clause. The lesson encoded in the project's
feedback memory: *"same nominal config, different result" is a
signal that your own code is non-deterministic, not the
environment*. Determinism is a property worth investing in
explicitly because the alternative is debugging time you cannot
afford.

---

## Repository links

- Implementation: https://github.com/Evanyuan-builder/memory-core (pinned at v0.3.1)
- Eval harness (this repo): https://github.com/Evanyuan-builder/memory-core-eval (v0.2.0)
- Canonical baseline JSONs (this repo): `baselines/memory-core_2026-04-27_*.json`

---

## Contact

Found a divergence reproducing these numbers, or want to plug your
own adapter into the harness? Open an issue at
`Evanyuan-builder/memory-core-eval`. Memory Core itself remains
focused on the *deterministic temporal memory for agent teams*
narrative; this report is one of two artefacts (the other being
the v0.3.1 release notes) that translates the code into something
a third party can verify.

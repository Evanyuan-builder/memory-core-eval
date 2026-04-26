# memory-core-eval

**Reproducible evaluation harness for agent memory systems.**

This repository lets anyone benchmark a memory system — a BM25 or hybrid
baseline, Memory Core, or a custom adapter — against the LongMemEval and
LoCoMo retrieval benchmarks, and produce comparable, auditable results.

It is the **open verification layer** of the
[Memory Core](https://github.com/Evanyuan-builder/memory-core) project. The
goal is simple: anyone should be able to reproduce a score, plug in their own
system, and compare head-to-head without trusting anyone's marketing.

---

## What this is / is not

**Is:**
- An eval harness with a stable `MemoryAdapter` interface.
- Built-in baselines: BM25, dense (sentence-transformers), BM25+dense RRF
  hybrid — the paper baselines.
- Two peer adapters for context: Hindsight (Vectorize AI) and m_flow
  (FlowElement-ai).
- A Memory Core adapter that talks to a self-hosted instance over HTTP.
- A reproducibility contract: pinned dataset revision + hash, deterministic
  ordering, full traces.

**Is not:**
- The Memory Core engine. Retrieval, ranking, and consolidation live in the
  main Memory Core repo.
- An end-to-end QA benchmark. This measures retrieval (Recall@k), not answer
  generation.

---

## Install

`memory-core-eval` is not on PyPI yet — install editable from source:

~~~bash
git clone https://github.com/Evanyuan-builder/memory-core-eval.git
cd memory-core-eval
pip install -e .                  # core + BM25 baseline
pip install -e ".[dense]"         # + sentence-transformers for dense / hybrid
pip install -e ".[dev]"           # + pytest, ruff
~~~

Hindsight and m_flow adapters require their upstream client packages
(`hindsight-client`, `m_flow`); install separately if you want to run those
peers.

---

## Quick start

~~~bash
# BM25 baseline on a 20-question stratified sample
mceval run --adapter bm25 --sample 20 --seed 0 --stratified

# LongMemEval session-haystack split
mceval run --adapter bm25 --dataset longmemeval --split s --sample 100

# LoCoMo (long-range conversational memory)
mceval run --adapter bm25 --dataset locomo --sample 100

# Memory Core against a self-hosted instance
mceval run --adapter memory-core --base-url http://localhost:8001 --sample 100

# Head-to-head comparison
mceval compare --adapters bm25,dense,hybrid-rrf,memory-core \
  --base-url http://localhost:8001 --sample 100
~~~

---

## Latest results

Paper baselines are included as anchors, **not apples-to-apples leaderboard
claims** — sample sizes and harness versions differ. They are the strongest
published numbers we know of for the same datasets, included so a reader can
position the current run within that landscape.

**LoCoMo** (Maharana et al. 2024) — long-range conversational memory.
Session-level Recall@k. n=100 stratified, seed=0, top_k=10:

| System | n | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| BM25 (paper anchor) | 100 | 54.0 | 74.0 | 84.0 |
| Hybrid-RRF (paper anchor) | 100 | 50.0 | 78.0 | 85.0 |
| **Memory Core (current run)** | **100** | **57.0** | **80.0** | **87.0** |

**LongMemEval-S** (Wu et al. 2024) — session-haystack
(~50 sessions / question). The Memory Core run is n=100 stratified; the paper
anchors are at n=500. Treat the gap as suggestive until the larger sweep lands.

| System | n | R@10 |
|---|---:|---:|
| BM25 (paper anchor) | 500 | 96.2 |
| Hybrid-RRF (paper anchor) | 500 | 97.9 |
| **Memory Core (current run)** | **100** | **98.9** |

Cross-restart stability is verified in reference benchmark runs. Canonical
reference JSONs live under `baselines/`.

LongMemEval-M and full n=500 sweeps are queued.

---

## Datasets

- **LongMemEval** (`xiaowu0162/longmemeval-cleaned` on HuggingFace) — three
  haystack splits via `--split`:
    - `oracle` (default): only evidence sessions, saturates at top.
    - `s`: ~50 sessions / question, the discriminative split.
    - `m`: long-horizon, multi-month haystack.
- **LoCoMo** (`snap-research/locomo10.json`) — 10 conversations × ~30 sessions
  × ~600 turns, ~200 QA pairs each. Session-level Recall@k (looser than the
  paper's dia_id-level metric, but apples-to-apples with LongMemEval here).

Both loaders pin a dataset revision + content hash for reproducibility.

---

## Adapter inventory

Built into the harness:

| Adapter | Role | Extra deps |
|---|---|---|
| `bm25` | Paper baseline (rank-bm25) | core only |
| `dense` | Paper baseline (sentence-transformers MiniLM) | `[dense]` |
| `hybrid-rrf` | Paper baseline (BM25 + dense, RRF k=60) | `[dense]` |
| `memory-core` | Memory Core HTTP client | core only |
| `hindsight` | Vectorize AI peer | `hindsight-client` (separate install) |
| `mflow` | FlowElement-ai m_flow peer | `m_flow` (separate install) |

---

## Writing an adapter

Implement four methods on the `MemoryAdapter` protocol:

~~~python
from datetime import datetime
from mceval.adapters.base import MemoryAdapter, Turn, Memory

class MyAdapter:
    name = "my-system"

    def reset(self, namespace: str) -> None: ...
    def store(self, namespace: str, turn: Turn) -> str: ...                # returns memory id
    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,                                # for relative-time queries
    ) -> list[Memory]: ...
~~~

`as_of_date` is the reference time for resolving phrases like "yesterday"
or "last Tuesday" against a stored timeline. Adapters that don't model time
can ignore it.

Run the contract tests against your adapter:

~~~bash
pytest tests/test_adapter_contract.py -k my_system
~~~

---

## Diagnostic tool

For investigating why a specific question lands or misses against a baseline,
use the A/B probe:

~~~bash
MEMORY_CORE_URL=http://127.0.0.1:8001 \
    python -m mceval.diagnose.ab conv-41:q19 conv-42:q186
~~~

Stores the same haystack in `memory-core` and `hybrid-rrf`, runs the question
through both, and reports where the gold session lands in each top-K. Tells
you whether a gap is upstream of ranking (retrieval candidate pool) or
downstream (rank ordering).

---

## Reproducibility

Each run pins:
- Dataset revision hash (HuggingFace dataset for LongMemEval; commit-pinned
  URL for LoCoMo).
- SHA-256 of the downloaded file.
- Adapter name and harness version.
- Full per-question trace (question → stored turns → search results →
  verdict) as JSONL when `--trace` is passed.

Canonical baseline JSONs (paper anchors, the current Memory Core canonical
state, and cross-restart determinism evidence) live under `baselines/`.

## License

Apache-2.0. See `LICENSE`.

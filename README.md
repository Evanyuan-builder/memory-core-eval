# memory-core-eval

**Reproducible evaluation harness for agent memory systems.**

This repository lets anyone benchmark a memory system (Memory Core, Mem0, Letta/MemGPT, Zep, a BM25 baseline, or their own) against [LongMemEval](https://arxiv.org/abs/2410.10813) and produce comparable, auditable results.

It is the **open verification layer** of the Memory Core project. The goal is simple:
anyone should be able to reproduce a score, plug in their own system, and compare head-to-head without trusting anyone's marketing.

---

## What this is / is not

**Is:**
- An eval harness with a stable `MemoryAdapter` interface
- Built-in baselines (BM25, dense, BM25+dense RRF hybrid — the paper baselines)
- A Memory Core adapter (default: hosted API; also supports self-hosted URL)
- A reproducibility contract (pinned dataset revision + hash, deterministic ordering, full traces)

**Is not:**
- The Memory Core engine (retrieval / ranking / consolidation stays closed)
- An end-to-end QA benchmark — this measures retrieval (Recall@k), not answer generation

---

## Install

```bash
pip install memory-core-eval                     # core + BM25 baseline
pip install "memory-core-eval[dense]"            # + sentence-transformers for dense/hybrid
pip install "memory-core-eval[extras]"           # + mem0 / letta adapters
```

## Quick start

```bash
# Run BM25 baseline on a 20-question sample
mceval run --adapter bm25 --sample 20

# Run Memory Core against hosted API (default)
mceval run --adapter memory-core --sample 20

# Self-hosted Memory Core (compatible — same adapter, different URL)
mceval run --adapter memory-core --base-url http://localhost:8001 --sample 20

# Head-to-head
mceval compare --adapters bm25,dense,hybrid-rrf,memory-core --sample 100
```

## Writing an adapter

Implement three methods:

```python
from mceval.adapters.base import MemoryAdapter, Turn, Memory

class MyAdapter:
    name = "my-system"

    def reset(self, namespace: str) -> None: ...
    def store(self, turn: Turn) -> str: ...                       # returns memory_id
    def search(self, query: str, namespace: str, top_k: int) -> list[Memory]: ...
```

Register it, run `pytest tests/test_adapter_contract.py` against it, and it's eligible for the leaderboard. See `docs/writing_adapter.md`.

## Reproducibility

Each run pins:
- Dataset revision hash (HuggingFace `xiaowu0162/longmemeval-cleaned`)
- SHA-256 of the downloaded file
- Adapter version
- Full trace (question → stored turns → search results → verdict) as JSONL

Sample traces for 50 audit questions live in `traces/sample_50/`. Full n=500 traces are attached to releases.

## License

Apache-2.0. See `LICENSE`.

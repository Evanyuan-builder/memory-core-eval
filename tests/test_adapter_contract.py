"""Contract test — every MemoryAdapter must pass this.

The eval harness will refuse to run an adapter that doesn't. To add your
adapter to the matrix, append its factory to ``ADAPTER_FACTORIES`` below (or
parametrize via a fixture in your own test file).
"""
from __future__ import annotations

import os
import uuid

import httpx
import pytest

from mceval.adapters.base import Memory, MemoryAdapter, Turn
from mceval.adapters.bm25_baseline import BM25BaselineAdapter
from mceval.adapters.memory_core import MemoryCoreAdapter

try:
    from mceval.adapters.dense_baseline import DenseBaselineAdapter, _DENSE_AVAILABLE
    from mceval.adapters.hybrid_rrf_baseline import HybridRRFBaselineAdapter
except ImportError:
    DenseBaselineAdapter = None  # type: ignore
    HybridRRFBaselineAdapter = None  # type: ignore
    _DENSE_AVAILABLE = False


def _memory_core_available() -> bool:
    """Return True iff a Memory Core API is reachable at MEMORY_CORE_URL (or
    the default self-hosted URL). Contract tests skip when it isn't.
    """
    url = os.getenv("MEMORY_CORE_URL", "http://127.0.0.1:8001")
    try:
        return httpx.get(f"{url}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


ADAPTER_FACTORIES = [
    pytest.param(BM25BaselineAdapter, id="bm25"),
    pytest.param(
        DenseBaselineAdapter,
        id="dense",
        marks=pytest.mark.skipif(
            not _DENSE_AVAILABLE,
            reason="sentence-transformers not installed; pip install '.[dense]'",
        ),
    ),
    pytest.param(
        HybridRRFBaselineAdapter,
        id="hybrid-rrf",
        marks=pytest.mark.skipif(
            not _DENSE_AVAILABLE,
            reason="sentence-transformers not installed; pip install '.[dense]'",
        ),
    ),
    pytest.param(
        MemoryCoreAdapter,
        id="memory-core",
        marks=pytest.mark.skipif(
            not _memory_core_available(),
            reason="Memory Core API not reachable; set MEMORY_CORE_URL to enable",
        ),
    ),
]


@pytest.fixture(params=ADAPTER_FACTORIES)
def adapter(request):
    return request.param()


@pytest.fixture
def ns(adapter) -> str:
    """Unique namespace per test to keep runs independent of server state."""
    name = "mceval-test-" + uuid.uuid4().hex[:12]
    adapter.reset(name)
    return name


def _unique(label: str) -> str:
    """Content string that won't collide with prior test runs (important for
    backends that globally dedup on content hash, e.g. mcp-memory-service).
    """
    return f"{label} [{uuid.uuid4().hex[:8]}]"


def _turn(i: int, content: str, role: str = "user") -> Turn:
    return Turn(
        content=content,
        role=role,
        session_id="sess-test",
        turn_idx=i,
        session_idx=0,
    )


def test_satisfies_protocol(adapter):
    assert isinstance(adapter, MemoryAdapter), (
        f"{type(adapter).__name__} does not satisfy the MemoryAdapter Protocol"
    )
    assert adapter.name, "adapter must declare a non-empty name"


def test_store_returns_id(adapter, ns):
    mem_id = adapter.store(ns, _turn(0, _unique("Alice likes espresso.")))
    assert mem_id, "store() must return a non-empty memory id"


def test_search_returns_relevant_first(adapter, ns):
    alice = _unique("Alice likes espresso.")
    bob = _unique("Bob went hiking in the Alps.")
    paris = _unique("The weather in Paris was rainy.")
    adapter.store(ns, _turn(0, alice))
    adapter.store(ns, _turn(1, bob))
    adapter.store(ns, _turn(2, paris))

    results = adapter.search(ns, "what does Alice drink?", top_k=3)
    assert results, "search must return at least one result"
    assert all(isinstance(r, Memory) for r in results)
    top = results[0].content.lower()
    assert "alice" in top or "espresso" in top


def test_search_result_shape(adapter, ns):
    content = _unique("hello world")
    adapter.store(ns, _turn(0, content))
    results = adapter.search(ns, "hello", top_k=1)
    assert len(results) >= 1
    r = results[0]
    assert r.id
    assert r.content
    assert r.score is not None
    assert r.session_id == "sess-test"
    assert r.turn_idx == 0


def test_reset_clears_namespace(adapter, ns):
    adapter.store(ns, _turn(0, _unique("to be cleared")))
    adapter.reset(ns)
    results = adapter.search(ns, "cleared", top_k=5)
    assert results == [], "reset() must remove all memories in the namespace"


def test_namespaces_isolated(adapter):
    ns_a = "mceval-test-" + uuid.uuid4().hex[:12]
    ns_b = "mceval-test-" + uuid.uuid4().hex[:12]
    adapter.reset(ns_a)
    adapter.reset(ns_b)
    apple = _unique("apple in namespace A")
    banana = _unique("banana in namespace B")
    adapter.store(ns_a, _turn(0, apple))
    adapter.store(ns_b, _turn(0, banana))

    r_a = adapter.search(ns_a, "apple", top_k=5)
    assert r_a
    assert "apple" in r_a[0].content.lower()
    assert all("banana" not in r.content.lower() for r in r_a)
    adapter.reset(ns_a)
    adapter.reset(ns_b)

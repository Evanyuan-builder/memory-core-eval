"""Dense retrieval baseline — sentence-transformers + numpy cosine similarity.

Uses ``all-MiniLM-L6-v2`` by default (matches the LongMemEval paper's dense
baseline and the mcp-memory-service backend). Per-namespace in-memory index;
no FAISS required at eval-scale corpus sizes.

Extras: ``pip install 'memory-core-eval[dense]'``
"""
from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from datetime import datetime

from .base import Memory, Turn

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _DENSE_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    _DENSE_AVAILABLE = False


class DenseBaselineAdapter:
    name = "dense"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not _DENSE_AVAILABLE:
            raise ImportError(
                "DenseBaselineAdapter requires sentence-transformers. "
                "Install with: pip install 'memory-core-eval[dense]'"
            )
        self._model = SentenceTransformer(model)
        self._corpus: dict[str, list[tuple[str, str, Turn]]] = defaultdict(list)
        self._vecs: dict[str, "np.ndarray"] = {}
        self._lock = threading.Lock()

    def reset(self, namespace: str) -> None:
        with self._lock:
            self._corpus.pop(namespace, None)
            self._vecs.pop(namespace, None)

    def store(self, namespace: str, turn: Turn) -> str:
        mem_id = uuid.uuid4().hex
        vec = self._model.encode([turn.content], normalize_embeddings=True)[0].astype(np.float32)
        with self._lock:
            self._corpus[namespace].append((mem_id, turn.content, turn))
            existing = self._vecs.get(namespace)
            if existing is None:
                self._vecs[namespace] = vec.reshape(1, -1)
            else:
                self._vecs[namespace] = np.vstack([existing, vec])
        return mem_id

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int,
        as_of_date: datetime | None = None,
    ) -> list[Memory]:
        del as_of_date
        with self._lock:
            entries = list(self._corpus.get(namespace, []))
            vecs = self._vecs.get(namespace)
        if vecs is None or not entries:
            return []

        q = self._model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = vecs @ q                                # cosine via normalized dot-product
        ranked = np.argsort(-scores)[:top_k]

        results: list[Memory] = []
        for i in ranked:
            mem_id, content, turn = entries[int(i)]
            results.append(Memory(
                id=mem_id,
                content=content,
                score=float(scores[int(i)]),
                session_id=turn.session_id,
                turn_idx=turn.turn_idx,
                session_idx=turn.session_idx,
                metadata={"role": turn.role},
            ))
        return results

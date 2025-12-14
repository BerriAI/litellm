"""FAISS-backed vector store for semantic MCP filtering."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

_faiss_error: Optional[Exception] = None
try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None
    _faiss_error = exc
else:
    _faiss_error = None

from litellm._logging import verbose_proxy_logger

from ..settings import SemanticFilterVectorStoreConfig
from .base import ToolVectorRecord, ToolVectorStore


class FaissVectorStore(ToolVectorStore):
    def __init__(self, config: SemanticFilterVectorStoreConfig):
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed; install litellm[semantic-router] or faiss-cpu"
            ) from _faiss_error

        self._config = config
        self._dimension = config.dimension
        self._metric = config.metric
        self._vectors: dict[str, np.ndarray] = {}
        self._id_list: List[str] = []
        self._index = None

    def replace_records(self, records: Iterable[ToolVectorRecord]) -> None:
        self._vectors = {}
        self.upsert_records(records)
        if not self._vectors:
            self._rebuild_index()

    def upsert_records(self, records: Iterable[ToolVectorRecord]) -> None:
        changed = False
        for record in records:
            prepared = self._prepare_vector(record.vector)
            self._vectors[record.tool_id] = prepared
            changed = True
        if changed:
            self._rebuild_index()

    def remove_records(self, tool_ids: Iterable[str]) -> None:
        removed = False
        for tool_id in tool_ids:
            if tool_id in self._vectors:
                del self._vectors[tool_id]
                removed = True
        if removed:
            self._rebuild_index()

    def query(self, vector: Sequence[float], top_k: int) -> List[Tuple[str, float]]:
        if not self._index or not self._vectors:
            return []

        query_vec = self._prepare_vector(vector)
        query_vec = np.expand_dims(query_vec, axis=0)
        scores, indices = self._index.search(query_vec, min(top_k, len(self._vectors)))
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            results.append((self._id_list[idx], float(score)))
        return results

    def _prepare_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype="float32").flatten()
        if arr.size == 0:
            raise ValueError("Empty embedding vector")
        if self._dimension is None:
            self._dimension = arr.size
        if arr.size != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self._dimension}, got {arr.size}"
            )
        if self._metric == "cosine":
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr

    def _rebuild_index(self) -> None:
        if not self._vectors:
            self._index = None
            self._id_list = []
            return

        matrix = np.stack(list(self._vectors.values()), axis=0)
        metric = self._metric
        if metric == "ip" or metric == "cosine":
            index = faiss.IndexFlatIP(self._dimension)
        else:
            index = faiss.IndexFlatL2(self._dimension)

        index.add(matrix)
        self._index = index
        self._id_list = list(self._vectors.keys())

        verbose_proxy_logger.debug(
            "Rebuilt FAISS index with %s vectors", len(self._id_list)
        )

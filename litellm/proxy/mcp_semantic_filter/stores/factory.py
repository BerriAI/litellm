"""Factory helpers for vector store backends."""

from __future__ import annotations

from typing import Optional

from litellm._logging import verbose_proxy_logger

from ..settings import SemanticFilterVectorStoreConfig
from .base import ToolVectorStore
from .faiss_store import FaissVectorStore


def create_vector_store(
    config: SemanticFilterVectorStoreConfig,
) -> Optional[ToolVectorStore]:
    if config.backend == "faiss":
        try:
            return FaissVectorStore(config)
        except Exception:
            verbose_proxy_logger.exception("Failed to initialize FAISS store")
            return None

    verbose_proxy_logger.warning(
        "Unsupported semantic MCP filter backend %s", config.backend
    )
    return None

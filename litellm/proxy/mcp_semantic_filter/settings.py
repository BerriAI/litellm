"""Pydantic models for semantic MCP filter configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SemanticFilterEmbeddingConfig(BaseModel):
    model: str = Field(
        ..., description="Embedding model name passed to litellm.embedding"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific params forwarded to litellm.embedding",
    )


class SemanticFilterVectorStoreConfig(BaseModel):
    backend: Literal["faiss"] = "faiss"
    metric: Literal["ip", "l2", "cosine"] = "ip"
    top_k: int = Field(default=5, ge=1)
    dimension: Optional[int] = Field(
        default=None,
        ge=1,
        description="Expected embedding dimension. Inferred on first build when omitted.",
    )


class SemanticFilterConfig(BaseModel):
    enabled: bool = False
    embedding: SemanticFilterEmbeddingConfig
    vector_store: SemanticFilterVectorStoreConfig = Field(
        default_factory=SemanticFilterVectorStoreConfig
    )
    include_servers: Optional[List[str]] = Field(
        default=None,
        description="If provided, only tools from these server labels are indexed.",
    )

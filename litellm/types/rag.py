"""
Type definitions for RAG (Retrieval Augmented Generation) Ingest API.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class RAGChunkingStrategy(TypedDict, total=False):
    """
    Chunking strategy config for RAG ingest using RecursiveCharacterTextSplitter.

    See: https://docs.langchain.com/oss/python/langchain/rag
    """

    chunk_size: int  # Maximum size of chunks (default: 1000)
    chunk_overlap: int  # Overlap between chunks (default: 200)
    separators: Optional[List[str]]  # Custom separators for splitting


class RAGIngestOCROptions(TypedDict, total=False):
    """OCR configuration for RAG ingest pipeline."""
    model: str  # e.g., "mistral/mistral-ocr-latest"


class RAGIngestEmbeddingOptions(TypedDict, total=False):
    """Embedding configuration for RAG ingest pipeline."""

    model: str  # e.g., "text-embedding-3-small"


class RAGIngestVectorStoreOptions(TypedDict, total=False):
    """Vector store configuration for RAG ingest pipeline."""

    custom_llm_provider: str  # e.g., "openai", "bedrock", "vertex_ai"
    vector_store_id: Optional[str]  # Existing vector store ID, or None to create new
    ttl_days: Optional[int]  # Time-to-live in days for indexed content


class RAGIngestOptions(TypedDict, total=False):
    """
    Combined options for RAG ingest pipeline.

    Example:
        {
            "name": "my-pipeline",
            "ocr": {"model": "mistral/mistral-ocr-latest"},
            "chunking_strategy": {"chunk_size": 1000, "chunk_overlap": 200},
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": {"custom_llm_provider": "openai", "vector_store_id": "vs_xxx"}
        }
    """

    name: Optional[str]  # Optional pipeline name for logging
    ocr: Optional[RAGIngestOCROptions]  # Optional OCR step
    chunking_strategy: Optional[RAGChunkingStrategy]  # RecursiveCharacterTextSplitter args
    embedding: Optional[RAGIngestEmbeddingOptions]  # Embedding model config
    vector_store: RAGIngestVectorStoreOptions  # Required: vector store config

class RAGIngestResponse(TypedDict, total=False):
    """Response from RAG ingest API."""

    id: str  # Unique ingest job ID
    status: Literal["completed", "in_progress", "failed"]
    vector_store_id: str  # The vector store ID (created or existing)
    file_id: Optional[str]  # The file ID in the vector store



class RAGIngestRequest(BaseModel):
    """Request body for RAG ingest API (for validation)."""

    file_url: Optional[str] = None  # URL to fetch file from
    file_id: Optional[str] = None  # Existing file ID
    ingest_options: Dict[str, Any]  # RAGIngestOptions as dict for flexibility

    class Config:
        extra = "allow"  # Allow additional fields


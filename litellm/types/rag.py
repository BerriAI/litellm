"""
Type definitions for RAG (Retrieval Augmented Generation) Ingest API.
"""

from typing import Any, Dict, List, Literal, Optional, Union

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


class OpenAIVectorStoreOptions(TypedDict, total=False):
    """
    OpenAI vector store configuration.

    Example (auto-create):
        {"custom_llm_provider": "openai"}

    Example (use existing):
        {"custom_llm_provider": "openai", "vector_store_id": "vs_xxx"}
    """

    custom_llm_provider: Literal["openai"]
    vector_store_id: Optional[str]  # Existing VS ID (auto-creates if not provided)
    ttl_days: Optional[int]  # Time-to-live in days for indexed content


class BedrockVectorStoreOptions(TypedDict, total=False):
    """
    Bedrock Knowledge Base configuration.

    Example (auto-create KB and all resources):
        {"custom_llm_provider": "bedrock"}

    Example (use existing KB):
        {"custom_llm_provider": "bedrock", "vector_store_id": "KB_ID"}

    Auto-creation creates: S3 bucket, OpenSearch Serverless collection,
    IAM role, Knowledge Base, and Data Source.
    """

    custom_llm_provider: Literal["bedrock"]
    vector_store_id: Optional[str]  # Existing KB ID (auto-creates if not provided)

    # Bedrock-specific options
    s3_bucket: Optional[str]  # S3 bucket (auto-created if not provided)
    s3_prefix: Optional[str]  # S3 key prefix (default: "data/")
    embedding_model: Optional[str]  # Embedding model (default: amazon.titan-embed-text-v2:0)
    data_source_id: Optional[str]  # For existing KB: override auto-detected DS
    wait_for_ingestion: Optional[bool]  # Wait for completion (default: False - returns immediately)
    ingestion_timeout: Optional[int]  # Timeout in seconds if wait_for_ingestion=True (default: 300)

    # AWS auth (uses BaseAWSLLM)
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_session_token: Optional[str]
    aws_region_name: Optional[str]  # default: us-west-2
    aws_role_name: Optional[str]
    aws_session_name: Optional[str]
    aws_profile_name: Optional[str]
    aws_web_identity_token: Optional[str]
    aws_sts_endpoint: Optional[str]
    aws_external_id: Optional[str]


# Union type for vector store options
RAGIngestVectorStoreOptions = Union[OpenAIVectorStoreOptions, BedrockVectorStoreOptions]


class RAGIngestOptions(TypedDict, total=False):
    """
    Combined options for RAG ingest pipeline.

    Unified interface - just specify custom_llm_provider:

    Example (OpenAI):
        from litellm.types.rag import RAGIngestOptions, OpenAIVectorStoreOptions

        options: RAGIngestOptions = {
            "vector_store": OpenAIVectorStoreOptions(
                custom_llm_provider="openai",
                vector_store_id="vs_xxx",  # optional
            )
        }

    Example (Bedrock):
        from litellm.types.rag import RAGIngestOptions, BedrockVectorStoreOptions

        options: RAGIngestOptions = {
            "vector_store": BedrockVectorStoreOptions(
                custom_llm_provider="bedrock",
                vector_store_id="KB_ID",  # optional - auto-creates if not provided
                wait_for_ingestion=True,
            )
        }
    """

    name: Optional[str]  # Optional pipeline name for logging
    ocr: Optional[RAGIngestOCROptions]  # Optional OCR step
    chunking_strategy: Optional[RAGChunkingStrategy]  # RecursiveCharacterTextSplitter args
    embedding: Optional[RAGIngestEmbeddingOptions]  # Embedding model config
    vector_store: RAGIngestVectorStoreOptions  # OpenAI or Bedrock config

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


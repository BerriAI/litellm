"""
Type definitions for RAG (Retrieval Augmented Generation) Ingest API.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from litellm.types.utils import ModelResponse


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

    Example (with credentials):
        {"custom_llm_provider": "openai", "litellm_credential_name": "my-openai-creds"}
    """

    custom_llm_provider: Literal["openai"]
    vector_store_id: Optional[str]  # Existing VS ID (auto-creates if not provided)
    ttl_days: Optional[int]  # Time-to-live in days for indexed content

    # Credentials (loaded from litellm.credential_list if litellm_credential_name is provided)
    litellm_credential_name: Optional[str]  # Credential name to load from litellm.credential_list
    api_key: Optional[str]  # Direct API key (alternative to litellm_credential_name)
    api_base: Optional[str]  # Direct API base (alternative to litellm_credential_name)


class BedrockVectorStoreOptions(TypedDict, total=False):
    """
    Bedrock Knowledge Base configuration.

    Example (auto-create KB and all resources):
        {"custom_llm_provider": "bedrock"}

    Example (use existing KB):
        {"custom_llm_provider": "bedrock", "vector_store_id": "KB_ID"}

    Example (with credentials):
        {"custom_llm_provider": "bedrock", "litellm_credential_name": "my-aws-creds"}

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

    # Credentials (loaded from litellm.credential_list if litellm_credential_name is provided)
    litellm_credential_name: Optional[str]  # Credential name to load from litellm.credential_list

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


class VertexAIVectorStoreOptions(TypedDict, total=False):
    """
    Vertex AI RAG Engine configuration.

    Example (use existing corpus):
        {"custom_llm_provider": "vertex_ai", "vector_store_id": "CORPUS_ID", "gcs_bucket": "my-bucket"}

    Requires:
    - gcloud auth application-default login (for ADC authentication)
    - Files are uploaded to GCS via litellm.files.create_file, then imported into RAG corpus
    - GCS bucket must be provided via gcs_bucket or GCS_BUCKET_NAME env var
    """

    custom_llm_provider: Literal["vertex_ai"]
    vector_store_id: str  # RAG corpus ID (required for Vertex AI)

    # GCP config
    vertex_project: Optional[str]  # GCP project ID (uses env VERTEXAI_PROJECT if not set)
    vertex_location: Optional[str]  # GCP region (default: us-central1)
    vertex_credentials: Optional[str]  # Path to credentials JSON (uses ADC if not set)
    gcs_bucket: Optional[str]  # GCS bucket for file uploads (uses env GCS_BUCKET_NAME if not set)

    # Import settings
    wait_for_import: Optional[bool]  # Wait for import to complete (default: True)
    import_timeout: Optional[int]  # Timeout in seconds (default: 600)


class S3VectorsVectorStoreOptions(TypedDict, total=False):
    """
    AWS S3 Vectors configuration.

    Example (auto-create):
        {"custom_llm_provider": "s3_vectors", "vector_bucket_name": "my-embeddings"}

    Example (use existing):
        {"custom_llm_provider": "s3_vectors", "vector_bucket_name": "my-embeddings",
         "index_name": "my-index"}

    Example (with credentials):
        {"custom_llm_provider": "s3_vectors", "vector_bucket_name": "my-embeddings",
         "litellm_credential_name": "my-aws-creds"}

    Auto-creation creates: S3 vector bucket and vector index (if not provided).
    Embeddings are generated using LiteLLM's embedding API (supports any provider).
    """

    custom_llm_provider: Literal["s3_vectors"]
    vector_bucket_name: str  # Required - S3 vector bucket name
    index_name: Optional[str]  # Vector index name (auto-creates if not provided)

    # Index configuration (for auto-creation)
    dimension: Optional[int]  # Vector dimension (auto-detected from embedding model, or default: 1024)
    distance_metric: Optional[Literal["cosine", "euclidean"]]  # Default: cosine
    non_filterable_metadata_keys: Optional[List[str]]  # Keys excluded from filtering (e.g., ["source_text"])

    # Credentials (loaded from litellm.credential_list if litellm_credential_name is provided)
    litellm_credential_name: Optional[str]  # Credential name to load from litellm.credential_list

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
RAGIngestVectorStoreOptions = Union[
    OpenAIVectorStoreOptions, BedrockVectorStoreOptions, VertexAIVectorStoreOptions, S3VectorsVectorStoreOptions
]


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
    error: Optional[str]  # Error message if status is "failed"



class RAGIngestRequest(BaseModel):
    """Request body for RAG ingest API (for validation)."""

    file_url: Optional[str] = None  # URL to fetch file from
    file_id: Optional[str] = None  # Existing file ID
    ingest_options: Dict[str, Any]  # RAGIngestOptions as dict for flexibility

    model_config = ConfigDict(extra="allow")  # Allow additional fields


class RAGRetrievalConfig(TypedDict, total=False):
    """Configuration for vector store retrieval."""

    vector_store_id: str
    custom_llm_provider: str
    top_k: int  # max results from vector store
    filters: Optional[Dict[str, Any]]  # optional - vector store filters


class RAGRerankConfig(TypedDict, total=False):
    """Configuration for reranking results."""

    enabled: bool
    model: str
    top_n: int  # final number of chunks after reranking
    return_documents: Optional[bool]


class RAGQueryRequest(BaseModel):
    """Request body for RAG query API."""

    model: str
    messages: List[Any]
    retrieval_config: RAGRetrievalConfig
    rerank: Optional[RAGRerankConfig] = None
    stream: Optional[bool] = False

    model_config = ConfigDict(extra="allow")


class RAGQueryResponse(ModelResponse):
    """Response from RAG query API."""

    pass


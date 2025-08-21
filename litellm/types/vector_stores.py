from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from annotated_types import Ge
from pydantic import BaseModel
from typing_extensions import TypedDict

from litellm.types.router import CredentialLiteLLMParams, GenericLiteLLMParams


class SupportedVectorStoreIntegrations(str, Enum):
    """Supported vector store integrations."""

    BEDROCK = "bedrock"


class LiteLLM_VectorStoreConfig(TypedDict, total=False):
    """Parameters for initializing a vector store on Litellm proxy config.yaml"""

    vector_store_name: str
    litellm_params: Optional[Dict[str, Any]]


class LiteLLM_ManagedVectorStore(TypedDict, total=False):
    """LiteLLM managed vector store object - this is is the object stored in the database"""

    vector_store_id: str
    custom_llm_provider: str

    vector_store_name: Optional[str]
    vector_store_description: Optional[str]
    vector_store_metadata: Optional[Union[Dict[str, Any], str]]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    # credential fields
    litellm_credential_name: Optional[str]

    # litellm_params
    litellm_params: Optional[Dict[str, Any]]


class LiteLLM_ManagedVectorStoreListResponse(TypedDict, total=False):
    """Response format for listing vector stores"""

    object: Literal["list"]  # Always "list"
    data: List[LiteLLM_ManagedVectorStore]
    total_count: Optional[int]
    current_page: Optional[int]
    total_pages: Optional[int]


class VectorStoreUpdateRequest(BaseModel):
    vector_store_id: str
    custom_llm_provider: Optional[str] = None
    vector_store_name: Optional[str] = None
    vector_store_description: Optional[str] = None
    vector_store_metadata: Optional[Dict] = None


class VectorStoreDeleteRequest(BaseModel):
    vector_store_id: str


class VectorStoreInfoRequest(BaseModel):
    vector_store_id: str


class VectorStoreResultContent(TypedDict, total=False):
    """Content of a vector store result"""

    text: Optional[str]
    type: Optional[str]


class VectorStoreSearchResult(TypedDict, total=False):
    """Result of a vector store search"""
    score: Optional[float]
    content: Optional[List[VectorStoreResultContent]]
    file_id: Optional[str]
    filename: Optional[str]
    attributes: Optional[Dict]


class VectorStoreSearchResponse(TypedDict, total=False):
    """Response after searching a vector store"""

    object: Literal[
        "vector_store.search_results.page"
    ]  # Always "vector_store.search_results.page"
    search_query: Optional[str]
    data: Optional[List[VectorStoreSearchResult]]

class VectorStoreSearchOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store search API."""
    filters: Optional[Dict]
    max_num_results: Optional[int]  
    ranking_options: Optional[Dict]
    rewrite_query: Optional[bool]

class VectorStoreSearchRequest(VectorStoreSearchOptionalRequestParams, total=False):
    """Request body for searching a vector store"""
    query: Union[str, List[str]]


# Vector Store Creation Types
class VectorStoreExpirationPolicy(TypedDict, total=False):
    """The expiration policy for a vector store"""
    anchor: Literal["last_active_at"]  # Anchor timestamp after which the expiration policy applies
    days: int  # Number of days after anchor time that the vector store will expire


class VectorStoreAutoChunkingStrategy(TypedDict, total=False):
    """Auto chunking strategy configuration"""
    type: Literal["auto"]  # Always "auto"


class VectorStoreStaticChunkingStrategyConfig(TypedDict, total=False):
    """Static chunking strategy configuration"""
    max_chunk_size_tokens: int  # Maximum number of tokens per chunk
    chunk_overlap_tokens: int  # Number of tokens to overlap between chunks


class VectorStoreStaticChunkingStrategy(TypedDict, total=False):
    """Static chunking strategy"""
    type: Literal["static"]  # Always "static"
    static: VectorStoreStaticChunkingStrategyConfig


class VectorStoreChunkingStrategy(TypedDict, total=False):
    """Union type for chunking strategies"""
    # This can be either auto or static
    type: Literal["auto", "static"]
    static: Optional[VectorStoreStaticChunkingStrategyConfig]


class VectorStoreFileCounts(TypedDict, total=False):
    """File counts for a vector store"""
    in_progress: int
    completed: int
    failed: int
    cancelled: int
    total: int


class VectorStoreCreateOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store create API."""
    name: Optional[str]  # Name of the vector store
    file_ids: Optional[List[str]]  # List of File IDs that the vector store should use
    expires_after: Optional[VectorStoreExpirationPolicy]  # Expiration policy for the vector store
    chunking_strategy: Optional[VectorStoreChunkingStrategy]  # Chunking strategy for the files
    metadata: Optional[Dict[str, str]]  # Set of key-value pairs for metadata


class VectorStoreCreateRequest(VectorStoreCreateOptionalRequestParams, total=False):
    """Request body for creating a vector store"""
    pass  # All fields are optional for vector store creation


class VectorStoreCreateResponse(TypedDict, total=False):
    """Response after creating a vector store"""
    id: str  # ID of the vector store
    object: Literal["vector_store"]  # Always "vector_store"
    created_at: int  # Unix timestamp of when the vector store was created
    name: Optional[str]  # Name of the vector store
    bytes: int  # Size of the vector store in bytes
    file_counts: VectorStoreFileCounts  # File counts for the vector store
    status: Literal["expired", "in_progress", "completed"]  # Status of the vector store
    expires_after: Optional[VectorStoreExpirationPolicy]  # Expiration policy
    expires_at: Optional[int]  # Unix timestamp of when the vector store expires
    last_active_at: Optional[int]  # Unix timestamp of when the vector store was last active
    metadata: Optional[Dict[str, str]]  # Metadata associated with the vector store
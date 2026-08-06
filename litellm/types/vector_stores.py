from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class SupportedVectorStoreIntegrations(str, Enum):
    """Supported vector store integrations."""

    BEDROCK = "bedrock"
    RAGFLOW = "ragflow"


class LiteLLM_VectorStoreConfig(TypedDict, total=False):
    """Parameters for initializing a vector store on Litellm proxy config.yaml"""

    vector_store_name: str
    litellm_params: dict[str, Any] | None


class LiteLLM_ManagedVectorStore(TypedDict, total=False):
    """LiteLLM managed vector store object - this is is the object stored in the database"""

    vector_store_id: str
    custom_llm_provider: str

    vector_store_name: str | None
    vector_store_description: str | None
    vector_store_metadata: dict[str, Any] | str | None
    created_at: datetime | None
    updated_at: datetime | None

    # credential fields
    litellm_credential_name: str | None

    # litellm_params
    litellm_params: dict[str, Any] | None

    # access control fields
    team_id: str | None
    user_id: str | None


class LiteLLM_ManagedVectorStoreListResponse(TypedDict, total=False):
    """Response format for listing vector stores"""

    object: Literal["list"]  # Always "list"
    data: list[LiteLLM_ManagedVectorStore]
    total_count: int | None
    current_page: int | None
    total_pages: int | None


class VectorStoreUpdateRequest(BaseModel):
    vector_store_id: str
    custom_llm_provider: str | None = None
    vector_store_name: str | None = None
    vector_store_description: str | None = None
    vector_store_metadata: dict | None = None


class VectorStoreDeleteRequest(BaseModel):
    vector_store_id: str


class VectorStoreInfoRequest(BaseModel):
    vector_store_id: str


class VectorStoreResultContent(TypedDict, total=False):
    """Content of a vector store result"""

    text: str | None
    type: str | None


class VectorStoreSearchResult(TypedDict, total=False):
    """Result of a vector store search"""

    score: float | None
    content: list[VectorStoreResultContent] | None
    file_id: str | None
    filename: str | None
    attributes: dict | None


class VectorStoreSearchResponse(TypedDict, total=False):
    """Response after searching a vector store"""

    object: Literal["vector_store.search_results.page"]  # Always "vector_store.search_results.page"
    search_query: str | None
    data: list[VectorStoreSearchResult] | None


class VectorStoreSearchOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store search API."""

    filters: dict | None
    max_num_results: int | None
    ranking_options: dict | None
    rewrite_query: bool | None


class VectorStoreSearchRequest(VectorStoreSearchOptionalRequestParams, total=False):
    """Request body for searching a vector store"""

    query: str | list[str]


class VertexSearchDataStoreExtraBody(TypedDict, total=False):
    """
    Native Discovery Engine ``SearchRequest`` fields callers may forward via
    ``extra_body`` when searching a Vertex AI Search **data store** serving
    config (``.../dataStores/{id}/servingConfigs/default_config``).

    The data store is scoped by the request URL path, so target-selecting
    fields (``servingConfig``, ``branch``, ``entity``) are intentionally
    omitted and rejected by the transformation layer. Engine/app-only fields
    such as ``dataStoreSpecs`` and ``numResultsPerDataStore`` live on
    ``VertexSearchEngineExtraBody`` instead.
    """

    query: str
    pageSize: int
    pageToken: str
    offset: int
    oneBoxPageSize: int
    pageCategories: list[str]
    imageQuery: dict[str, Any]
    filter: str
    canonicalFilter: str
    orderBy: str
    userInfo: dict[str, Any]
    languageCode: str
    facetSpecs: list[dict[str, Any]]
    boostSpec: dict[str, Any]
    params: dict[str, Any]
    queryExpansionSpec: dict[str, Any]
    spellCorrectionSpec: dict[str, Any]
    userPseudoId: str
    contentSearchSpec: dict[str, Any]
    rankingExpression: str
    rankingExpressionBackend: str
    safeSearch: bool
    userLabels: dict[str, str]
    naturalLanguageQueryUnderstandingSpec: dict[str, Any]
    searchAsYouTypeSpec: dict[str, Any]
    displaySpec: dict[str, Any]
    crowdingSpecs: list[dict[str, Any]]
    relevanceThreshold: str
    relevanceScoreSpec: dict[str, Any]
    customRankingParams: dict[str, Any]


class VertexSearchEngineExtraBody(VertexSearchDataStoreExtraBody, total=False):
    """
    Native Discovery Engine ``SearchRequest`` fields callers may forward via
    ``extra_body`` when searching a Vertex AI Search **engine/app** serving
    config (``.../engines/{id}/servingConfigs/default_serving_config``).

    Inherits every data-store field and adds fields that only make sense when
    an app fans out across multiple member data stores, e.g. ``dataStoreSpecs``
    (per-store scoping/filtering) and ``numResultsPerDataStore``.
    """

    dataStoreSpecs: list[dict[str, Any]]
    numResultsPerDataStore: int


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
    static: VectorStoreStaticChunkingStrategyConfig | None


class VectorStoreFileCounts(TypedDict, total=False):
    """File counts for a vector store"""

    in_progress: int
    completed: int
    failed: int
    cancelled: int
    total: int


class VectorStoreCreateOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store create API."""

    name: str | None  # Name of the vector store
    file_ids: list[str] | None  # List of File IDs that the vector store should use
    expires_after: VectorStoreExpirationPolicy | None  # Expiration policy for the vector store
    chunking_strategy: VectorStoreChunkingStrategy | None  # Chunking strategy for the files
    metadata: dict[str, str] | None  # Set of key-value pairs for metadata


class VectorStoreCreateRequest(VectorStoreCreateOptionalRequestParams, total=False):
    """Request body for creating a vector store"""

    # All fields are optional for vector store creation


class VectorStoreCreateResponse(TypedDict, total=False):
    """Response after creating a vector store"""

    id: str  # ID of the vector store
    object: Literal["vector_store"]  # Always "vector_store"
    created_at: int  # Unix timestamp of when the vector store was created
    name: str | None  # Name of the vector store
    bytes: int  # Size of the vector store in bytes
    file_counts: VectorStoreFileCounts  # File counts for the vector store
    status: Literal["expired", "in_progress", "completed"]  # Status of the vector store
    expires_after: VectorStoreExpirationPolicy | None  # Expiration policy
    expires_at: int | None  # Unix timestamp of when the vector store expires
    last_active_at: int | None  # Unix timestamp of when the vector store was last active
    metadata: dict[str, str] | None  # Metadata associated with the vector store


class IndexCreateLiteLLMParams(BaseModel):
    vector_store_index: str
    vector_store_name: str


class IndexCreateRequest(BaseModel):
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: dict[str, Any] | None = None


class BaseVectorStoreAuthCredentials(TypedDict, total=False):
    headers: dict
    query_params: dict


class LiteLLM_ManagedVectorStoreIndex(BaseModel):
    """LiteLLM managed vector store index object - this is is the object stored in the database"""

    id: str
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: dict[str, Any] | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class VectorStoreIndexType(str, Enum):
    """Type of vector store index"""

    READ = "read"
    WRITE = "write"


class VectorStoreIndexEndpoints(TypedDict):
    """Endpoints for vector store index"""

    read: list[
        tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
    ]  # endpoints for reading a vector store index
    write: list[
        tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
    ]  # endpoints for writing a vector store index


VECTOR_STORE_OPENAI_PARAMS = Literal[
    "filters",
    "max_num_results",
    "ranking_options",
    "rewrite_query",
]


@dataclass
class VectorStoreToolParams:
    """Parameters extracted from a file_search tool definition"""

    filters: dict | None = None
    max_num_results: int | None = None
    ranking_options: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values"""
        return {
            k: v
            for k, v in {
                "filters": self.filters,
                "max_num_results": self.max_num_results,
                "ranking_options": self.ranking_options,
            }.items()
            if v is not None
        }

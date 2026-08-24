from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class SupportedVectorStoreIntegrations(str, Enum):
    """Supported vector store integrations."""

    BEDROCK = "bedrock"
    RAGFLOW = "ragflow"


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

    # access control fields
    team_id: Optional[str]
    user_id: Optional[str]


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

    object: Literal["vector_store.search_results.page"]  # Always "vector_store.search_results.page"
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
    pageCategories: List[str]
    imageQuery: Dict[str, Any]
    filter: str
    canonicalFilter: str
    orderBy: str
    userInfo: Dict[str, Any]
    languageCode: str
    facetSpecs: List[Dict[str, Any]]
    boostSpec: Dict[str, Any]
    params: Dict[str, Any]
    queryExpansionSpec: Dict[str, Any]
    spellCorrectionSpec: Dict[str, Any]
    userPseudoId: str
    contentSearchSpec: Dict[str, Any]
    rankingExpression: str
    rankingExpressionBackend: str
    safeSearch: bool
    userLabels: Dict[str, str]
    naturalLanguageQueryUnderstandingSpec: Dict[str, Any]
    searchAsYouTypeSpec: Dict[str, Any]
    displaySpec: Dict[str, Any]
    crowdingSpecs: List[Dict[str, Any]]
    relevanceThreshold: str
    relevanceScoreSpec: Dict[str, Any]
    customRankingParams: Dict[str, Any]


class VertexSearchEngineExtraBody(VertexSearchDataStoreExtraBody, total=False):
    """
    Native Discovery Engine ``SearchRequest`` fields callers may forward via
    ``extra_body`` when searching a Vertex AI Search **engine/app** serving
    config (``.../engines/{id}/servingConfigs/default_serving_config``).

    Inherits every data-store field and adds fields that only make sense when
    an app fans out across multiple member data stores, e.g. ``dataStoreSpecs``
    (per-store scoping/filtering) and ``numResultsPerDataStore``.
    """

    dataStoreSpecs: List[Dict[str, Any]]
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


class IndexCreateLiteLLMParams(BaseModel):
    vector_store_index: str
    vector_store_name: str


class IndexCreateRequest(BaseModel):
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: Optional[Dict[str, Any]] = None


class BaseVectorStoreAuthCredentials(TypedDict, total=False):
    headers: dict
    query_params: dict


class LiteLLM_ManagedVectorStoreIndex(BaseModel):
    """LiteLLM managed vector store index object - this is is the object stored in the database"""

    id: str
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class VectorStoreIndexType(str, Enum):
    """Type of vector store index"""

    READ = "read"
    WRITE = "write"


class VectorStoreIndexEndpoints(TypedDict):
    """Endpoints for vector store index"""

    read: List[
        Tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
    ]  # endpoints for reading a vector store index
    write: List[
        Tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
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

    filters: Optional[Dict] = None
    max_num_results: Optional[int] = None
    ranking_options: Optional[Dict] = None

    def to_dict(self) -> Dict:
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

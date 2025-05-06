from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict

from litellm.types.router import CredentialLiteLLMParams


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


class VectorStoreSearchResponse(TypedDict, total=False):
    """Response after searching a vector store"""

    object: Literal[
        "vector_store.search_results.page"
    ]  # Always "vector_store.search_results.page"
    search_query: Optional[str]
    data: Optional[List[VectorStoreSearchResult]]

from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

from litellm.types.router import GenericLiteLLMParams


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
    vector_store_metadata: Optional[Dict[str, Any]]
    created_at: Optional[int]
    updated_at: Optional[int]

    # litellm fields
    litellm_params: Optional[Dict[str, Any]]


class LiteLLM_ManagedVectorStoreListResponse(TypedDict, total=False):
    """Response format for listing vector stores"""

    object: Literal["list"]  # Always "list"
    data: List[LiteLLM_ManagedVectorStore]
    total_count: Optional[int]
    current_page: Optional[int]
    total_pages: Optional[int]

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from litellm.types.router import GenericLiteLLMParams


class SupportedVectorStoreIntegrations(str, Enum):
    """Supported vector store integrations."""

    BEDROCK = "bedrock"


class VectorStoreLiteLLMParams(GenericLiteLLMParams):
    """Parameters for initializing a vector store on Litellm"""

    id: str
    custom_llm_provider: Optional[str] = None


class VectorStoreConfig(TypedDict, total=False):
    """Configuration for a vector store"""

    vector_store_name: str
    litellm_params: VectorStoreLiteLLMParams

"""
Base Search transformation configuration.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SearchResult(LiteLLMPydanticObjectBase):
    """Single search result."""
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    last_updated: Optional[str] = None
    
    model_config = {"extra": "allow"}


class SearchResponse(LiteLLMPydanticObjectBase):
    """
    Standard Search response format.
    Standardized to Perplexity Search format - other providers should transform to this format.
    """
    results: List[SearchResult]
    object: str = "search"
    
    model_config = {"extra": "allow"}


class BaseSearchConfig:
    """
    Base configuration for Search transformations.
    Handles provider-agnostic Search operations.
    """

    def __init__(self) -> None:
        pass

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        Override in provider-specific implementations.
        """
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("get_complete_url must be implemented by provider")

    def transform_search_request(
        self,
        model: str,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to provider-specific format.
        Override in provider-specific implementations.
        
        Args:
            model: Model name
            query: Search query (string or list of strings)
            optional_params: Optional parameters for the request
            
        Returns:
            Dict with request data
        """
        raise NotImplementedError("transform_search_request must be implemented by provider")

    def transform_search_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform provider-specific Search response to standard format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_search_response must be implemented by provider")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )


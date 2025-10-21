"""
Calls Perplexity's /search endpoint to search the web.
"""
from typing import Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse
from litellm.secret_managers.main import get_secret_str


class PerplexitySearchConfig(BaseSearchConfig):
    PERPLEXITY_API_BASE = "https://api.perplexity.ai"
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
        """
        api_key = api_key or get_secret_str("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY is not set")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
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
        """
        api_base = api_base or get_secret_str("PERPLEXITY_API_BASE") or self.PERPLEXITY_API_BASE
        
        # append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            api_base = f"{api_base}/search"

        return api_base
        

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


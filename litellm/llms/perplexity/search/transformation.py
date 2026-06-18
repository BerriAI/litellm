"""
Calls Perplexity's /search endpoint to search the web.
"""

from typing import Dict, List, Optional, TypedDict, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _PerplexitySearchRequestRequired(TypedDict):
    """Required fields for Perplexity Search API request."""

    query: Union[str, List[str]]  # Required - search query or queries


class PerplexitySearchRequest(_PerplexitySearchRequestRequired, total=False):
    """
    Perplexity Search API request format.
    Based on: https://docs.perplexity.ai/api-reference/search-post
    """

    max_results: int  # Optional - maximum number of results (1-20), default 10
    search_domain_filter: List[str]  # Optional - list of domains to filter (max 20)
    max_tokens_per_page: int  # Optional - max tokens per page, default 1024
    country: str  # Optional - country code filter (e.g., 'US', 'GB', 'DE')


class PerplexitySearchConfig(BaseSearchConfig):
    PERPLEXITY_API_BASE = "https://api.perplexity.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Perplexity"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        """
        api_key = api_key or get_secret_str("PERPLEXITYAI_API_KEY")
        if not api_key:
            raise ValueError(
                "PERPLEXITYAI_API_KEY is not set. Set `PERPLEXITYAI_API_KEY` environment variable."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        api_base = (
            api_base
            or get_secret_str("PERPLEXITY_API_BASE")
            or self.PERPLEXITY_API_BASE
        )

        # append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            api_base = f"{api_base}/search"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to Perplexity API format.

        Perplexity's native parameter names match LiteLLM's unified search spec,
        so every set optional parameter is passed through as-is rather than an
        arbitrary subset. None-valued params are omitted.

        https://docs.perplexity.ai/api-reference/search-post

        Args:
            query: Search query (string or list of strings)
            optional_params: Optional Perplexity Search API parameters, forwarded
                as-is (e.g. max_results, search_domain_filter, country,
                max_tokens_per_page, search_recency_filter, search_after_date_filter,
                search_before_date_filter, last_updated_after_filter,
                last_updated_before_filter, search_language_filter,
                search_context_size, max_tokens)

        Returns:
            Dict with the Perplexity Search request body
        """
        request_data: PerplexitySearchRequest = {"query": query}

        forwarded = {
            key: value
            for key, value in optional_params.items()
            if value is not None
        }

        return dict(request_data, **forwarded)

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Perplexity API response to standard SearchResponse format.

        Args:
            raw_response: Raw httpx response from Perplexity API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        # Transform results to SearchResult objects
        results = []
        for result in response_json.get("results", []):
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("snippet", ""),
                date=result.get("date"),
                last_updated=result.get("last_updated"),
            )
            results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

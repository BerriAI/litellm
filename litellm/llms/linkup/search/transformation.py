"""
Calls Linkup's /search endpoint to search the web.

Linkup API Reference: https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search
"""
from typing import Dict, List, Literal, Optional, TypedDict, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _LinkupSearchRequestRequired(TypedDict):
    """Required fields for Linkup Search API request."""

    q: str  # Required - The natural language question for which you want to retrieve context
    depth: Literal["deep", "standard"]  # Required - Defines the precision of the search
    outputType: Literal[
        "searchResults", "sourcedAnswer", "structured"
    ]  # Required - The type of output


class LinkupSearchRequest(_LinkupSearchRequestRequired, total=False):
    """
    Linkup Search API request format.
    Based on: https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search
    """

    structuredOutputSchema: str  # Required only when outputType is "structured"
    includeSources: bool  # Optional - Include sources in response (default false)
    includeImages: bool  # Optional - Include images in results (default false)
    fromDate: str  # Optional - Start date for results (YYYY-MM-DD)
    toDate: str  # Optional - End date for results (YYYY-MM-DD)
    includeDomains: List[str]  # Optional - Domains to search on (max 100)
    excludeDomains: List[str]  # Optional - Domains to exclude
    includeInlineCitations: bool  # Optional - Include inline citations (default false)
    maxResults: int  # Optional - Maximum number of results to return


class LinkupSearchConfig(BaseSearchConfig):
    LINKUP_API_BASE = "https://api.linkup.so/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Linkup"

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
        api_key = api_key or get_secret_str("LINKUP_API_KEY")
        if not api_key:
            raise ValueError(
                "LINKUP_API_KEY is not set. Set `LINKUP_API_KEY` environment variable."
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
            api_base or get_secret_str("LINKUP_API_BASE") or self.LINKUP_API_BASE
        )

        # Append "/search" to the api base if it's not already there
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
        Transform Search request to Linkup API format.

        Transforms Perplexity unified spec parameters:
        - query -> q
        - max_results -> maxResults
        - search_domain_filter -> includeDomains
        - country -> (not directly supported)
        - max_tokens_per_page -> (not applicable)

        All other Linkup-specific parameters are passed through as-is.

        Args:
            query: Search query (string or list of strings). Linkup only supports single string queries.
            optional_params: Optional parameters for the request

        Returns:
            Dict with typed request data following LinkupSearchRequest spec
        """
        if isinstance(query, list):
            # Linkup only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: LinkupSearchRequest = {
            "q": query,
            "depth": optional_params.get("depth", "standard"),
            "outputType": optional_params.get("outputType", "searchResults"),
        }

        # Transform Perplexity unified spec parameters to Linkup format
        if "max_results" in optional_params:
            request_data["maxResults"] = optional_params["max_results"]

        if "search_domain_filter" in optional_params:
            request_data["includeDomains"] = optional_params["search_domain_filter"]

        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)

        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Linkup API response to LiteLLM unified SearchResponse format.

        Linkup -> LiteLLM mappings:
        - results[].name -> SearchResult.title
        - results[].url -> SearchResult.url
        - results[].content -> SearchResult.snippet
        - No date field in results (set to None)
        - No last_updated field in Linkup response (set to None)

        Args:
            raw_response: Raw httpx response from Linkup API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        # Transform results to SearchResult objects
        results = []

        # Process results array
        raw_results = response_json.get("results", [])

        for result in raw_results:
            # Handle both text and image result types
            result_type = result.get("type", "text")

            if result_type == "text":
                search_result = SearchResult(
                    title=result.get("name", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    date=None,
                    last_updated=None,
                )
                results.append(search_result)
            elif result_type == "image":
                # For image results, use the URL as both title and snippet if name not provided
                search_result = SearchResult(
                    title=result.get("name", result.get("url", "")),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    date=None,
                    last_updated=None,
                )
                results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )


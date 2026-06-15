"""
Calls fastCRW's /v1/search endpoint to search the web.

fastCRW is a Firecrawl-compatible web data engine (single Rust binary; self-host
or cloud). The search response uses the Firecrawl-compatible envelope
{ "success": true, "data": [ { "title", "url", "description", "markdown"? } ] }.

fastCRW API Reference: https://fastcrw.com/docs/rest-api
"""

from typing import Optional, TypedDict, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _FastCRWSearchRequestRequired(TypedDict):
    """Required fields for fastCRW Search API request."""

    query: str  # Required - search query


class FastCRWSearchRequest(_FastCRWSearchRequestRequired, total=False):
    """
    fastCRW Search API request format.
    Based on: https://fastcrw.com/docs/rest-api
    """

    limit: int  # Optional - maximum number of results to return
    sources: list[
        str
    ]  # Optional - sources to search ('web', 'images'), default ['web']
    scrapeOptions: dict  # Optional - options for scraping search results


class FastCRWSearchConfig(BaseSearchConfig):
    FASTCRW_API_BASE = "https://fastcrw.com/api/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "fastCRW"

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Validate environment and return headers.
        """
        api_key = api_key or get_secret_str("CRW_API_KEY")
        if not api_key:
            raise ValueError(
                "CRW_API_KEY is not set. Set `CRW_API_KEY` environment variable."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[dict, list[dict]]] = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        api_base = api_base or get_secret_str("CRW_API_BASE") or self.FASTCRW_API_BASE

        # Append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            api_base = f"{api_base}/search"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, list[str]],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        """
        Transform Search request to fastCRW API format.

        Transforms Perplexity unified spec parameters:
        - query -> query (same)
        - max_results -> limit

        All other fastCRW-specific parameters are passed through as-is.

        Args:
            query: Search query (string or list of strings). fastCRW only supports single string queries.
            optional_params: Optional parameters for the request

        Returns:
            Dict with typed request data following FastCRWSearchRequest spec
        """
        if isinstance(query, list):
            # fastCRW only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: FastCRWSearchRequest = {
            "query": query,
        }

        # Transform Perplexity unified spec parameters to fastCRW format
        if "max_results" in optional_params:
            request_data["limit"] = optional_params["max_results"]

        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)

        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        # By default, request markdown content if not explicitly specified
        # fastCRW doesn't return content unless explicitly requested via scrapeOptions
        if "scrapeOptions" not in result_data:
            result_data["scrapeOptions"] = {
                "formats": ["markdown"],
                "onlyMainContent": True,
            }

        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform fastCRW API response to LiteLLM unified SearchResponse format.

        fastCRW (Firecrawl-compatible) returns:
            {"success": true, "data": [{"url": "...", "title": "...", "description": "...", "markdown"?: "..."}, ...]}

        Args:
            raw_response: Raw httpx response from fastCRW API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        results = []

        data = response_json.get("data", [])

        if isinstance(data, list):
            for result in data:
                snippet = result.get("markdown") or result.get("description", "")
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=snippet,
                    date=None,
                    last_updated=None,
                )
                results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

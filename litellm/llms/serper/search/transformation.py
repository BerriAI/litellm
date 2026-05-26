"""
Calls Serper's /search endpoint to search Google.

Serper API Reference: https://serper.dev
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


class _SerperSearchRequestRequired(TypedDict):
    """Required fields for Serper Search API request."""

    q: str  # Required - search query


class SerperSearchRequest(_SerperSearchRequestRequired, total=False):
    """
    Serper Search API request format.
    Based on: https://serper.dev
    """

    num: int  # Optional - number of results to return, default 10
    page: int  # Optional - page number (default 1)
    gl: str  # Optional - country/geolocation code (e.g., "us", "gb")
    hl: str  # Optional - language code (e.g., "en", "de")
    location: str  # Optional - specific location for search targeting
    autocorrect: bool  # Optional - enable autocorrect (default True)
    tbs: str  # Optional - time-based search filter (e.g., "qdr:h", "qdr:d", "qdr:w")


class SerperSearchConfig(BaseSearchConfig):
    SERPER_API_BASE = "https://google.serper.dev"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Serper"

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
        api_key = api_key or get_secret_str("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY is not set. Set `SERPER_API_KEY` environment variable."
            )
        headers["X-API-KEY"] = api_key
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
        api_base = api_base or get_secret_str("SERPER_API_BASE") or self.SERPER_API_BASE
        api_base = api_base.rstrip("/")

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
        Transform Search request to Serper API format.

        Args:
            query: Search query (string or list of strings). Serper only supports single string queries.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results -> maps to `num`
                - search_domain_filter: List of domains -> appended as site: clauses to `q`
                - country: Country code filter (e.g., 'US', 'GB') -> maps to `gl` (lowercased)

        Returns:
            Dict with typed request data following SerperSearchRequest spec
        """
        if isinstance(query, list):
            query = " ".join(query)

        request_data: SerperSearchRequest = {
            "q": query,
        }

        if "max_results" in optional_params:
            request_data["num"] = optional_params["max_results"]

        if "country" in optional_params:
            request_data["gl"] = optional_params["country"].lower()

        if "search_domain_filter" in optional_params:
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                domain_clauses = " OR ".join(f"site:{d}" for d in domains)
                request_data["q"] = f"({request_data['q']}) ({domain_clauses})"

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
        Transform Serper API response to LiteLLM unified SearchResponse format.

        Serper -> LiteLLM mappings:
        - organic[].title -> SearchResult.title
        - organic[].link  -> SearchResult.url
        - organic[].snippet -> SearchResult.snippet
        - organic[].date -> SearchResult.date (optional, not always present)

        Args:
            raw_response: Raw httpx response from Serper API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        results = []
        for result in response_json.get("organic", []):
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("link", ""),
                snippet=result.get("snippet", ""),
                date=result.get("date"),
                last_updated=None,
            )
            results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

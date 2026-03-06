"""
Calls SearchAPI.io's Google Search API endpoint.

SearchAPI.io API Reference: https://www.searchapi.io/docs/google
"""
from typing import Dict, List, Literal, Optional, TypedDict, Union
from urllib.parse import urlencode

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _SearchAPIRequestRequired(TypedDict):
    """Required fields for SearchAPI.io request."""
    engine: str  # Required - search engine (e.g., 'google')
    q: str  # Required - search query


class SearchAPIRequest(_SearchAPIRequestRequired, total=False):
    """
    SearchAPI.io request format for Google Search.
    Based on: https://www.searchapi.io/docs/google
    """
    kgmid: str  # Optional - Knowledge Graph identifier
    device: str  # Optional - device type ('desktop', 'mobile', 'tablet')
    location: str  # Optional - geographic location
    uule: str  # Optional - Google-encoded location
    google_domain: str  # Optional - Google domain (deprecated)
    gl: str  # Optional - country code (e.g., 'us', 'uk')
    hl: str  # Optional - interface language (e.g., 'en', 'es')
    lr: str  # Optional - language restriction (e.g., 'lang_en')
    cr: str  # Optional - country restriction
    nfpr: int  # Optional - exclude auto-corrected results (0 or 1)
    filter: int  # Optional - duplicate/host crowding filter (0 or 1)
    safe: str  # Optional - SafeSearch ('active', 'off')
    time_period: str  # Optional - time period ('last_hour', 'last_day', 'last_week', 'last_month', 'last_year')
    time_period_min: str  # Optional - start date (MM/DD/YYYY)
    time_period_max: str  # Optional - end date (MM/DD/YYYY)
    num: int  # Optional - number of results (phased out by Google, constant 10)
    page: int  # Optional - page number for pagination
    optimization_strategy: str  # Optional - 'performance' or 'ads'


class SearchAPIConfig(BaseSearchConfig):
    SEARCHAPI_API_BASE = "https://www.searchapi.io/api/v1/search"
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "SearchAPI.io (Google Search)"
    
    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        SearchAPI.io uses GET requests for search.
        """
        return "GET"
    
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
        api_key = api_key or get_secret_str("SEARCHAPI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "SEARCHAPI_API_KEY is not set. Set `SEARCHAPI_API_KEY` environment variable."
            )
        
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
        Get complete URL for Search endpoint with query parameters.

        SearchAPI.io uses GET requests and includes api_key in query params.
        """
        api_base = api_base or get_secret_str("SEARCHAPI_API_BASE") or self.SEARCHAPI_API_BASE

        # Build query parameters from the transformed request body
        if data and isinstance(data, dict) and "_searchapi_params" in data:
            params = data["_searchapi_params"]
            query_string = urlencode(params, doseq=True)
            return f"{api_base}?{query_string}"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to SearchAPI.io format.

        Transforms unified spec parameters:
        - query → q
        - max_results → num (limited to 10 by Google)
        - search_domain_filter → q (append site: filters)
        - country → gl

        Args:
            query: Search query (string or list of strings)
            optional_params: Optional parameters for the request
            api_key: API key for authentication

        Returns:
            Dict with typed request data following SearchAPI.io spec
        """
        if isinstance(query, list):
            query = " ".join(query)

        # Get API key from parameter or environment
        api_key = api_key or get_secret_str("SEARCHAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "SEARCHAPI_API_KEY is not set. Set `SEARCHAPI_API_KEY` environment variable."
            )

        request_data: SearchAPIRequest = {
            "engine": "google",
            "q": query,
        }

        # Add API key to request
        result_data = dict(request_data)
        result_data["api_key"] = api_key

        # Transform unified spec parameters to SearchAPI.io format
        if "max_results" in optional_params:
            # Google now returns constant 10 results, but we can still set num
            num_results = min(optional_params["max_results"], 10)
            result_data["num"] = num_results

        if "search_domain_filter" in optional_params:
            # Convert to multiple "site:domain" clauses
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                result_data["q"] = self._append_domain_filters(
                    str(result_data["q"]), domains
                )

        if "country" in optional_params:
            # Map to gl parameter
            result_data["gl"] = optional_params["country"].lower()

        # Pass through all other SearchAPI.io-specific parameters
        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        # Store params in special key for URL building (GET request)
        return {
            "_searchapi_params": result_data,
        }

    @staticmethod
    def _append_domain_filters(query: str, domains: List[str]) -> str:
        """
        Add site: filters to restrict search to specific domains.
        """
        domain_clauses = [f"site:{domain}" for domain in domains]
        domain_query = " OR ".join(domain_clauses)

        return f"({query}) AND ({domain_query})"

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Optional[LiteLLMLoggingObj],
        **kwargs,
    ) -> SearchResponse:
        """
        Transform SearchAPI.io response to LiteLLM unified SearchResponse format.
        
        SearchAPI.io → LiteLLM mappings:
        - organic_results[].title → SearchResult.title
        - organic_results[].link → SearchResult.url
        - organic_results[].snippet → SearchResult.snippet
        - organic_results[].date → SearchResult.date
        """
        response_json = raw_response.json()

        # Transform results to SearchResult objects
        results: List[SearchResult] = []

        # Process organic results
        for result in response_json.get("organic_results", []):
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "")
            date = result.get("date")  # SearchAPI.io provides date in some results
            
            search_result = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                date=date,
                last_updated=None,  # SearchAPI.io doesn't provide last_updated
            )

            results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

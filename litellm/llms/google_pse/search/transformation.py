"""
Calls Google Programmable Search Engine (PSE) API to search the web.

Google PSE API Reference: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
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


class _GooglePSESearchRequestRequired(TypedDict):
    """Required fields for Google PSE Search API request."""
    q: str  # Required - search query
    cx: str  # Required - Programmable Search Engine ID
    key: str  # Required - API key


class GooglePSESearchRequest(_GooglePSESearchRequestRequired, total=False):
    """
    Google Programmable Search Engine API request format.
    Based on: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    """
    num: int  # Optional - number of results (1-10), default 10
    start: int  # Optional - index of first result (default 1)
    cr: str  # Optional - country restrict (e.g., 'countryUS', 'countryGB')
    dateRestrict: str  # Optional - restricts results by date (e.g., 'd[number]', 'w[number]', 'm[number]', 'y[number]')
    exactTerms: str  # Optional - phrase that all documents must contain
    excludeTerms: str  # Optional - word or phrase to exclude
    fileType: str  # Optional - file type to restrict results to
    filter: str  # Optional - controls duplicate content filtering ('0'=off, '1'=on)
    gl: str  # Optional - geolocation of end user (2-letter country code)
    hq: str  # Optional - append query terms to query
    imgSize: str  # Optional - returns images of specified size
    imgType: str  # Optional - returns images of specified type
    linkSite: str  # Optional - specifies all search results should contain a link to a URL
    lr: str  # Optional - language restrict (e.g., 'lang_en', 'lang_es')
    orTerms: str  # Optional - provides additional search terms
    relatedSite: str  # Optional - specifies all search results should be pages related to URL
    rights: str  # Optional - filters based on licensing
    safe: str  # Optional - search safety level ('active', 'off')
    searchType: str  # Optional - specifies search type ('image')
    siteSearch: str  # Optional - restricts results to URLs from specified site
    siteSearchFilter: str  # Optional - controls whether to include or exclude siteSearch ('e'=exclude, 'i'=include)
    sort: str  # Optional - sort expression


class GooglePSESearchConfig(BaseSearchConfig):
    GOOGLE_PSE_API_BASE = "https://www.googleapis.com/customsearch/v1"
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "Google PSE"
    
    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        Google PSE uses GET requests with query parameters.
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
        
        Google PSE uses API key as a query parameter, not in headers.
        This method is called but headers are not used for authentication.
        """
        api_key = api_key or get_secret_str("GOOGLE_PSE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_PSE_API_KEY is not set. Set `GOOGLE_PSE_API_KEY` environment variable.")
        
        # Also check for search engine ID
        search_engine_id = kwargs.get("search_engine_id") or get_secret_str("GOOGLE_PSE_ENGINE_ID")
        if not search_engine_id:
            raise ValueError("GOOGLE_PSE_ENGINE_ID is not set. Set `GOOGLE_PSE_ENGINE_ID` environment variable or pass `search_engine_id` parameter.")
        
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
        
        Google PSE uses GET requests, so we build the full URL with query params here.
        The transformed request body (data) contains the parameters needed for the URL.
        """
        from urllib.parse import urlencode
        
        api_base = api_base or get_secret_str("GOOGLE_PSE_API_BASE") or self.GOOGLE_PSE_API_BASE
        
        # Build query parameters from the transformed request body
        if data and isinstance(data, dict) and "_google_pse_params" in data:
            params = data["_google_pse_params"]
            query_string = urlencode(params)
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
        Transform Search request to Google PSE API format.
        
        Transforms Perplexity unified spec parameters:
        - query → q (same)
        - max_results → num
        - search_domain_filter → siteSearch
        - country → gl
        - max_tokens_per_page → (not applicable, ignored)
        
        All other Google PSE-specific parameters are passed through as-is.
        
        Args:
            query: Search query (string or list of strings). Google PSE supports single string queries.
            optional_params: Optional parameters for the request
            api_key: Google API key
            search_engine_id: Google Programmable Search Engine ID (cx parameter)
            
        Returns:
            Dict with typed request data following GooglePSESearchRequest spec
        """
        if isinstance(query, list):
            # Google PSE only supports single string queries
            query = " ".join(query)

        # Get API credentials
        api_key = api_key or get_secret_str("GOOGLE_PSE_API_KEY")
        search_engine_id = search_engine_id or get_secret_str("GOOGLE_PSE_ENGINE_ID")
        
        if not api_key:
            raise ValueError("GOOGLE_PSE_API_KEY is required")
        if not search_engine_id:
            raise ValueError("GOOGLE_PSE_ENGINE_ID is required")

        request_data: GooglePSESearchRequest = {
            "q": query,
            "cx": search_engine_id,
            "key": api_key,
        }
        
        # Transform unified spec parameters to Google PSE format
        if "max_results" in optional_params:
            # Google PSE supports 1-10 results per request
            num_results = min(optional_params["max_results"], 10)
            request_data["num"] = num_results
        
        if "search_domain_filter" in optional_params:
            # Convert list to single domain (take first if multiple)
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                request_data["siteSearch"] = domains[0]
                request_data["siteSearchFilter"] = "i"  # include
            elif isinstance(domains, str):
                request_data["siteSearch"] = domains
                request_data["siteSearchFilter"] = "i"  # include
        
        if "country" in optional_params:
            # Google PSE uses 2-letter country codes for gl parameter
            request_data["gl"] = optional_params["country"].upper()
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
        
        # Pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value
        
        # Store params in special key for URL building (Google PSE uses GET not POST)
        # Return a wrapper dict that stores params for get_complete_url to use
        return {
            "_google_pse_params": result_data,
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Google PSE API response to LiteLLM unified SearchResponse format.
        
        Google PSE → LiteLLM mappings:
        - items[].title → SearchResult.title
        - items[].link → SearchResult.url
        - items[].snippet → SearchResult.snippet
        - No date/last_updated fields in Google PSE response (set to None)
        
        Args:
            raw_response: Raw httpx response from Google PSE API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Transform results to SearchResult objects
        results = []
        for item in response_json.get("items", []):
            search_result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                date=None,  # Google PSE doesn't provide date in standard response
                last_updated=None,  # Google PSE doesn't provide last_updated in response
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )



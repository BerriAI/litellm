"""
Calls SearXNG's /search endpoint to search the web.

SearXNG API Reference: https://docs.searxng.org/dev/search_api.html
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


class _SearXNGSearchRequestRequired(TypedDict):
    """Required fields for SearXNG Search API request."""
    q: str  # Required - search query


class SearXNGSearchRequest(_SearXNGSearchRequestRequired, total=False):
    """
    SearXNG Search API request format.
    Based on: https://docs.searxng.org/dev/search_api.html
    """
    categories: str  # Optional - comma-separated list of categories
    engines: str  # Optional - comma-separated list of engines
    language: str  # Optional - language code
    pageno: int  # Optional - page number (default 1)
    time_range: str  # Optional - time range filter (day, month, year)
    format: str  # Optional - output format (json, csv, rss) - should be 'json'


class SearXNGSearchConfig(BaseSearchConfig):
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "SearXNG"
    
    def get_http_method(self):
        """
        SearXNG supports both GET and POST, but we'll use GET for simplicity.
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
        SearXNG is open-source and doesn't require an API key by default.
        Some instances may require authentication via headers.
        """
        # SearXNG typically doesn't require API keys, but support optional auth
        api_key = api_key or get_secret_str("SEARXNG_API_KEY")
        if api_key:
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
        Get complete URL for Search endpoint with query parameters.
        
        SearXNG uses GET requests, so we build the full URL with query params here.
        The transformed request body (data) contains the parameters needed for the URL.
        """
        from urllib.parse import urlencode
        
        api_base = api_base or get_secret_str("SEARXNG_API_BASE")
        
        if not api_base:
            raise ValueError(
                "SEARXNG_API_BASE is not set. Please set the `SEARXNG_API_BASE` environment variable "
                "or pass `api_base` parameter. Example: os.environ['SEARXNG_API_BASE'] = 'https://your-searxng-instance.com'"
            )
        
        # Append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            if api_base.endswith("/"):
                api_base = f"{api_base}search"
            else:
                api_base = f"{api_base}/search"
        
        # Build query parameters from the transformed request body
        if data and isinstance(data, dict) and "_searxng_params" in data:
            params = data["_searxng_params"]
            query_string = urlencode(params)
            return f"{api_base}?{query_string}"

        return api_base
        

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to SearXNG API format.
        
        Transforms Perplexity unified spec parameters:
        - query → q
        - max_results → (handled via pageno, SearXNG returns ~20 results per page)
        - search_domain_filter → (not directly supported)
        - country → language (approximate mapping)
        - max_tokens_per_page → (not applicable, ignored)
        
        All other SearXNG-specific parameters are passed through as-is.
        
        Args:
            query: Search query (string or list of strings). SearXNG only supports single string queries.
            optional_params: Optional parameters for the request
            
        Returns:
            Dict with typed request data following SearXNGSearchRequest spec
        """
        if isinstance(query, list):
            # SearXNG only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: SearXNGSearchRequest = {
            "q": query,
            "format": "json",  # Always request JSON format
        }
        
        # Transform Perplexity unified spec parameters to SearXNG format
        if "country" in optional_params:
            # Map country code to language (approximate)
            country = optional_params["country"].lower()
            if country == "us" or country == "uk":
                request_data["language"] = "en"
            elif country == "de":
                request_data["language"] = "de"
            elif country == "fr":
                request_data["language"] = "fr"
            elif country == "es":
                request_data["language"] = "es"
            elif country == "jp":
                request_data["language"] = "ja"
            else:
                request_data["language"] = country  # Pass through as-is
        
        # Handle max_results via pagination (SearXNG returns ~20 results per page by default)
        # For simplicity, we'll just use page 1 and let SearXNG return its default number of results
        if "max_results" in optional_params:
            # Note: We could calculate pageno based on max_results, but for now we'll ignore this
            # and let SearXNG return its default results
            pass
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
        
        # Pass through all other SearXNG-specific parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value
        
        # Store params in special key for GET request URL building
        # This will be used by get_complete_url to build the query string
        return {"_searxng_params": result_data}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform SearXNG API response to LiteLLM unified SearchResponse format.
        
        SearXNG → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].content → SearchResult.snippet
        - results[].publishedDate OR results[].pubdate → SearchResult.date
        - No last_updated field in SearXNG response (set to None)
        
        Args:
            raw_response: Raw httpx response from SearXNG API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Transform results to SearchResult objects
        # Note: SearXNG doesn't natively support limiting results via API params
        # It returns ~20 results per page by default
        results = []
        for result in response_json.get("results", []):
            # Get date from either publishedDate or pubdate field
            date = result.get("publishedDate") or result.get("pubdate")
            
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", ""),  # SearXNG uses "content" for snippet
                date=date,
                last_updated=None,  # SearXNG doesn't provide last_updated in response
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


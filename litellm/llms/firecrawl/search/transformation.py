"""
Calls Firecrawl's /search endpoint to search the web.

Firecrawl API Reference: https://docs.firecrawl.dev/api-reference/endpoint/search
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


class _FirecrawlSearchRequestRequired(TypedDict):
    """Required fields for Firecrawl Search API request."""
    query: str  # Required - search query


class FirecrawlSearchRequest(_FirecrawlSearchRequestRequired, total=False):
    """
    Firecrawl Search API request format.
    Based on: https://docs.firecrawl.dev/api-reference/endpoint/search
    """
    limit: int  # Optional - maximum number of results to return (default 5, max 100)
    sources: List[str]  # Optional - sources to search ('web', 'images', 'news'), default ['web']
    categories: List[Dict[str, str]]  # Optional - categories to filter by (github, research, pdf)
    tbs: str  # Optional - time-based search parameter
    location: str  # Optional - location parameter for geo-targeting
    country: str  # Optional - ISO country code (default 'US')
    timeout: int  # Optional - timeout in milliseconds (default 60000)
    ignoreInvalidURLs: bool  # Optional - exclude invalid URLs (default false)
    scrapeOptions: Dict  # Optional - options for scraping search results


class FirecrawlSearchConfig(BaseSearchConfig):
    FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v2"
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "Firecrawl"
    
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
        api_key = api_key or get_secret_str("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set. Set `FIRECRAWL_API_KEY` environment variable.")
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
        api_base = api_base or get_secret_str("FIRECRAWL_API_BASE") or self.FIRECRAWL_API_BASE
        
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
        Transform Search request to Firecrawl API format.
        
        Transforms Perplexity unified spec parameters:
        - query → query (same)
        - max_results → limit
        - search_domain_filter → (not directly supported, can use scrapeOptions)
        - country → country
        - max_tokens_per_page → (not applicable, ignored)
        
        All other Firecrawl-specific parameters are passed through as-is.
        
        Args:
            query: Search query (string or list of strings). Firecrawl only supports single string queries.
            optional_params: Optional parameters for the request
            
        Returns:
            Dict with typed request data following FirecrawlSearchRequest spec
        """
        if isinstance(query, list):
            # Firecrawl only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: FirecrawlSearchRequest = {
            "query": query,
        }
        
        # Transform Perplexity unified spec parameters to Firecrawl format
        if "max_results" in optional_params:
            request_data["limit"] = optional_params["max_results"]
        
        if "country" in optional_params:
            request_data["country"] = optional_params["country"]
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
        
        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value
        
        # By default, request markdown content if not explicitly specified
        # Firecrawl doesn't return content unless explicitly requested via scrapeOptions
        if "scrapeOptions" not in result_data:
            result_data["scrapeOptions"] = {
                "formats": ["markdown"],
                "onlyMainContent": True
            }
        
        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Firecrawl API response to LiteLLM unified SearchResponse format.
        
        Firecrawl → LiteLLM mappings:
        - data.web[].title → SearchResult.title
        - data.web[].url → SearchResult.url
        - data.web[].description OR data.web[].markdown → SearchResult.snippet
        - No date field in web results (set to None)
        - No last_updated field in Firecrawl response (set to None)
        
        Note: Firecrawl v2 returns results organized by source type (web, images, news).
        We primarily use web results for the unified format.
        
        Args:
            raw_response: Raw httpx response from Firecrawl API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Transform results to SearchResult objects
        results = []
        
        # Process web results (primary source)
        data = response_json.get("data", {})
        web_results = data.get("web", [])
        
        for result in web_results:
            # Use markdown if available, otherwise fall back to description
            snippet = result.get("markdown") or result.get("description", "")
            
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=snippet,
                date=None,  # Web results don't include date
                last_updated=None,  # Firecrawl doesn't provide last_updated in response
            )
            results.append(search_result)
        
        # Process news results if available (they have date field)
        news_results = data.get("news", [])
        for result in news_results:
            snippet = result.get("markdown") or result.get("snippet", "")
            
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=snippet,
                date=result.get("date"),  # News results include date
                last_updated=None,
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


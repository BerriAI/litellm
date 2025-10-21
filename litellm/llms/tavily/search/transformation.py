"""
Calls Tavily's /search endpoint to search the web.

Tavily API Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""
from typing import Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class TavilySearchConfig(BaseSearchConfig):
    TAVILY_API_BASE = "https://api.tavily.com"
    
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
        api_key = api_key or get_secret_str("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set. Set `TAVILY_API_KEY` environment variable.")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        api_base = api_base or get_secret_str("TAVILY_API_BASE") or self.TAVILY_API_BASE
        
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
        Transform Search request from LiteLLM unified format to Tavily API format.
        
        LiteLLM → Tavily mappings:
        - query → query (same)
        - max_results → max_results (same)
        - search_domain_filter → include_domains (Tavily-specific field name)
        - country → country (same)
        - max_tokens_per_page → Not applicable to Tavily
        
        https://docs.tavily.com/documentation/api-reference/endpoint/search
        
        Args:
            query: Search query (string or list of strings)
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results (0-20)
                - search_domain_filter: List of domains to include (becomes include_domains)
                - country: Country code to boost results from
                - max_tokens_per_page: Not used by Tavily
            
        Returns:
            Dict with request data following Tavily API spec
        """
        # Tavily only accepts string queries, not lists
        # If query is a list, join with " OR " or take first item
        if isinstance(query, list):
            query_str = query[0] if len(query) > 0 else ""
        else:
            query_str = query
            
        request_data: Dict = {
            "query": query_str,
        }
        
        # Map max_results (same field name)
        max_results = optional_params.get("max_results")
        if max_results is not None:
            request_data["max_results"] = max_results
        
        # Map search_domain_filter → include_domains (different field name)
        search_domain_filter = optional_params.get("search_domain_filter")
        if search_domain_filter is not None:
            request_data["include_domains"] = search_domain_filter
        
        # Map country (same field name)
        country = optional_params.get("country")
        if country is not None:
            # Tavily expects lowercase country names
            request_data["country"] = country.lower() if isinstance(country, str) else country
        
        # max_tokens_per_page is not applicable to Tavily
        # It has chunks_per_source instead, but we don't map it
        
        return request_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Tavily API response to LiteLLM unified SearchResponse format.
        
        Tavily → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].content → SearchResult.snippet
        - No date/last_updated fields in Tavily response (set to None)
        
        Args:
            raw_response: Raw httpx response from Tavily API
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
                snippet=result.get("content", ""),  # Tavily uses "content" instead of "snippet"
                date=None,  # Tavily doesn't provide date in response
                last_updated=None,  # Tavily doesn't provide last_updated in response
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


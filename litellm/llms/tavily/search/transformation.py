"""
Calls Tavily's /search endpoint to search the web.

Tavily API Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
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


class _TavilySearchRequestRequired(TypedDict):
    """Required fields for Tavily Search API request."""
    query: str  # Required - search query


class TavilySearchRequest(_TavilySearchRequestRequired, total=False):
    """
    Tavily Search API request format.
    Based on: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    max_results: int  # Optional - maximum number of results (0-20), default 5
    include_domains: List[str]  # Optional - list of domains to include (max 300)
    exclude_domains: List[str]  # Optional - list of domains to exclude (max 150)
    topic: str  # Optional - category of search ('general', 'news', 'finance'), default 'general'
    search_depth: str  # Optional - depth of search ('basic', 'advanced'), default 'basic'
    include_answer: Union[bool, str]  # Optional - include LLM-generated answer
    include_raw_content: Union[bool, str]  # Optional - include raw HTML content
    include_images: bool  # Optional - perform image search
    include_image_descriptions: bool  # Optional - add descriptions for images
    include_favicon: bool  # Optional - include favicon URL
    time_range: str  # Optional - time range filter ('day', 'week', 'month', 'year', 'd', 'w', 'm', 'y')
    start_date: str  # Optional - start date filter (YYYY-MM-DD)
    end_date: str  # Optional - end date filter (YYYY-MM-DD)
    country: str  # Optional - country code filter (e.g., 'US', 'GB', 'DE')


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
        Transform Search request to Tavily API format.
        
        Args:
            query: Search query (string or list of strings). Tavily only supports single string queries.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results (0-20)
                - search_domain_filter: List of domains to include (max 300) -> maps to `include_domains`
                - exclude_domains: List of domains to exclude (max 150)
                - topic: Category of search ('general', 'news', 'finance')
                - search_depth: Depth of search ('basic', 'advanced')
                - include_answer: Include LLM-generated answer (bool or 'basic', 'advanced')
                - include_raw_content: Include raw HTML content (bool or 'markdown', 'text')
                - include_images: Perform image search (bool)
                - include_image_descriptions: Add descriptions for images (bool)
                - include_favicon: Include favicon URL (bool)
                - time_range: Time range filter ('day', 'week', 'month', 'year', 'd', 'w', 'm', 'y')
                - start_date: Start date filter (YYYY-MM-DD)
                - end_date: End date filter (YYYY-MM-DD)
                - country: Country code filter (e.g., 'US', 'GB', 'DE')
            
        Returns:
            Dict with typed request data following TavilySearchRequest spec
        """
        if isinstance(query, list):
            # Tavily only supports single string queries
            query = " ".join(query)

        request_data: TavilySearchRequest = {
            "query": query,
        }
        
        # Map max_results (same field name)
        if "max_results" in optional_params:
            request_data["max_results"] = optional_params["max_results"]
        
        # Map search_domain_filter → include_domains (different field name in Tavily)
        if "search_domain_filter" in optional_params:
            request_data["include_domains"] = optional_params["search_domain_filter"]
        
        # Map exclude_domains (same field name)
        if "exclude_domains" in optional_params:
            request_data["exclude_domains"] = optional_params["exclude_domains"]
        
        # Map topic (same field name)
        if "topic" in optional_params:
            request_data["topic"] = optional_params["topic"]
        
        # Map search_depth (same field name)
        if "search_depth" in optional_params:
            request_data["search_depth"] = optional_params["search_depth"]
        
        # Map include_answer (same field name)
        if "include_answer" in optional_params:
            request_data["include_answer"] = optional_params["include_answer"]
        
        # Map include_raw_content (same field name)
        if "include_raw_content" in optional_params:
            request_data["include_raw_content"] = optional_params["include_raw_content"]
        
        # Map include_images (same field name)
        if "include_images" in optional_params:
            request_data["include_images"] = optional_params["include_images"]
        
        # Map include_image_descriptions (same field name)
        if "include_image_descriptions" in optional_params:
            request_data["include_image_descriptions"] = optional_params["include_image_descriptions"]
        
        # Map include_favicon (same field name)
        if "include_favicon" in optional_params:
            request_data["include_favicon"] = optional_params["include_favicon"]
        
        # Map time_range (same field name)
        if "time_range" in optional_params:
            request_data["time_range"] = optional_params["time_range"]
        
        # Map start_date (same field name)
        if "start_date" in optional_params:
            request_data["start_date"] = optional_params["start_date"]
        
        # Map end_date (same field name)
        if "end_date" in optional_params:
            request_data["end_date"] = optional_params["end_date"]
        
        # Map country (same field name, but lowercase for Tavily)
        if "country" in optional_params:
            request_data["country"] = optional_params["country"].lower()
        
        return dict(request_data)

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


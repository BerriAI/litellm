"""
Calls Exa AI's /search endpoint to search the web.

Exa AI API Reference: https://docs.exa.ai/reference/search
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


class _ExaAISearchRequestRequired(TypedDict):
    """Required fields for Exa AI Search API request."""
    query: str  # Required - search query


class ExaAISearchRequest(_ExaAISearchRequestRequired, total=False):
    """
    Exa AI Search API request format.
    Based on: https://docs.exa.ai/reference/search
    """
    type: str  # Optional - search type ('keyword', 'neural', 'fast', 'auto'), default 'auto'
    category: str  # Optional - data category ('company', 'research paper', 'news', 'pdf', 'github', 'tweet', 'personal site', 'linkedin profile', 'financial report')
    userLocation: str  # Optional - two-letter ISO country code
    numResults: int  # Optional - number of results (max 100), default 10
    includeDomains: List[str]  # Optional - list of domains to include
    excludeDomains: List[str]  # Optional - list of domains to exclude
    startCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    endCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    startPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    endPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    includeText: List[str]  # Optional - strings that must be present in webpage text
    excludeText: List[str]  # Optional - strings that must not be present in webpage text
    context: Union[bool, dict]  # Optional - format results for LLMs
    moderation: bool  # Optional - enable content moderation, default false
    contents: dict  # Optional - content retrieval options


class ExaAISearchConfig(BaseSearchConfig):
    EXA_AI_API_BASE = "https://api.exa.ai"
    
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
        api_key = api_key or get_secret_str("EXA_API_KEY")
        if not api_key:
            raise ValueError("EXA_API_KEY is not set. Set `EXA_API_KEY` environment variable.")
        headers["x-api-key"] = api_key
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
        api_base = api_base or get_secret_str("EXA_API_BASE") or self.EXA_AI_API_BASE
        
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
        Transform Search request to Exa AI API format.
        
        Transforms Perplexity unified spec parameters:
        - query → query (same)
        - max_results → numResults
        - search_domain_filter → includeDomains
        - country → userLocation
        - max_tokens_per_page → (not applicable, ignored)
        
        All other Exa-specific parameters are passed through as-is.
        
        Args:
            query: Search query (string or list of strings). Exa AI only supports single string queries.
            optional_params: Optional parameters for the request
            
        Returns:
            Dict with typed request data following ExaAISearchRequest spec
        """
        if isinstance(query, list):
            # Exa AI only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: ExaAISearchRequest = {
            "query": query,
        }
        
        # Transform Perplexity unified spec parameters to Exa format
        if "max_results" in optional_params:
            request_data["numResults"] = optional_params["max_results"]
        
        if "search_domain_filter" in optional_params:
            request_data["includeDomains"] = optional_params["search_domain_filter"]
        
        if "country" in optional_params:
            request_data["userLocation"] = optional_params["country"]
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
        
        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value
        
        # By default, request text content if not explicitly specified
        # Exa AI doesn't return content/text unless explicitly requested
        if "contents" not in result_data:
            result_data["contents"] = {"text": True}
        
        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Exa AI API response to LiteLLM unified SearchResponse format.
        
        Exa AI → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].text → SearchResult.snippet
        - results[].publishedDate → SearchResult.date
        - No last_updated field in Exa AI response (set to None)
        
        Args:
            raw_response: Raw httpx response from Exa AI API
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
                snippet=result.get("text", ""),  # Exa AI uses "text" for content
                date=result.get("publishedDate"),  # ISO 8601 datetime string
                last_updated=None,  # Exa AI doesn't provide last_updated in response
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


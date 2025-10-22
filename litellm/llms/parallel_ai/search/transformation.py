"""
Calls Parallel AI's /search endpoint to search the web.

Parallel AI API Reference: https://docs.parallel.ai/api-reference/search-and-extract-api-beta/search
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


class _ParallelAISourcePolicy(TypedDict, total=False):
    """Source policy for Parallel AI search results."""
    allowed_domains: List[str]  # Optional - list of allowed domains
    disallowed_domains: List[str]  # Optional - list of disallowed domains


class _ParallelAISearchRequestRequired(TypedDict):
    """Required fields for Parallel AI Search API request."""
    # Note: At least one of objective or search_queries must be provided
    pass


class ParallelAISearchRequest(_ParallelAISearchRequestRequired, total=False):
    """
    Parallel AI Search API request format.
    Based on: https://docs.parallel.ai/api-reference/search-and-extract-api-beta/search
    """
    objective: str  # Optional - natural-language description of search goal
    search_queries: List[str]  # Optional - list of keyword search queries
    processor: str  # Optional - search processor ('base', 'pro'), default 'base'
    max_results: int  # Optional - maximum number of results, default 10
    max_chars_per_result: int  # Optional - max characters per result excerpt
    source_policy: _ParallelAISourcePolicy  # Optional - source policy for allowed/disallowed domains


class ParallelAISearchConfig(BaseSearchConfig):
    PARALLEL_AI_API_BASE = "https://api.parallel.ai"
    PARALLEL_HEADER_SEARCH_EXTRACT_VALUE = "search-extract-2025-10-10"
    
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
        api_key = api_key or get_secret_str("PARALLEL_AI_API_KEY") or get_secret_str("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError("PARALLEL_API_KEY is not set. Set `PARALLEL_API_KEY` environment variable.")
        headers["x-api-key"] = api_key
        headers["Content-Type"] = "application/json"
        headers["parallel-beta"] = self.PARALLEL_HEADER_SEARCH_EXTRACT_VALUE
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
        api_base = api_base or get_secret_str("PARALLEL_AI_API_BASE") or self.PARALLEL_AI_API_BASE
        
        # Parallel AI search endpoint is at /v1beta/search
        if not api_base.endswith("/v1beta/search"):
            if api_base.endswith("/"):
                api_base = f"{api_base}v1beta/search"
            else:
                api_base = f"{api_base}/v1beta/search"

        return api_base
    
    def _transform_query_to_objective(self, query: Union[str, List[str]]) -> str:
        """
        Transform query to objective.
        """
        if isinstance(query, list):
            return " ".join(query)
        return query
        

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to Parallel AI API format.
        
        Args:
            query: Search query (string or list of strings)
                - If string: maps to `objective` (natural language)
                - If list: maps to `search_queries` (keyword queries)
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results (default 10)
                - search_domain_filter: List of domains to include -> maps to `source_policy.allowed_domains`
                - exclude_domains: List of domains to exclude -> maps to `source_policy.disallowed_domains`
                - processor: Search processor ('base', 'pro')
                - max_chars_per_result: Max characters per result excerpt
            
        Returns:
            Dict with typed request data following ParallelAISearchRequest spec
        """
        request_data: ParallelAISearchRequest = {}
        
        # Map query to objective (string or list both become objective)
        if isinstance(query, list):
            request_data["objective"] = self._transform_query_to_objective(query)
        else:
            request_data["objective"] = query
        
        # Transform Perplexity unified spec parameters to Parallel AI format
        if "max_results" in optional_params:
            request_data["max_results"] = optional_params["max_results"]
        
        # Map domain filters to source_policy
        source_policy: _ParallelAISourcePolicy = {}
        
        if "search_domain_filter" in optional_params:
            source_policy["allowed_domains"] = optional_params["search_domain_filter"]
        
        if "exclude_domains" in optional_params:
            source_policy["disallowed_domains"] = optional_params["exclude_domains"]
        
        if source_policy:
            request_data["source_policy"] = source_policy
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
        
        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value
        
        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Parallel AI API response to LiteLLM unified SearchResponse format.
        
        Parallel AI → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].excerpts (array) → SearchResult.snippet (joined string)
        - No date/last_updated fields in Parallel AI response (set to None)
        
        Args:
            raw_response: Raw httpx response from Parallel AI API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Transform results to SearchResult objects
        results = []
        for result in response_json.get("results", []):
            # Join excerpts array into a single snippet string
            excerpts = result.get("excerpts", [])
            snippet = " ... ".join(excerpts) if excerpts else ""
            
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=snippet,
                date=None,  # Parallel AI doesn't provide date in response
                last_updated=None,  # Parallel AI doesn't provide last_updated in response
            )
            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


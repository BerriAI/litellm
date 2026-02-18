"""
Calls DuckDuckGo's Instant Answer API to search the web.

DuckDuckGo API Reference: https://duckduckgo.com/api
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


class _DuckDuckGoSearchRequestRequired(TypedDict):
    """Required fields for DuckDuckGo Search API request."""
    q: str  # Required - search query


class DuckDuckGoSearchRequest(_DuckDuckGoSearchRequestRequired, total=False):
    """
    DuckDuckGo Instant Answer API request format.
    Based on: https://duckduckgo.com/api
    """
    format: str  # Optional - output format ('json', 'xml'), default 'json'
    pretty: int  # Optional - pretty print (0 or 1), default 1
    no_redirect: int  # Optional - skip HTTP redirects (0 or 1), default 0
    no_html: int  # Optional - remove HTML from text (0 or 1), default 0
    skip_disambig: int  # Optional - skip disambiguation results (0 or 1), default 0


class DuckDuckGoSearchConfig(BaseSearchConfig):
    DUCKDUCKGO_API_BASE = "https://api.duckduckgo.com"
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "DuckDuckGo"
    
    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        Get HTTP method for search requests.
        DuckDuckGo Instant Answer API uses GET requests.
        
        Returns:
            HTTP method 'GET'
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
        DuckDuckGo Instant Answer API does not require authentication.
        """
        # DuckDuckGo API is free and doesn't require API key
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
        DuckDuckGo uses query parameters, so we construct the URL with the query.
        """
        api_base = api_base or get_secret_str("DUCKDUCKGO_API_BASE") or self.DUCKDUCKGO_API_BASE
        
        # Build query parameters from the transformed request body
        if data and isinstance(data, dict) and "_duckduckgo_params" in data:
            params = data["_duckduckgo_params"]
            query_string = urlencode(params, doseq=True)
            return f"{api_base}/?{query_string}"
        
        return api_base
        

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to DuckDuckGo API format.
        
        Args:
            query: Search query (string or list of strings). DuckDuckGo only supports single string queries.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results (DuckDuckGo API doesn't directly support this, used for filtering)
                - format: Output format ('json', 'xml')
                - pretty: Pretty print (0 or 1)
                - no_redirect: Skip HTTP redirects (0 or 1)
                - no_html: Remove HTML from text (0 or 1)
                - skip_disambig: Skip disambiguation results (0 or 1)
            
        Returns:
            Dict with typed request data following DuckDuckGoSearchRequest spec
        """
        if isinstance(query, list):
            # DuckDuckGo only supports single string queries
            query = " ".join(query)

        request_data: DuckDuckGoSearchRequest = {
            "q": query,
            "format": "json",  # Always use JSON format
        }
        
        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)
    
        if "max_results" in optional_params:
            result_data["_max_results"] = optional_params["max_results"]
        
        # Pass through DuckDuckGo-specific parameters
        ddg_params = ["pretty", "no_redirect", "no_html", "skip_disambig"]
        for param in ddg_params:
            if param in optional_params:
                result_data[param] = optional_params[param]
        
        return {
            "_duckduckgo_params": result_data,
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform DuckDuckGo API response to LiteLLM unified SearchResponse format.
        
        DuckDuckGo → LiteLLM mappings:
        - RelatedTopics[].Text → SearchResult.title + snippet
        - RelatedTopics[].FirstURL → SearchResult.url
        - RelatedTopics[].Text → SearchResult.snippet
        - No date/last_updated fields in DuckDuckGo response (set to None)
        
        Args:
            raw_response: Raw httpx response from DuckDuckGo API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Extract max_results from the request URL params
        query_params = raw_response.request.url.params if raw_response.request else {}
        max_results = None
        if "_max_results" in query_params:
            try:
                max_results = int(query_params["_max_results"])
            except (ValueError, TypeError):
                pass
        
        # Transform results to SearchResult objects
        results = []
        
        # DuckDuckGo can return results in different fields
        # Priority: Abstract > Answer > RelatedTopics
        
        # Check if there's an Abstract with URL
        if response_json.get("AbstractURL") and response_json.get("AbstractText"):
            abstract_result = SearchResult(
                title=response_json.get("Heading", ""),
                url=response_json.get("AbstractURL", ""),
                snippet=response_json.get("AbstractText", ""),
                date=None,
                last_updated=None,
            )
            results.append(abstract_result)
        
        # Process RelatedTopics
        related_topics = response_json.get("RelatedTopics", [])
        for topic in related_topics:
            # Stop if we've reached max_results
            if max_results is not None and len(results) >= max_results:
                break
            
            if isinstance(topic, dict):
                # Check if it's a direct result
                if "FirstURL" in topic and "Text" in topic:
                    text = topic.get("Text", "")
                    url = topic.get("FirstURL", "")
                    
                    # Try to split title and snippet
                    if " - " in text:
                        parts = text.split(" - ", 1)
                        title = parts[0]
                        snippet = parts[1] if len(parts) > 1 else text
                    else:
                        title = text[:50] + "..." if len(text) > 50 else text
                        snippet = text
                    
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        date=None,
                        last_updated=None,
                    )
                    results.append(search_result)
                
                # Check if it contains nested topics
                elif "Topics" in topic:
                    nested_topics = topic.get("Topics", [])
                    for nested_topic in nested_topics:
                        # Stop if we've reached max_results
                        if max_results is not None and len(results) >= max_results:
                            break
                        
                        if "FirstURL" in nested_topic and "Text" in nested_topic:
                            text = nested_topic.get("Text", "")
                            url = nested_topic.get("FirstURL", "")
                            
                            # Try to split title and snippet
                            if " - " in text:
                                parts = text.split(" - ", 1)
                                title = parts[0]
                                snippet = parts[1] if len(parts) > 1 else text
                            else:
                                title = text[:50] + "..." if len(text) > 50 else text
                                snippet = text
                            
                            search_result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                date=None,
                                last_updated=None,
                            )
                            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )

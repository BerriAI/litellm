"""
Calls DataForSEO SERP API to search the web.

DataForSEO API Reference: https://docs.dataforseo.com/v3/serp/google/organic/live/advanced/?bash
"""
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from litellm.constants import DEFAULT_DATAFORSEO_LOCATION_CODE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class DataForSEOSearchConfig(BaseSearchConfig):
    """
    Configuration for DataForSEO SERP API search.
    
    DataForSEO uses HTTP Basic Auth with login:password credentials.
    API endpoint: https://api.dataforseo.com/v3/serp/google/organic/live/advanced
    """
    
    DATAFORSEO_API_BASE = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "DataForSEO"
    
    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        DataForSEO uses POST requests with JSON body.
        """
        return "POST"
    
    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate DataForSEO environment and set up authentication.
        
        DataForSEO uses HTTP Basic Auth with login:password format.
        The credentials should be in DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD env vars,
        or passed as api_key in "login:password" format.
        """
        import base64

        # Get login and password
        login = get_secret_str("DATAFORSEO_LOGIN")
        password = get_secret_str("DATAFORSEO_PASSWORD")
        
        # If api_key is provided in "login:password" format, use it
        if api_key and ":" in api_key:
            login, password = api_key.split(":", 1)
        
        if not login:
            raise ValueError("DATAFORSEO_LOGIN is not set. Set `DATAFORSEO_LOGIN` environment variable or pass credentials in api_key parameter.")
        
        if not password:
            raise ValueError("DATAFORSEO_PASSWORD is not set. Set `DATAFORSEO_PASSWORD` environment variable or pass credentials in api_key parameter.")
        
        # Create Basic Auth header
        credentials = f"{login}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded_credentials}"
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
        Get complete URL for DataForSEO SERP API endpoint.
        
        DataForSEO uses POST requests, so no query parameters in URL.
        """
        return api_base or get_secret_str("DATAFORSEO_API_BASE") or self.DATAFORSEO_API_BASE

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict, List[Dict]]:
        """
        Transform Search request to DataForSEO SERP API format.
        
        Args:
            query: Search query (string or list of strings). DataForSEO supports single string queries.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results → maps to `depth` (max 700)
                - country: Country name → maps to `location_name`
                - search_domain_filter: Domain to filter results → maps to `domain`
                - Plus any DataForSEO-specific parameters (location_code, language_code, device, os, etc.)
            api_key: DataForSEO credentials (login:password format)
            
        Returns:
            List[Dict]: Request body for DataForSEO API (array of task objects as required by API)
        """
        # DataForSEO expects an array of task objects
        task: Dict[str, Any] = {}
        
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = query[0] if query else ""
        
        # Required field: keyword
        task["keyword"] = query
        
        # Map unified parameters to DataForSEO parameters
        if "max_results" in optional_params and optional_params["max_results"]:
            # DataForSEO uses 'depth' for number of results (max 700)
            depth = min(int(optional_params["max_results"]), 700)
            task["depth"] = depth
        
        if "country" in optional_params and optional_params["country"]:
            # DataForSEO uses location_code (e.g., 2840 for USA)
            # For simplicity, we'll use location_name which accepts country names
            task["location_name"] = optional_params["country"]
        
        if "search_domain_filter" in optional_params and optional_params["search_domain_filter"]:
            # DataForSEO uses 'domain' parameter to filter by domain
            task["domain"] = optional_params["search_domain_filter"]
        
        # Add defaults if not specified
        if "language_code" not in task and "language_name" not in task:
            task["language_code"] = "en"
        
        # DataForSEO requires a location - use default from constants if not specified
        if "location_code" not in task and "location_name" not in task:
            task["location_code"] = DEFAULT_DATAFORSEO_LOCATION_CODE
        
        # Pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in task:
                task[param] = value
        
        # DataForSEO API expects an array of tasks
        return [task]

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform DataForSEO SERP API response to LiteLLM unified SearchResponse format.
        
        DataForSEO → LiteLLM mappings:
        - tasks[0].result[*].items[*].title → SearchResult.title
        - tasks[0].result[*].items[*].url → SearchResult.url
        - tasks[0].result[*].items[*].description → SearchResult.snippet
        - No date/last_updated fields in standard response (set to None)
        
        Args:
            raw_response: Raw httpx response from DataForSEO API
            logging_obj: Logging object for tracking
            
        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()
        
        # Transform results to SearchResult objects
        results = []
        
        # DataForSEO wraps results in tasks array
        if "tasks" in response_json and len(response_json["tasks"]) > 0:
            task = response_json["tasks"][0]
            
            # Check if task was successful
            if task.get("status_code") == 20000 and "result" in task:
                # Result is an array, take first element
                if len(task["result"]) > 0:
                    result = task["result"][0]
                    
                    # Items contain the actual search results
                    for item in result.get("items", []):
                        # Only process organic search results
                        if item.get("type") == "organic":
                            search_result = SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                snippet=item.get("description", ""),
                                date=None,  # DataForSEO doesn't provide date in standard response
                                last_updated=None,
                            )
                            results.append(search_result)
        
        return SearchResponse(
            results=results,
            object="search",
        )


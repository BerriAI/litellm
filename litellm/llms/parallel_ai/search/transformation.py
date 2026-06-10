"""
Calls Parallel AI's /v1/search endpoint to search the web.

Parallel AI API Reference: https://docs.parallel.ai/api-reference/search/search
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
    include_domains: List[str]
    exclude_domains: List[str]
    after_date: str


class _ParallelAIExcerptSettings(TypedDict, total=False):
    max_chars_per_result: int


class _ParallelAIAdvancedSettings(TypedDict, total=False):
    source_policy: _ParallelAISourcePolicy
    excerpt_settings: _ParallelAIExcerptSettings
    fetch_policy: Dict
    location: str
    max_results: int


class ParallelAISearchRequest(TypedDict, total=False):
    """
    Parallel AI v1 Search API request format.
    Based on: https://docs.parallel.ai/api-reference/search/search
    """

    search_queries: List[str]  # Required - at least one keyword search query
    objective: str  # Optional - natural-language description of search goal
    mode: str  # Optional - 'turbo', 'basic', or 'advanced' (default 'advanced')
    max_chars_total: int  # Optional - upper bound on total excerpt characters
    session_id: str  # Optional - tracks calls across search/extract requests
    client_model: str  # Optional - model consuming the results
    advanced_settings: _ParallelAIAdvancedSettings


LEGACY_PROCESSOR_TO_MODE = {"base": "basic", "pro": "advanced"}


class ParallelAISearchConfig(BaseSearchConfig):
    PARALLEL_AI_API_BASE = "https://api.parallel.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Parallel AI"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        api_key = (
            api_key
            or get_secret_str("PARALLEL_AI_API_KEY")
            or get_secret_str("PARALLEL_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY is not set. Set `PARALLEL_API_KEY` environment variable."
            )
        headers["x-api-key"] = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("PARALLEL_AI_API_BASE")
            or self.PARALLEL_AI_API_BASE
        )

        if not api_base.endswith("/v1/search"):
            api_base = f"{api_base.rstrip('/')}/v1/search"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to Parallel AI v1 API format.

        Args:
            query: Search query (string or list of strings)
                - If string: maps to `search_queries` (single item) and `objective`
                - If list: maps to `search_queries` (keyword queries)
            optional_params: Optional parameters for the request
                - mode: Search mode ('turbo', 'basic', 'advanced'; API default 'advanced')
                - max_results: Maximum number of search results -> `advanced_settings.max_results`
                - search_domain_filter: Domains to include -> `advanced_settings.source_policy.include_domains`
                - exclude_domains: Domains to exclude -> `advanced_settings.source_policy.exclude_domains`
                - country: ISO 3166-1 alpha-2 code -> `advanced_settings.location`
                - max_chars_per_result: -> `advanced_settings.excerpt_settings.max_chars_per_result`
                - processor: Legacy v1beta param; 'base' maps to mode 'basic', 'pro' to 'advanced'

        Returns:
            Dict with typed request data following ParallelAISearchRequest spec
        """
        request_data: ParallelAISearchRequest = {}

        if isinstance(query, list):
            request_data["search_queries"] = query
        else:
            request_data["search_queries"] = [query]
            request_data["objective"] = query

        if "mode" in optional_params:
            request_data["mode"] = optional_params["mode"]
        elif "processor" in optional_params:
            request_data["mode"] = LEGACY_PROCESSOR_TO_MODE.get(
                optional_params["processor"], optional_params["processor"]
            )

        advanced_settings: _ParallelAIAdvancedSettings = {}

        if "max_results" in optional_params:
            advanced_settings["max_results"] = optional_params["max_results"]

        if "country" in optional_params:
            advanced_settings["location"] = optional_params["country"]

        if "max_chars_per_result" in optional_params:
            advanced_settings["excerpt_settings"] = {
                "max_chars_per_result": optional_params["max_chars_per_result"]
            }

        source_policy: _ParallelAISourcePolicy = {}

        if "search_domain_filter" in optional_params:
            source_policy["include_domains"] = optional_params["search_domain_filter"]

        if "exclude_domains" in optional_params:
            source_policy["exclude_domains"] = optional_params["exclude_domains"]

        if source_policy:
            advanced_settings["source_policy"] = source_policy

        if "advanced_settings" in optional_params:
            advanced_settings.update(optional_params["advanced_settings"])

        if advanced_settings:
            request_data["advanced_settings"] = advanced_settings

        result_data = dict(request_data)

        handled_params = self.get_supported_perplexity_optional_params() | {
            "mode",
            "processor",
            "exclude_domains",
            "max_chars_per_result",
            "advanced_settings",
        }
        for param, value in optional_params.items():
            if param not in handled_params and param not in result_data:
                result_data[param] = value

        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Parallel AI v1 API response to LiteLLM unified SearchResponse format.

        Parallel AI -> LiteLLM mappings:
        - results[].title -> SearchResult.title
        - results[].url -> SearchResult.url
        - results[].excerpts (array) -> SearchResult.snippet (joined string)
        - results[].publish_date -> SearchResult.date
        """
        response_json = raw_response.json()

        results = []
        for result in response_json.get("results", []):
            excerpts = result.get("excerpts") or []
            snippet = " ... ".join(excerpts) if excerpts else ""

            search_result = SearchResult(
                title=result.get("title") or "",
                url=result.get("url") or "",
                snippet=snippet,
                date=result.get("publish_date"),
                last_updated=None,
            )
            results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

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
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("PARALLEL_AI_API_KEY", "PARALLEL_API_KEY"),
            base_env_var="PARALLEL_AI_API_BASE",
            default_api_base=self.PARALLEL_AI_API_BASE,
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

        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1/search"):
            api_base = f"{api_base.removesuffix('/v1')}/v1/search"

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
                - mode: Search mode ('turbo', 'basic', 'advanced'); defaults to 'basic'
                - processor: Legacy v1beta param; 'base' maps to mode 'basic', 'pro' to 'advanced'
                - max_results: Maximum number of search results -> `advanced_settings.max_results`
                - search_domain_filter: Domains to include -> `advanced_settings.source_policy.include_domains`
                - exclude_domains: Domains to exclude -> `advanced_settings.source_policy.exclude_domains`
                - country: ISO 3166-1 alpha-2 code -> `advanced_settings.location`
                - max_chars_per_result: -> `advanced_settings.excerpt_settings.max_chars_per_result`
                - Any other params are passed through to the request body as-is

        Returns:
            Dict with request data following the v1 search request spec
        """
        params = dict(optional_params)

        request_data: ParallelAISearchRequest = {}

        if isinstance(query, list):
            request_data["search_queries"] = query
        else:
            request_data["search_queries"] = [query]
            request_data["objective"] = query

        mode = params.pop("mode", None)
        processor = params.pop("processor", None)
        if mode is None and processor is not None:
            mode = LEGACY_PROCESSOR_TO_MODE.get(processor, processor)
        # the v1 API defaults to 'advanced' when mode is omitted; default to 'basic'
        # instead to keep v1beta's default tier (processor 'base') and litellm's
        # $0.004/query cost map entry for `parallel_ai/search` accurate
        request_data["mode"] = mode or "basic"

        advanced_settings: _ParallelAIAdvancedSettings = {}

        if "max_results" in params:
            advanced_settings["max_results"] = params.pop("max_results")

        if "country" in params:
            advanced_settings["location"] = params.pop("country")

        if "max_chars_per_result" in params:
            advanced_settings["excerpt_settings"] = {
                "max_chars_per_result": params.pop("max_chars_per_result")
            }

        source_policy: _ParallelAISourcePolicy = {}

        if "search_domain_filter" in params:
            source_policy["include_domains"] = params.pop("search_domain_filter")

        if "exclude_domains" in params:
            source_policy["exclude_domains"] = params.pop("exclude_domains")

        if source_policy:
            advanced_settings["source_policy"] = source_policy

        advanced_settings.update(params.pop("advanced_settings", {}))

        if advanced_settings:
            request_data["advanced_settings"] = advanced_settings

        # unified-spec param with no v1 equivalent
        params.pop("max_tokens_per_page", None)

        result_data: Dict = dict(request_data)
        result_data.update(params)
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

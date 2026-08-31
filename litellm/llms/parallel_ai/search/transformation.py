"""
Calls Parallel AI's /v1/search endpoint to search the web.

Parallel AI API Reference: https://docs.parallel.ai/api-reference/search/search
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypedDict

import httpx
from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.parallel_ai.search.cost_calculator import PARALLEL_AI_USAGE_PARAM
from litellm.secret_managers.main import get_secret_str


class _ParallelAIV1SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = ""
    title: str | None = None
    publish_date: str | None = None
    excerpts: Sequence[str] = ()


class _ParallelAIV1SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_id: str | None = None
    session_id: str | None = None
    results: Sequence[_ParallelAIV1SearchResult] = ()
    usage: Sequence[Mapping[str, object]] | None = None
    warnings: Sequence[Mapping[str, object]] | None = None


class _ParallelAISourcePolicy(TypedDict, total=False):
    include_domains: list[str]
    exclude_domains: list[str]
    after_date: str


class _ParallelAIExcerptSettings(TypedDict, total=False):
    max_chars_per_result: int


class _ParallelAIFetchPolicy(TypedDict, total=False):
    max_age_seconds: ReadOnly[int]
    timeout_seconds: ReadOnly[float]
    disable_cache_fallback: ReadOnly[bool]


class _ParallelAIAdvancedSettings(TypedDict, total=False):
    source_policy: _ParallelAISourcePolicy
    excerpt_settings: _ParallelAIExcerptSettings
    fetch_policy: _ParallelAIFetchPolicy
    location: str
    max_results: int


class ParallelAISearchRequest(TypedDict, total=False):
    """
    Parallel AI v1 Search API request format.
    Based on: https://docs.parallel.ai/api-reference/search/search
    """

    search_queries: list[str]  # Required - at least one keyword search query
    objective: str  # Optional - natural-language description of search goal
    mode: str  # Optional - 'turbo', 'fast', 'basic', or 'advanced' (default 'advanced')
    max_chars_total: int  # Optional - upper bound on total excerpt characters
    session_id: str  # Optional - tracks calls across search/extract requests
    client_model: str  # Optional - model consuming the results
    advanced_settings: _ParallelAIAdvancedSettings


LEGACY_PROCESSOR_TO_MODE: Final = MappingProxyType({"base": "basic", "pro": "advanced"})


class ParallelAISearchConfig(BaseSearchConfig):
    PARALLEL_AI_API_BASE = "https://api.parallel.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Parallel AI"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("PARALLEL_AI_API_KEY", "PARALLEL_API_KEY"),
            base_env_var="PARALLEL_AI_API_BASE",
            default_api_base=self.PARALLEL_AI_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("PARALLEL_API_KEY is not set. Set `PARALLEL_API_KEY` environment variable.")
        headers["x-api-key"] = resolved_api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: dict | list[dict] | None = None,
        **kwargs,
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("PARALLEL_AI_API_BASE") or self.PARALLEL_AI_API_BASE

        trimmed: Final = resolved_api_base.rstrip("/")
        if trimmed.endswith("/v1/search"):
            return trimmed
        return f"{trimmed.removesuffix('/v1')}/v1/search"

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        """
        Transform Search request to Parallel AI v1 API format.

        Args:
            query: Search query (string or list of strings)
                - If string: maps to `search_queries` (single item) and `objective`
                - If list: maps to `search_queries` (keyword queries)
            optional_params: Optional parameters for the request
                - mode: Search mode ('turbo', 'fast', 'basic', 'advanced'); defaults to 'basic'
                - processor: Legacy v1beta param; 'base' maps to mode 'basic', 'pro' to 'advanced'
                - max_results: Maximum number of search results -> `advanced_settings.max_results`
                - search_domain_filter / include_domains: Domains to include -> `advanced_settings.source_policy.include_domains`
                - exclude_domains: Domains to exclude -> `advanced_settings.source_policy.exclude_domains`
                - after_date: RFC 3339 date (YYYY-MM-DD) -> `advanced_settings.source_policy.after_date`
                - country / location: ISO 3166-1 alpha-2 code -> `advanced_settings.location`
                - max_chars_per_result: -> `advanced_settings.excerpt_settings.max_chars_per_result`
                - fetch_policy: Cache vs live-fetch policy -> `advanced_settings.fetch_policy`
                - Any other params (objective, max_chars_total, session_id, client_model, ...)
                  are passed through to the request body as-is

        Returns:
            Dict with request data following the v1 search request spec
        """
        params: Final = dict(optional_params)

        request_data: Final[ParallelAISearchRequest] = {}

        if isinstance(query, list):
            request_data["search_queries"] = query
        else:
            request_data["search_queries"] = [query]
            request_data["objective"] = query

        mode = params.pop("mode", None)
        processor: Final = params.pop("processor", None)
        if mode is None and processor is not None:
            mode = LEGACY_PROCESSOR_TO_MODE.get(processor, processor)
        # the v1 API defaults to 'advanced' when mode is omitted; default to 'basic'
        # instead to keep v1beta's default tier (processor 'base') and litellm's
        # cost map entry for `parallel_ai/search` accurate
        request_data["mode"] = mode or "basic"

        advanced_settings: Final[_ParallelAIAdvancedSettings] = {}

        if "max_results" in params:
            advanced_settings["max_results"] = params.pop("max_results")

        if "country" in params:
            advanced_settings["location"] = params.pop("country")

        if "location" in params:
            advanced_settings["location"] = params.pop("location")

        if "max_chars_per_result" in params:
            advanced_settings["excerpt_settings"] = {"max_chars_per_result": params.pop("max_chars_per_result")}

        if "fetch_policy" in params:
            advanced_settings["fetch_policy"] = params.pop("fetch_policy")

        source_policy: Final[_ParallelAISourcePolicy] = {}

        if "search_domain_filter" in params:
            source_policy["include_domains"] = params.pop("search_domain_filter")

        if "include_domains" in params:
            source_policy["include_domains"] = params.pop("include_domains")

        if "exclude_domains" in params:
            source_policy["exclude_domains"] = params.pop("exclude_domains")

        if "after_date" in params:
            source_policy["after_date"] = params.pop("after_date")

        if source_policy:
            advanced_settings["source_policy"] = source_policy

        advanced_settings.update(params.pop("advanced_settings", {}))

        if advanced_settings:
            request_data["advanced_settings"] = advanced_settings

        # unified-spec param with no v1 equivalent
        params.pop("max_tokens_per_page", None)

        # reserved for the provider's own reported usage, which prices the request;
        # a caller-supplied value would otherwise set its own cost
        params.pop(PARALLEL_AI_USAGE_PARAM, None)

        return {**request_data, **params}

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
        - results[].excerpts (array) -> SearchResult.snippet (joined string); the raw
          array is preserved as an extra `excerpts` field on each result
        - results[].publish_date -> SearchResult.date
        - search_id / session_id / warnings are preserved as extra fields on the
          response; usage is preserved as `parallel_usage` (the `usage` name is
          reserved for LiteLLM's token-usage object)
        """
        parsed: Final = _ParallelAIV1SearchResponse.model_validate(raw_response.json())

        # written unconditionally: leaving a caller-supplied value in place when the
        # provider reports no usage would let the caller price its own request
        logging_obj.optional_params = {
            **logging_obj.optional_params,
            PARALLEL_AI_USAGE_PARAM: parsed.usage,
        }

        results: Final = tuple(
            SearchResult.model_validate(
                MappingProxyType(
                    {
                        "title": result.title or "",
                        "url": result.url,
                        "snippet": " ... ".join(result.excerpts) if result.excerpts else "",
                        "date": result.publish_date,
                        "last_updated": None,
                        "excerpts": result.excerpts,
                    }
                )
            )
            for result in parsed.results
        )

        extra_fields: Final = MappingProxyType(
            {
                key: value
                for key, value in (
                    ("search_id", parsed.search_id),
                    ("session_id", parsed.session_id),
                    ("parallel_usage", parsed.usage),
                    ("warnings", parsed.warnings),
                )
                if value is not None
            }
        )

        return SearchResponse.model_validate(MappingProxyType({"results": results, "object": "search", **extra_fields}))

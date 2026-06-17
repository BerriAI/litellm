"""
TinyFish Search API.
Endpoint: GET https://api.search.tinyfish.ai
Docs: https://docs.tinyfish.ai/search-api
"""

from __future__ import annotations

from typing import Literal, TypedDict
from urllib.parse import urlencode

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _TinyfishSearchRequestRequired(TypedDict):
    query: str


class TinyfishSearchRequest(_TinyfishSearchRequestRequired, total=False):
    location: str
    language: str
    page: int
    include_thumbnail: bool
    max_results: int


_TINYFISH_PARAMS_KEY = "_tinyfish_params"


class TinyfishSearchConfig(BaseSearchConfig):
    TINYFISH_API_BASE = "https://api.search.tinyfish.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "TinyFish"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    def validate_environment(
        self,
        headers: dict[str, str],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,
    ) -> dict[str, str]:
        resolved_key = api_key or get_secret_str("TINYFISH_API_KEY")
        if not resolved_key:
            raise ValueError(
                "TINYFISH_API_KEY is not set. Set `TINYFISH_API_KEY` environment variable."
            )
        return {**headers, "X-API-Key": resolved_key, "Accept": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],
        data: dict[str, object] | list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> str:
        resolved_base = (
            api_base or get_secret_str("TINYFISH_API_BASE") or self.TINYFISH_API_BASE
        )
        if isinstance(data, dict) and _TINYFISH_PARAMS_KEY in data:
            params = data[_TINYFISH_PARAMS_KEY]
            if isinstance(params, dict):
                query_string = urlencode(params, doseq=True)
                return f"{resolved_base}?{query_string}"
        return resolved_base

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        resolved_query = " ".join(query) if isinstance(query, list) else query

        request_data: TinyfishSearchRequest = {"query": resolved_query}

        country = optional_params.get("country")
        if isinstance(country, str):
            request_data["location"] = country

        raw_max = optional_params.get("max_results")
        if raw_max is not None:
            request_data["max_results"] = max(1, min(int(raw_max), 20))

        domain_filter = optional_params.get("search_domain_filter")
        if isinstance(domain_filter, list) and len(domain_filter) > 0:
            request_data["query"] = _append_domain_filters(
                request_data["query"],
                [str(d) for d in domain_filter],
            )

        result_data: dict[str, object] = dict(request_data)

        supported_perplexity = self.get_supported_perplexity_optional_params()
        for param, value in optional_params.items():
            if param not in supported_perplexity and param not in result_data:
                result_data[param] = value

        return {_TINYFISH_PARAMS_KEY: result_data}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None,
        **kwargs: object,
    ) -> SearchResponse:
        response_json: dict[str, object] = raw_response.json()

        raw_max: object = {}
        if raw_response.request:
            raw_max = raw_response.request.url.params.get("max_results", 20)
        max_results = min(int(raw_max) if raw_max else 20, 20)

        raw_results = response_json.get("results")
        items: list[object] = list(raw_results) if isinstance(raw_results, list) else []

        results = tuple(
            SearchResult(
                title=item.get("title", "") if isinstance(item, dict) else "",
                url=item.get("url", "") if isinstance(item, dict) else "",
                snippet=item.get("snippet", "") if isinstance(item, dict) else "",
            )
            for item in items[:max_results]
        )

        return SearchResponse(results=list(results), object="search")


def _append_domain_filters(query: str, domains: list[str]) -> str:
    domain_clauses = " OR ".join(f"site:{d}" for d in domains)
    return f"({query}) AND ({domain_clauses})"

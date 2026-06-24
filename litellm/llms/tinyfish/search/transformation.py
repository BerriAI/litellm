"""
TinyFish Search API.
Endpoint: GET https://api.search.tinyfish.ai
Docs: https://docs.tinyfish.ai/search-api
"""

from __future__ import annotations

from typing import Literal, TypedDict
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

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


class _TinyfishResultItem(BaseModel, frozen=True):
    title: str = ""
    url: str = ""
    snippet: str = ""


class _TinyfishApiResponse(BaseModel, frozen=True):
    results: tuple[_TinyfishResultItem, ...] = ()


_UrlEncodableParams = TypeAdapter(dict[str, str | int | bool])
_StrList = TypeAdapter(list[str])
_StrFrozenSet = TypeAdapter(frozenset[str])

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
        resolved_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("TINYFISH_API_KEY",),
            base_env_var="TINYFISH_API_BASE",
            default_api_base=self.TINYFISH_API_BASE,
        )
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
            validated_params = _UrlEncodableParams.validate_python(
                data[_TINYFISH_PARAMS_KEY]
            )
            return f"{resolved_base}?{urlencode(validated_params, doseq=True)}"
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
        if isinstance(raw_max, (int, float, str)):
            request_data["max_results"] = max(1, min(int(raw_max), 20))

        try:
            domains = _StrList.validate_python(
                optional_params.get("search_domain_filter")
            )
        except (ValidationError, TypeError):
            domains = []
        if domains:
            request_data["query"] = _append_domain_filters(
                request_data["query"], domains
            )

        result_data: dict[str, object] = dict(request_data)

        raw_supported: object = (
            self.get_supported_perplexity_optional_params()  # any-ok: base class returns bare set
        )
        supported_perplexity = _StrFrozenSet.validate_python(raw_supported)
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
        raw_json: object = raw_response.json()  # any-ok: httpx Response.json() -> Any
        parsed = _TinyfishApiResponse.model_validate(raw_json)

        max_results_str: str = "20"
        if raw_response.request:
            raw_param: object = (
                raw_response.request.url.params.get(  # any-ok: httpx QueryParams.get() -> Any
                    "max_results", "20"
                )
            )
            max_results_str = str(raw_param)
        max_results: int = min(int(max_results_str), 20)

        results = [
            SearchResult(title=item.title, url=item.url, snippet=item.snippet)
            for item in parsed.results[:max_results]
        ]

        return SearchResponse(results=results, object="search")


def _append_domain_filters(query: str, domains: list[str]) -> str:
    domain_clauses = " OR ".join(f"site:{d}" for d in domains)
    return f"({query}) ({domain_clauses})"

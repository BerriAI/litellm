"""
Calls Keenable's /v1/search endpoint to search the web.

Keenable API Reference: https://docs.keenable.ai/api-reference/search
"""

from collections.abc import Iterable, Mapping
from typing import Final, TypedDict

import httpx
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import NotRequired, ReadOnly

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))
_UNIFIED_PARAMS: Final = frozenset({"max_results", "search_domain_filter", "country", "max_tokens_per_page"})


class KeenableSearchRequest(TypedDict):
    """
    Keenable Search API request format.
    Based on: https://docs.keenable.ai/api-reference/search
    """

    query: ReadOnly[str]
    mode: NotRequired[ReadOnly[str]]
    site: NotRequired[ReadOnly[str]]
    acquired_after: NotRequired[ReadOnly[str]]
    acquired_before: NotRequired[ReadOnly[str]]
    published_after: NotRequired[ReadOnly[str]]
    published_before: NotRequired[ReadOnly[str]]
    query_time: NotRequired[ReadOnly[str]]
    snippet_max_length: NotRequired[ReadOnly[int]]
    max_results: NotRequired[ReadOnly[int]]


def _first_domain(domains: object) -> str | None:
    if isinstance(domains, list) and domains and isinstance(domains[0], str) and domains[0]:
        return domains[0]
    if isinstance(domains, str) and domains:
        return domains
    return None


def _search_result(raw_result: object) -> SearchResult:
    result: Final = _JSON_OBJECT.validate_python(raw_result)
    return SearchResult.model_validate(
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet") or result.get("description") or "",
            "date": result.get("published_at"),
            "last_updated": result.get("acquired_at"),
        }
    )


class KeenableSearchConfig(BaseSearchConfig):
    KEENABLE_API_BASE = "https://api.keenable.ai/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Keenable"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Authenticated calls send ``X-API-Key``. Without a key, Keenable's public
        search tier still works and requires ``X-Keenable-Title`` instead.
        """
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("KEENABLE_API_KEY",),
            base_env_var="KEENABLE_API_BASE",
            default_api_base=self.KEENABLE_API_BASE,
        )
        headers["Content-Type"] = "application/json"
        if api_key:
            headers["X-API-Key"] = api_key
        else:
            headers["X-Keenable-Title"] = get_secret_str("KEENABLE_TITLE") or "LiteLLM"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: dict | list[dict] | None = None,
        **kwargs,
    ) -> str:
        api_key: Final = kwargs.get("api_key") or get_secret_str("KEENABLE_API_KEY")
        route: Final = "search" if api_key else "search/public"
        return ensure_api_route(api_base or get_secret_str("KEENABLE_API_BASE") or self.KEENABLE_API_BASE, route)

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        """
        Transform Search request to Keenable API format.

        Transforms Perplexity unified spec parameters:
        - query → query (same)
        - max_results → max_results (same)
        - search_domain_filter → site (first domain only; Keenable takes a single site)
        - country → (not applicable, ignored)
        - max_tokens_per_page → (not applicable, ignored; Keenable's snippet_max_length is
          characters in 180–10000, not tokens)

        All other Keenable-specific parameters are passed through as-is.
        """
        site: Final = _first_domain(optional_params.get("search_domain_filter"))
        unified: Final[KeenableSearchRequest] = {
            "query": " ".join(query) if isinstance(query, list) else query,
            **({"max_results": optional_params["max_results"]} if "max_results" in optional_params else {}),
            **({"site": site} if site is not None else {}),
        }
        passthrough: Final = {
            param: value
            for param, value in optional_params.items()
            if param not in _UNIFIED_PARAMS and param not in unified
        }
        return {**unified, **passthrough}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Keenable API response to LiteLLM unified SearchResponse format.

        Keenable → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].snippet → SearchResult.snippet
        - results[].published_at → SearchResult.date
        - results[].acquired_at → SearchResult.last_updated
        - every top-level key except results → SearchResponse.provider_metadata
        """
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())
        return SearchResponse.model_validate(
            {
                "results": tuple(
                    _search_result(raw_result)
                    for raw_result in _JSON_VALUES.validate_python(response_json.get("results", []))
                ),
                "object": "search",
                "provider_metadata": {key: value for key, value in response_json.items() if key != "results"},
            }
        )

"""
Calls SerpApi's search endpoint.

SerpApi API Reference: https://serpapi.com/search-api
"""

from typing import Final, Literal, TypedDict
from urllib.parse import urlencode

import httpx
from typing_extensions import ReadOnly

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _SerpApiSearchRequestRequired(TypedDict):
    engine: ReadOnly[str]
    q: ReadOnly[str]
    api_key: ReadOnly[str]


class SerpApiSearchRequest(_SerpApiSearchRequestRequired, total=False):
    """
    SerpApi request format.
    Based on: https://serpapi.com/search-api
    """

    gl: ReadOnly[str]
    hl: ReadOnly[str]
    location: ReadOnly[str]
    device: ReadOnly[str]
    safe: ReadOnly[str]
    num: ReadOnly[int]
    start: ReadOnly[int]
    tbs: ReadOnly[str]
    tbm: ReadOnly[str]


class SerpApiSearchConfig(BaseSearchConfig):
    SERPAPI_API_BASE: Final = "https://serpapi.com/search.json"

    @staticmethod
    def ui_friendly_name() -> str:
        return "SerpApi"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: base class override
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: base class override
    ) -> dict[str, str]:  # mutable-ok: base class override
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SERPAPI_API_KEY",),
            base_env_var="SERPAPI_API_BASE",
            default_api_base=self.SERPAPI_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("SERPAPI_API_KEY is not set. Set `SERPAPI_API_KEY` environment variable.")
        return {
            **headers,
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: base class override
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base class override
        **kwargs: object,  # kwargs-ok: base class override
    ) -> str:
        base: Final = api_base or get_secret_str("SERPAPI_API_BASE") or self.SERPAPI_API_BASE
        if data and isinstance(data, dict) and "_serpapi_params" in data:
            params: Final = data["_serpapi_params"]
            if isinstance(params, dict):
                return f"{base}?{urlencode(params, doseq=True)}"  # pyright: ignore[reportUnknownArgumentType]  # params is a dict
        return base

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: base class override
        optional_params: dict[str, object],  # mutable-ok: base class override
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: base class override
    ) -> dict[str, object]:  # mutable-ok: base class override
        q_str: Final = " ".join(query) if isinstance(query, list) else query

        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SERPAPI_API_KEY",),
            base_env_var="SERPAPI_API_BASE",
            default_api_base=self.SERPAPI_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("SERPAPI_API_KEY is not set. Set `SERPAPI_API_KEY` environment variable.")

        engine: Final = str(optional_params.get("engine", "google"))

        domains: Final = optional_params.get("search_domain_filter")
        q_final: Final = (
            f"({q_str}) ({' OR '.join(f'site:{d}' for d in domains)})"  # pyright: ignore[reportUnknownVariableType]  # items are strings
            if isinstance(domains, list) and len(domains) > 0  # pyright: ignore[reportUnknownArgumentType]  # domains is a list
            else q_str
        )

        base_params: Final[dict[str, object]] = {  # mutable-ok: mapping type
            "engine": engine,
            "q": q_final,
            "api_key": resolved_api_key,
        }

        optional_mappings: Final[dict[str, object]] = {  # mutable-ok: mapping type
            **({"num": optional_params["max_results"]} if "max_results" in optional_params else {}),
            **({"gl": str(optional_params["country"]).lower()} if "country" in optional_params else {}),
        }

        extra_params: Final[dict[str, object]] = {  # mutable-ok: mapping type
            param: value
            for param, value in optional_params.items()
            if param not in self.get_supported_perplexity_optional_params()
            and param not in base_params
            and param not in optional_mappings
        }

        request_data: Final[dict[str, object]] = {  # mutable-ok: mapping type
            **base_params,
            **optional_mappings,
            **extra_params,
        }

        return {"_serpapi_params": request_data}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: base class override
    ) -> SearchResponse:
        """
        Transform SerpApi response to LiteLLM unified SearchResponse format.

        SerpApi -> LiteLLM mappings:
        - organic_results[].title -> SearchResult.title
        - organic_results[].link  -> SearchResult.url
        - organic_results[].snippet -> SearchResult.snippet
        - organic_results[].date -> SearchResult.date (optional)
        - organic_results[].position -> SearchResult (preserved via extra fields)
        """
        response_json: Final = raw_response.json()

        results_list: Final = (  # pyright: ignore[reportUnknownVariableType]  # standard response shape
            response_json.get("organic_results")
            or response_json.get("news_results")
            or response_json.get("shopping_results")
            or response_json.get("local_results")
            or response_json.get("jobs_results")
            or []
        )

        results: Final = [
            SearchResult(
                title=result.get("title", ""),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # parsing generic json dict
                url=result.get("link", ""),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # parsing generic json dict
                snippet=result.get("snippet", ""),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # parsing generic json dict
                date=result.get("date"),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # parsing generic json dict
                last_updated=None,
            )
            for result in results_list  # pyright: ignore[reportUnknownVariableType]  # iterating generic json array
        ]

        return SearchResponse(
            results=results,
            object="search",
        )

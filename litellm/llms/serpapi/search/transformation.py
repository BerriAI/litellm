"""
Calls SerpApi's Search API endpoint.

SerpApi API Reference: https://serpapi.com/search-api
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)

_SERPAPI_PARAMS_KEY: Final = "_serpapi_params"
_SERPAPI_REQUEST_KEYS: Final = frozenset(("engine", "q", "num", "gl"))
_SEARCH_RESULT_RESERVED_FIELDS: Final = frozenset(SearchResult.model_fields)
_SerpApiUrlParams: Final = TypeAdapter(dict[str, str | int | float | bool | list[str]])
_StringTuple: Final = TypeAdapter(tuple[str, ...])
_StringFrozenSet: Final = TypeAdapter(frozenset[str])
_SerpApiResultExtras: Final = TypeAdapter(dict[str, object])


class _SerpApiOrganicResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    title: str | None = None
    link: str | None = None
    snippet: str | None = None
    date: str | None = None


class _SerpApiSearchMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    status: str | None = None


class _SerpApiSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    organic_results: tuple[_SerpApiOrganicResult, ...] = ()
    search_metadata: _SerpApiSearchMetadata | None = None
    error: str | None = None


class SerpApiSearchConfig(BaseSearchConfig):
    SERPAPI_API_BASE: Final = "https://serpapi.com/search.json"

    def __init__(self) -> None:
        super().__init__()
        self._max_results: int | None = None

    @staticmethod
    def ui_friendly_name() -> str:
        return "SerpApi"

    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        SerpApi uses GET requests for search.
        """
        return "GET"

    def _resolve_api_key(
        self,
        api_key: str | None,
    ) -> str:
        """
        Resolve a caller or configured SerpApi key.
        """
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=None,
            key_env_vars=("SERPAPI_KEY", "SERPAPI_API_KEY"),
            base_env_var=None,
            default_api_base=self.SERPAPI_API_BASE,
        )
        if not resolved_key:
            raise ValueError("SERPAPI_KEY is not set. Set `SERPAPI_KEY` or `SERPAPI_API_KEY` environment variable.")
        return resolved_key

    def validate_environment(
        self,
        headers: Mapping[str, str],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig provider interface forwards extensible request options
    ) -> dict[str, str]:  # mutable-ok: BaseSearchConfig handler contract requires a mutable header dict
        """
        Validate SerpApi credentials and return request headers.
        """
        self._resolve_api_key(api_key=api_key)
        resolved_headers: Final = MappingProxyType({**headers, "Content-Type": "application/json"})
        return resolved_headers.copy()

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: mirrors the BaseSearchConfig override contract
        data: dict[str, object]  # mutable-ok: mirrors the BaseSearchConfig override contract
        | list[dict[str, object]]
        | None = None,
        api_key: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig provider interface forwards extensible request options
    ) -> str:
        """
        Get complete URL for Search endpoint with query parameters.

        SerpApi uses GET requests and includes api_key in query params.
        """
        resolved_base: Final = self.SERPAPI_API_BASE
        if not isinstance(data, Mapping) or _SERPAPI_PARAMS_KEY not in data:
            return resolved_base

        try:
            params: Final = _SerpApiUrlParams.validate_python(data[_SERPAPI_PARAMS_KEY])
        except ValidationError as exc:
            invalid_params: Final = ", ".join(
                sorted(frozenset(str(error["loc"][0]) for error in exc.errors() if error["loc"]))
            )
            raise ValueError(
                f"Invalid SerpApi URL parameter value for: {invalid_params or 'request parameters'}"
            ) from None
        resolved_key: Final = self._resolve_api_key(api_key=api_key)
        query_params: Final = tuple(
            (
                key,
                str(value).lower() if isinstance(value, bool) else value,
            )
            for key, value in params.items()
        ) + (("api_key", resolved_key),)
        query_string: Final = urlencode(query_params, doseq=True)
        separator: Final = "&" if "?" in resolved_base else "?"
        return f"{resolved_base}{separator}{query_string}"

    def transform_search_request(
        self,
        query: str | Sequence[str],
        optional_params: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: BaseSearchConfig provider interface forwards extensible request options
    ) -> dict[str, object]:  # mutable-ok: BaseSearchConfig request contract requires a JSON dict
        """
        Transform Search request to SerpApi format.

        Transforms unified spec parameters:
        - query -> q
        - max_results -> num
        - search_domain_filter -> q (append site: filters)
        - country -> gl

        Args:
            query: Search query (string or sequence of strings)
            optional_params: Optional parameters for the request

        Returns:
            Dict containing SerpApi query parameters for URL construction
        """
        base_query: Final = " ".join(query) if not isinstance(query, str) else query
        raw_domains: Final = optional_params.get("search_domain_filter")
        domains: Final = _StringTuple.validate_python(raw_domains or ())
        resolved_query: Final = self._append_domain_filters(base_query, domains) if domains else base_query

        engine: Final = optional_params.get("engine")
        max_results: Final = optional_params.get("max_results")
        country: Final = optional_params.get("country")
        resolved_max_results: Final = (
            max_results
            if isinstance(max_results, int) and not isinstance(max_results, bool) and max_results > 0
            else None
        )
        self._max_results = resolved_max_results
        num_param: Final[Mapping[str, int]] = (
            MappingProxyType({"num": resolved_max_results})
            if resolved_max_results is not None
            else MappingProxyType({})
        )
        country_param: Final[Mapping[str, str]] = (
            MappingProxyType({"gl": country.lower()}) if isinstance(country, str) else MappingProxyType({})
        )
        supported_params: Final = _StringFrozenSet.validate_python(
            self.get_supported_perplexity_optional_params()  # pyright: ignore[reportUnknownMemberType]  # base returns bare set
        )
        passthrough_params: Final = MappingProxyType(
            {
                param: value
                for param, value in optional_params.items()
                if value is not None and param not in supported_params and param not in _SERPAPI_REQUEST_KEYS
            }
        )
        request_data: Final = MappingProxyType(
            {
                "engine": engine if isinstance(engine, str) else "google",
                "q": resolved_query,
                **num_param,
                **country_param,
                **passthrough_params,
            }
        )
        serializable_request_data: Final[object] = request_data.copy()
        request_payload: Final[MappingProxyType[str, object]] = MappingProxyType(
            {_SERPAPI_PARAMS_KEY: serializable_request_data}
        )
        return request_payload.copy()

    @staticmethod
    def _append_domain_filters(query: str, domains: Sequence[str]) -> str:
        """
        Add site: filters to restrict search to specific domains.
        """
        domain_clauses: Final = " OR ".join(f"site:{domain}" for domain in domains)
        return f"({query}) ({domain_clauses})"

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig provider interface forwards extensible response options
    ) -> SearchResponse:
        """
        Transform SerpApi response to LiteLLM unified SearchResponse format.

        SerpApi -> LiteLLM mappings:
        - organic_results[].title -> SearchResult.title
        - organic_results[].link -> SearchResult.url
        - organic_results[].snippet -> SearchResult.snippet
        - organic_results[].date -> SearchResult.date

        Args:
            raw_response: Raw httpx response from SerpApi
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_headers: Final = raw_response.headers
        if not 200 <= raw_response.status_code < 300:
            raise BaseLLMException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        try:
            payload: Final[object] = raw_response.json()  # pyright: ignore[reportAny]  # httpx returns Any
        except ValueError as exc:
            raise BaseLLMException(
                message=f"Expected a JSON body from SerpApi, got: {raw_response.text[:200]}",
                status_code=raw_response.status_code,
                headers=response_headers,
            ) from exc

        try:
            response: Final = _SerpApiSearchResponse.model_validate(payload)
        except ValidationError as exc:
            raise BaseLLMException(
                message=f"Unrecognized SerpApi response shape: {exc}",
                status_code=raw_response.status_code,
                headers=response_headers,
            ) from exc
        if response.search_metadata is not None and response.search_metadata.status == "Error":
            raise BaseLLMException(
                message=response.error or raw_response.text,
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        results: Final = tuple(
            SearchResult(
                title=result.title or "",
                url=result.link or "",
                snippet=result.snippet or "",
                date=result.date,
                last_updated=None,
                **MappingProxyType(
                    {
                        key: value
                        for key, value in _SerpApiResultExtras.validate_python(
                            result.model_extra or MappingProxyType({})
                        ).items()
                        if key not in _SEARCH_RESULT_RESERVED_FIELDS
                    }
                ),
            )
            for result in response.organic_results
        )
        limited_results: Final = results[: self._max_results] if self._max_results is not None else results
        return SearchResponse(
            results=list(limited_results),  # mutable-ok: SearchResponse schema requires a list
            object="search",
        )

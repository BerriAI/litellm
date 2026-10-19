from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class StaanSearchRequest(TypedDict, total=False):
    q: ReadOnly[str]
    market: ReadOnly[Literal["fr-fr", "en-us", "de-de"]]
    offset: ReadOnly[Literal[0, 10, 20, 30]]
    include_domains: ReadOnly[Sequence[str]]
    exclude_domains: ReadOnly[Sequence[str]]
    extra_snippets: ReadOnly[bool]
    max_snippets: ReadOnly[Annotated[int, Field(ge=1, le=10)]]
    min_score: ReadOnly[Annotated[float, Field(ge=0, le=1)]]
    full_content: ReadOnly[Literal["markdown", "html"]]


class _StaanExtraSnippet(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk: str
    score: float


class _StaanFullContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    format: Literal["markdown", "html"]
    length: int


class _StaanWebResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = ""
    url: str = ""
    snippet: str = ""
    published_date: str | None = None
    extra_snippets: Sequence[_StaanExtraSnippet] | None = None
    full_content: _StaanFullContent | None = None


class _StaanWebResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: Sequence[_StaanWebResult] = ()


class _StaanSearchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    web: _StaanWebResponse = Field(default_factory=_StaanWebResponse)


_STAAN_REQUEST_ADAPTER: Final = TypeAdapter(StaanSearchRequest)
_STAAN_RESPONSE_ADAPTER: Final = TypeAdapter(_StaanSearchResponse)
_STAAN_DOMAIN_LIST_ADAPTER: Final = TypeAdapter(
    Annotated[list[Annotated[str, Field(min_length=1)]], Field(max_length=10)]
)
_STAAN_MARKETS: Final = frozenset(("fr-fr", "en-us", "de-de"))
_STAAN_OFFSETS: Final = frozenset((0, 10, 20, 30))
_COUNTRY_TO_MARKET: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fr": "fr-fr",
        "fr-fr": "fr-fr",
        "de": "de-de",
        "de-de": "de-de",
        "us": "en-us",
        "en-us": "en-us",
    }
)
_TRUE_STRINGS: Final = frozenset(("true", "1", "yes"))
_FALSE_STRINGS: Final = frozenset(("false", "0", "no"))
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})
_NORMALIZED_RESPONSE_FIELDS: Final = frozenset(("object", "results"))


def _normalize_domains(*domain_groups: Iterable[str]) -> frozenset[str]:
    return frozenset(domain.lower().removeprefix("www.") for group in domain_groups for domain in group)


def _hostname_matches(hostname: str, domains: Collection[str]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


class StaanSearchConfig(BaseSearchConfig):
    STAAN_API_BASE = "https://api.staan.ai/v2/search/web"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Staan"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "POST"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig hook accepts provider keyword arguments
    ) -> dict:  # mutable-ok: BaseSearchConfig requires mutable request headers
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("STAAN_API_KEY",),
            base_env_var="STAAN_API_BASE",
            default_api_base=self.STAAN_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("STAAN_API_KEY is not set. Set `STAAN_API_KEY` environment variable.")
        request_headers: Final[dict[str, str]] = {  # mutable-ok: HTTP handler hooks require mutable request headers
            **headers,
            "Accept": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }
        return request_headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,  # mutable-ok: BaseSearchConfig hook requires this signature
        data: dict | list[dict] | None = None,  # mutable-ok: BaseSearchConfig hook requires this signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig hook accepts provider keyword arguments
    ) -> str:
        return api_base or get_secret_str("STAAN_API_BASE") or self.STAAN_API_BASE

    def transform_search_request(
        self,
        query: str | Sequence[str],
        optional_params: dict,  # mutable-ok: BaseSearchConfig hook requires this signature
        api_key: str | None = None,
        api_base: str | None = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig hook accepts provider keyword arguments
    ) -> StaanSearchRequest:
        search_params: Final = MappingProxyType(optional_params)
        search_query: Final = " ".join(query) if not isinstance(query, str) else query
        if len(search_query) > 400:
            raise ValueError("query must be at most 400 characters")
        self._validate_max_results(search_params.get("max_results"))

        market: Final = self._resolve_market(search_params)
        offset: Final = search_params.get("offset")
        if offset is not None and offset not in _STAAN_OFFSETS:
            raise ValueError("offset must be one of: 0, 10, 20, 30")
        self._validate_count(search_params.get("count"))

        domain_params: Final = self._build_domain_params(search_params)
        enrichment_params: Final = self._build_enrichment_params(search_params)
        request_fields: Final = (
            ("q", search_query),
            *((("market", market),) if market is not None else ()),
            *((("offset", offset),) if offset is not None else ()),
            *domain_params.items(),
            *enrichment_params.items(),
        )
        request_data: Final[StaanSearchRequest] = MappingProxyType(dict(request_fields))
        try:
            validated_request: Final = _STAAN_REQUEST_ADAPTER.validate_python(request_data)
            return validated_request
        except ValidationError as exc:
            raise ValueError(f"Invalid Staan search parameters: {exc}") from exc

    @staticmethod
    def _resolve_market(optional_params: Mapping[str, object]) -> str | None:
        explicit_market: Final = optional_params.get("market")
        if explicit_market:
            if explicit_market not in _STAAN_MARKETS:
                raise ValueError("market must be one of: fr-fr, en-us, de-de")
            return explicit_market

        country: Final = optional_params.get("country")
        if country is not None:
            country_market: Final = _COUNTRY_TO_MARKET.get(country.lower())
            if country_market is None and explicit_market is None:
                raise ValueError("country must map to a supported Staan market: FR, DE, or US")
            if country_market is not None:
                return country_market

        return None

    @staticmethod
    def _validate_count(count: int | None) -> None:
        if count is not None and count != 10:
            raise ValueError("Staan fixes count at 10; use max_results to limit results")

    @staticmethod
    def _validate_max_results(max_results: object) -> int | None:
        if max_results is None:
            return None
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        return max_results

    @classmethod
    def _build_domain_params(
        cls,
        optional_params: Mapping[str, object],
    ) -> StaanSearchRequest:
        raw_include_domains: Final = optional_params.get("include_domains")
        raw_search_domain_filter: Final = optional_params.get("search_domain_filter")
        raw_exclude_domains: Final = optional_params.get("exclude_domains")
        include_domains: Final = cls._validate_domain_group(raw_include_domains, "include_domains")
        search_domain_filter: Final = cls._validate_domain_group(raw_search_domain_filter, "search_domain_filter")
        exclude_domains: Final = cls._validate_domain_group(raw_exclude_domains, "exclude_domains")
        merged_include_domains: Final = tuple(dict.fromkeys((*include_domains, *search_domain_filter)))
        if merged_include_domains and exclude_domains:
            raise ValueError("include_domains and exclude_domains are mutually exclusive in Staan")
        try:
            domain_entries: Final = (
                (
                    (
                        "include_domains",
                        _STAAN_DOMAIN_LIST_ADAPTER.validate_python(merged_include_domains),
                    ),
                )
                if merged_include_domains
                else ()
            )
            excluded_entry: Final = (
                (
                    (
                        "exclude_domains",
                        _STAAN_DOMAIN_LIST_ADAPTER.validate_python(exclude_domains),
                    ),
                )
                if exclude_domains
                else ()
            )
            domain_params: Final[StaanSearchRequest] = MappingProxyType(dict(domain_entries + excluded_entry))
            return domain_params
        except ValidationError as exc:
            raise ValueError(f"Invalid Staan domain filters: {exc}") from exc

    @staticmethod
    def _validate_domain_group(value: object, name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        try:
            return tuple(_STAAN_DOMAIN_LIST_ADAPTER.validate_python(value))
        except ValidationError as exc:
            raise ValueError(f"Invalid Staan domain filter `{name}`: {exc}") from exc

    @classmethod
    def _build_enrichment_params(
        cls,
        optional_params: Mapping[str, object],
    ) -> StaanSearchRequest:
        extra_snippets: Final = optional_params.get("extra_snippets")
        should_fetch_snippets: Final = cls._parse_bool(
            optional_params.get("max_snippets") is not None if extra_snippets is None else extra_snippets
        )
        max_snippets: Final = optional_params.get("max_snippets")
        min_score: Final = optional_params.get("min_score")
        full_content: Final = optional_params.get("full_content")

        if should_fetch_snippets and max_snippets is not None:
            if isinstance(max_snippets, bool) or not isinstance(max_snippets, int) or not 1 <= max_snippets <= 10:
                raise ValueError("max_snippets must be between 1 and 10")
        if should_fetch_snippets and min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not 0 <= min_score <= 1:
                raise ValueError("min_score must be between 0 and 1")

        enrichment_entries: Final = (("extra_snippets", True),) if should_fetch_snippets else ()
        max_snippet_entry: Final = (
            (("max_snippets", max_snippets),) if should_fetch_snippets and max_snippets is not None else ()
        )
        score_entry: Final = (("min_score", min_score),) if should_fetch_snippets and min_score is not None else ()
        content_entry: Final = (("full_content", full_content),) if full_content is not None else ()
        enrichment_params: Final[StaanSearchRequest] = MappingProxyType(
            dict(enrichment_entries + max_snippet_entry + score_entry + content_entry)
        )
        return enrichment_params

    @staticmethod
    def _parse_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if not isinstance(value, str):
            raise TypeError("extra_snippets must be a boolean")
        normalized: Final = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise ValueError("extra_snippets must be a boolean")

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict | None = None,  # mutable-ok: required by the BaseSearchConfig hook
        **kwargs: object,  # kwargs-ok: required by the BaseSearchConfig hook
    ) -> SearchResponse:
        response: Final = _STAAN_RESPONSE_ADAPTER.validate_python(raw_response.json())
        search_options: Final = MappingProxyType(optional_params or _EMPTY_MAPPING)
        included_domains: Final = _normalize_domains(
            search_options.get("search_domain_filter", ()),
            search_options.get("include_domains", ()),
        )
        excluded_domains: Final = _normalize_domains(search_options.get("exclude_domains", ()))
        results: Final = tuple(
            self._transform_result(result)
            for result in response.web.results
            if self._result_matches_domains(result.url, included_domains, excluded_domains)
        )
        max_results: Final = self._validate_max_results(search_options.get("max_results"))
        response_extra: Final = tuple((response.model_extra or _EMPTY_MAPPING).items())
        web_extra: Final = tuple((response.web.model_extra or _EMPTY_MAPPING).items())
        response_extra_fields: Final[Mapping[str, object]] = MappingProxyType(
            {key: value for key, value in response_extra + web_extra if key not in _NORMALIZED_RESPONSE_FIELDS}
        )
        normalized_response: Final[SearchResponse] = SearchResponse(
            results=tuple(results[:max_results] if max_results is not None else results),
            object="search",
        )
        return normalized_response.model_copy(update=response_extra_fields)

    @staticmethod
    def _result_matches_domains(
        url: str,
        included_domains: Collection[str],
        excluded_domains: Collection[str],
    ) -> bool:
        hostname: Final = (urlsplit(url).hostname or "").lower().removeprefix("www.")
        if included_domains and (not hostname or not _hostname_matches(hostname, included_domains)):
            return False
        return not (excluded_domains and _hostname_matches(hostname, excluded_domains))

    @staticmethod
    def _transform_result(result: _StaanWebResult) -> SearchResult:
        extra_snippets: Final = result.extra_snippets or ()
        snippet: Final = "\n\n".join(
            text
            for text in (
                result.snippet,
                *(extra.chunk for extra in extra_snippets),
            )
            if text
        )
        snippet_fields: Final = (
            (("extra_snippets", tuple(extra.model_dump() for extra in extra_snippets)),)
            if result.extra_snippets is not None
            else ()
        )
        content_fields: Final = (
            (("full_content", result.full_content.model_dump()),) if result.full_content is not None else ()
        )
        additional_fields: Final[Mapping[str, object]] = MappingProxyType(
            dict(tuple((result.model_extra or _EMPTY_MAPPING).items()) + snippet_fields + content_fields)
        )
        standard_result: Final = SearchResult(
            title=result.title,
            url=result.url,
            snippet=snippet,
            date=result.published_date,
            last_updated=None,
        )
        return standard_result.model_copy(update=additional_fields)

"""
Calls Serply's /v1/search endpoint to search Google.

Serply API Reference: https://serply.io/docs
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SERPLY_DOCS_URL: Final = "https://serply.io/docs"
_PARAMS_KEY: Final = "_serply_params"


class _ResultMetadata(BaseModel):
    """The slice of a result's free-form `metadata` object that maps onto SearchResult."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    published_time: str | None = None


class _SerplyResult(BaseModel):
    """One entry of Serply's `results` array. Every field is optional so a single degraded
    result degrades to empty strings instead of failing the whole call."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str | None = None
    link: str | None = None
    description: str | None = None
    # Free-form per Serply's schema (display_url, sitelinks, attributes, ...), so an
    # unexpected shape must not fail the search.
    metadata: object = None


class _SerplySearchResponse(BaseModel):
    """Serply's /v1/search response envelope."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Required: a search with no hits returns `[]`, so a null or absent `results` means the
    # body is not a search response and must not be reported as a successful empty search.
    results: tuple[_SerplyResult, ...]


class _ErrorEnvelope(BaseModel):
    """Serply reports errors as `{"detail": ...}`, e.g. a 401 `{"detail":"Invalid API key"}`."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    detail: str | None = None


_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])
_ParamsAdapter: Final = TypeAdapter(dict[str, object])

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


def _optional(key: str, value: object) -> Mapping[str, object]:
    """A one-entry mapping to spread into a payload, or nothing when the value is absent."""
    return MappingProxyType({key: value}) if value is not None else _NOTHING


class SerplySearchConfig(BaseSearchConfig):
    SERPLY_API_BASE = "https://api.serply.io/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Serply"

    def get_http_method(self) -> Literal["GET", "POST"]:
        """Serply takes its parameters in the query string."""
        return "GET"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        """
        Validate environment and return headers.

        Returns a new dict rather than mutating ``headers``: the http handler calls this
        a second time after ``litellm/search/main.py`` already did, so it has to be idempotent.
        """
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SERPLY_API_KEY",),
            base_env_var="SERPLY_API_BASE",
            default_api_base=self.SERPLY_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("SERPLY_API_KEY is not set. Set `SERPLY_API_KEY` environment variable.")
        return {  # mutable-ok: httpx requires a plain dict of headers
            **headers,
            "X-Api-Key": resolved_api_key,
            "Content-Type": "application/json",
            "User-Agent": "litellm",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        """
        Get complete URL for the Search endpoint, including the query string.

        Serply is a GET API, so the parameters transform_search_request built are encoded
        here; the handler sends this URL as-is and never serializes a body.
        """
        resolved_base: Final = (api_base or get_secret_str("SERPLY_API_BASE") or self.SERPLY_API_BASE).rstrip("/")
        endpoint: Final = resolved_base if resolved_base.endswith("/search") else f"{resolved_base}/search"

        try:
            params: Final = _ParamsAdapter.validate_python(data.get(_PARAMS_KEY) if isinstance(data, dict) else None)
        except ValidationError:
            return endpoint
        return f"{endpoint}?{urlencode(params, doseq=True)}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: get_complete_url reads this back to build the query string
        """
        Transform Search request to Serply API format.

        - query -> q (a list is joined with spaces; Serply takes a single string)
        - max_results -> num, sent unclamped: Serply serves one page of Google results, so a
          larger value returns about ten rather than failing, and clamping would only hide that
        - country -> gl, lower-cased to the form Google expects
        - search_domain_filter -> `site:` clauses appended to q, with `-`-prefixed entries
          becoming `-site:` exclusions
        - max_tokens_per_page -> dropped (no Serply equivalent)

        Everything else is forwarded as-is, so the rest of Serply's surface (`tbs` for time
        ranges, `hl`, `start`, ...) stays reachable without LiteLLM tracking it.

        Returns the parameters under a private key for get_complete_url to encode, matching
        the other GET-based search providers.
        """
        unified_params: Final = self.get_supported_perplexity_optional_params()
        country: Final = optional_params.get("country")

        passthrough: Final = MappingProxyType(
            {param: value for param, value in optional_params.items() if param not in unified_params}
        )
        params: Final = MappingProxyType(
            {
                "q": _apply_domain_filter(
                    " ".join(query) if isinstance(query, list) else query,
                    optional_params.get("search_domain_filter"),
                ),
                **_optional("num", optional_params.get("max_results")),
                **_optional("gl", country.lower() if isinstance(country, str) else None),
                **passthrough,
            }
        )
        return {_PARAMS_KEY: params}  # mutable-ok: BaseSearchConfig.transform_search_request returns a dict

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        Transform Serply API response to LiteLLM unified SearchResponse format.

        Serply -> LiteLLM mappings:
        - results[].title -> SearchResult.title
        - results[].link -> SearchResult.url
        - results[].description -> SearchResult.snippet
        - results[].metadata.published_time -> SearchResult.date

        `published_time` is the date Google displays, so it is absolute ("Aug 11, 2026") for
        older pages and relative ("3 days ago") for ones from the last week. It is passed
        through verbatim either way, matching how the other Google SERP providers here report
        the same field. The whole `metadata` object also rides through as an extra on
        `SearchResult`, so `display_url`, `sitelinks` and `attributes` are not lost.

        Serply ranks results itself via `position`, so the order is preserved as received.
        A body that does not match the documented schema raises an attributed error rather than
        being reported as a successful empty search. Parsing the response bytes rather than
        `.json()` covers the non-JSON case through that same path.
        """
        try:
            parsed: Final = _SerplySearchResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the documented /v1/search schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )

        return SearchResponse(
            results=[  # mutable-ok: SearchResponse.results is declared list[SearchResult]
                SearchResult(
                    title=result.title or "",
                    url=result.link or "",
                    snippet=result.description or "",
                    date=_published_time(result.metadata),
                    last_updated=None,
                    **_optional("metadata", result.metadata),
                )
                for result in parsed.results
            ],
            object="search",
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.get_error_class signature
    ) -> Exception:
        detail: Final = _unwrap_error_detail(error_message).rstrip(". ")
        return BaseLLMException(
            status_code=status_code,
            message=f"Serply Search: {detail}. See {_SERPLY_DOCS_URL} for details.",
            headers=headers,
        )


def _unwrap_error_detail(error_message: str) -> str:
    """
    Surface the human-readable message inside Serply's error envelope.

    Falls back to the raw body for anything else (CDN HTML pages, plain text, other shapes).
    """
    try:
        body: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return body.detail or error_message


def _apply_domain_filter(query: str, search_domain_filter: object) -> str:
    """
    Narrow a query with the unified `search_domain_filter`.

    Follows the Perplexity unified spec, where a `-` prefix means "exclude this domain", and
    expresses both halves in Google's own syntax: alternatives are OR-ed inside a group, and
    exclusions become `-site:`. Anything that is not a list of strings is ignored rather than
    raising, since the filter only ever narrows a search that is otherwise valid.
    """
    try:
        domains: Final = _DomainListAdapter.validate_python(search_domain_filter)
    except ValidationError:
        return query

    include: Final = tuple(d for d in domains if d and not d.startswith("-"))
    exclude: Final = tuple(d[1:] for d in domains if d.startswith("-") and len(d) > 1)
    if not include and not exclude:
        return query

    clauses: Final = (
        f"({query})",
        *(("({})".format(" OR ".join(f"site:{d}" for d in include)),) if include else ()),
        *(f"-site:{d}" for d in exclude),
    )
    return " ".join(clauses)


def _published_time(metadata: object) -> str | None:
    try:
        return _ResultMetadata.model_validate(metadata).published_time
    except ValidationError:
        return None

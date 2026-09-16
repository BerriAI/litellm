"""
Calls Search1API's /search endpoint to search the web through Google, Bing, DuckDuckGo and other engines.

Search1API API Reference: https://s1.dev/docs/basic/search
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

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

_SEARCH1API_DOCS_URL: Final = "https://s1.dev/docs/basic/search"
_UNIFIED_DEFAULT_MAX_RESULTS: Final = 10
_UNSUPPORTED_PARAMS: Final = frozenset(("crawl_results", "image"))


class _Search1APIResult(BaseModel):
    """One entry of Search1API's `results` array. Every field is optional so a single degraded
    result degrades to empty strings instead of failing the whole call."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str | None = None
    link: str | None = None
    snippet: str | None = None


class _Search1APISearchResponse(BaseModel):
    """Search1API's /search response envelope."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    results: tuple[_Search1APIResult, ...]


class _ErrorEnvelope(BaseModel):
    """Search1API reports errors as `{"ok": false, "error": ..., "message": ...}`."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    message: str | None = None
    error: str | None = None


_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


class Search1APISearchConfig(BaseSearchConfig):
    SEARCH1API_API_BASE = "https://api.search1api.com"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Search1API"

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
        ``SEARCH1API_KEY`` is the name Search1API's own CLI, SDKs and MCP server read, so it is
        honored as a fallback to the LiteLLM-style ``SEARCH1API_API_KEY``.
        """
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SEARCH1API_API_KEY", "SEARCH1API_KEY"),
            base_env_var="SEARCH1API_API_BASE",
            default_api_base=self.SEARCH1API_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("SEARCH1API_API_KEY is not set. Set `SEARCH1API_API_KEY` environment variable.")
        return {  # mutable-ok: httpx requires a plain dict of headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = (api_base or get_secret_str("SEARCH1API_API_BASE") or self.SEARCH1API_API_BASE).rstrip(
            "/"
        )
        if resolved_base.endswith("/search"):
            return resolved_base
        return f"{resolved_base}/search"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as the JSON body
        """
        Transform Search request to Search1API format.

        - query -> query (a list is joined with spaces; Search1API takes a single string per search)
        - max_results -> max_results (Search1API's own default is 5, so the unified spec's 10 is sent
          explicitly when the caller gives none; anything else is passed unclamped so Search1API's
          1-50 validation reports the error)
        - search_domain_filter -> include_sites, with `-`-prefixed entries going to exclude_sites
        - country, max_tokens_per_page -> dropped (no Search1API equivalent)
        - crawl_results, image -> rejected when enabled, dropped when disabled: the unified response
          has no field for fetched page text or image URLs, and each fetched page bills an extra
          Search1API credit that LiteLLM cost tracking cannot see

        Everything else (search_service, time_range, language, include_sites, exclude_sites) is forwarded
        as-is, so the rest of Search1API's search surface stays reachable without LiteLLM tracking it.
        An explicitly supplied include_sites or exclude_sites wins over search_domain_filter.
        """
        enabled_unsupported: Final = tuple(sorted(param for param in _UNSUPPORTED_PARAMS if optional_params.get(param)))
        if enabled_unsupported:
            raise ValueError(
                f"Search1API {', '.join(enabled_unsupported)} is not supported through LiteLLM's unified search: "
                f"the response has no field for fetched page text or image URLs. "
                f"Call Search1API's /crawl endpoint directly instead. See {_SEARCH1API_DOCS_URL} for details."
            )
        dropped: Final = self.get_supported_perplexity_optional_params() | _UNSUPPORTED_PARAMS
        passthrough: Final = MappingProxyType(
            {param: value for param, value in optional_params.items() if param not in dropped}
        )

        return {  # mutable-ok: httpx requires a plain dict for the JSON body
            **_site_filters(optional_params.get("search_domain_filter")),
            **passthrough,
            "query": " ".join(query) if isinstance(query, list) else query,
            "max_results": optional_params.get("max_results", _UNIFIED_DEFAULT_MAX_RESULTS),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        Transform Search1API response to LiteLLM unified SearchResponse format.

        Search1API -> LiteLLM mappings:
        - results[].title -> SearchResult.title
        - results[].link -> SearchResult.url
        - results[].snippet -> SearchResult.snippet

        Search1API returns no publication date, so `date` stays None. A non-2xx body is surfaced through
        get_error_class with Search1API's own message, and a 2xx body that does not match the documented
        schema raises an attributed error rather than being reported as a successful empty search. Parsing
        the response bytes rather than `.json()` covers the non-JSON case through that same path.
        """
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )
        try:
            parsed: Final = _Search1APISearchResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the documented /search schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )

        return SearchResponse(
            results=[  # mutable-ok: SearchResponse.results is declared list[SearchResult]
                SearchResult(
                    title=result.title or "",
                    url=result.link or "",
                    snippet=result.snippet or "",
                    date=None,
                    last_updated=None,
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
            message=f"Search1API: {detail}. See {_SEARCH1API_DOCS_URL} for details.",
            headers=headers,
        )


def _unwrap_error_detail(error_message: str) -> str:
    """
    Surface the human-readable message inside Search1API's error envelope.

    Falls back to the raw body for anything else (CDN HTML pages, plain text, other shapes).
    """
    try:
        body: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return body.message or body.error or error_message


def _site_filters(search_domain_filter: object) -> Mapping[str, object]:
    """
    Split the unified `search_domain_filter` into Search1API's include_sites/exclude_sites lists.

    Follows the Perplexity unified spec, where a `-` prefix means "exclude this domain".
    Anything that is not a list of strings is ignored rather than raising, since it only
    ever narrows a search that is otherwise valid.
    """
    try:
        domains: Final = _DomainListAdapter.validate_python(search_domain_filter)
    except ValidationError:
        return _NOTHING
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("include_sites", tuple(d for d in domains if d and not d.startswith("-"))),
                ("exclude_sites", tuple(d[1:] for d in domains if d.startswith("-") and len(d) > 1)),
            )
            if value
        }
    )

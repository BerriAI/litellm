"""
Calls Nimble's /v2/search endpoint to search the web.

Nimble API Reference: https://docs.nimbleway.com/api-reference/search/search
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

_NIMBLE_DOCS_URL: Final = "https://docs.nimbleway.com/api-reference/search/search"


class _NimbleResult(BaseModel):
    """One entry of Nimble's `results` array. Every field is optional so a single degraded
    result degrades to empty strings instead of failing the whole call."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str | None = None
    url: str | None = None
    content: str | None = None
    description: str | None = None
    # Free-form per Nimble's schema, so an unexpected shape must not fail the search.
    additional_data: object = None


class _NimbleSearchResponse(BaseModel):
    """Nimble's /v2/search response envelope."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Required: a search with no hits returns `[]`, so a null or absent `results` means the
    # body is not a search response and must not be reported as a successful empty search.
    results: tuple[_NimbleResult, ...]


class _AdditionalData(BaseModel):
    """The slice of a result's free-form `additional_data` that maps onto SearchResult."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    publish_date: str | None = None


class _ErrorEnvelope(BaseModel):
    """Nimble reports errors as either `{"detail": ...}` (validation) or
    `{"success": "false", "task_id": ..., "message": ...}` (collection)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    detail: str | None = None
    message: str | None = None


_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


def _optional(key: str, value: object) -> Mapping[str, object]:
    """A one-entry mapping to spread into a payload, or nothing when the value is absent."""
    return MappingProxyType({key: value}) if value is not None else _NOTHING


class NimbleSearchConfig(BaseSearchConfig):
    NIMBLE_API_BASE = "https://sdk.nimbleway.com/v2"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Nimble"

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
            key_env_vars=("NIMBLE_API_KEY",),
            base_env_var="NIMBLE_API_BASE",
            default_api_base=self.NIMBLE_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("NIMBLE_API_KEY is not set. Set `NIMBLE_API_KEY` environment variable.")
        return {  # mutable-ok: httpx requires a plain dict of headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
            # Nimble's client-attribution header: names the calling software, nothing else.
            "X-Client-Source": "litellm",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = (api_base or get_secret_str("NIMBLE_API_BASE") or self.NIMBLE_API_BASE).rstrip("/")
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
        Transform Search request to Nimble API format.

        Nimble already uses the Perplexity unified spec's names, so this is close to a pass-through:
        - query -> query (a list is joined with spaces; Nimble takes a single string)
        - max_results -> max_results (sent unclamped so Nimble's own 1-100 validation reports the error)
        - country -> country, upper-cased to the ISO form Nimble documents
        - search_domain_filter -> include_domains, with `-`-prefixed entries going to exclude_domains
        - max_tokens_per_page -> dropped (no Nimble equivalent)

        Everything else is forwarded as-is, so the rest of Nimble's surface stays reachable
        without LiteLLM tracking it.
        """
        unified_params: Final = self.get_supported_perplexity_optional_params()
        country: Final = optional_params.get("country")

        # Spread after the derived domain filters so an explicitly supplied `include_domains`
        # or `exclude_domains` wins over anything read out of `search_domain_filter`.
        passthrough: Final = MappingProxyType(
            {param: value for param, value in optional_params.items() if param not in unified_params}
        )

        return {  # mutable-ok: httpx requires a plain dict for the JSON body
            **_domain_filters(optional_params.get("search_domain_filter")),
            **passthrough,
            "query": " ".join(query) if isinstance(query, list) else query,
            **_optional("max_results", optional_params.get("max_results")),
            **_optional("country", country.upper() if isinstance(country, str) else None),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        Transform Nimble API response to LiteLLM unified SearchResponse format.

        `date` carries only the absolute `publish_date`. News results often carry a relative
        `publish_date_raw` ("1 day ago") instead, which is not a date, so the whole
        `additional_data` object rides through as an extra on `SearchResult` and nothing is lost.

        Nimble ranks results itself via metadata.position, so the order is preserved as received.
        A body that does not match the documented schema raises an attributed error rather than
        being reported as a successful empty search. Parsing the response bytes rather than
        `.json()` covers the non-JSON case through that same path.
        """
        try:
            parsed: Final = _NimbleSearchResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the documented /v2/search schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )

        return SearchResponse(
            results=[  # mutable-ok: SearchResponse.results is declared list[SearchResult]
                SearchResult(
                    title=result.title or "",
                    url=result.url or "",
                    snippet=result.content or result.description or "",
                    date=_publish_date(result.additional_data),
                    last_updated=None,
                    **_optional("additional_data", result.additional_data),
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
            message=f"Nimble Search: {detail}. See {_NIMBLE_DOCS_URL} for details.",
            headers=headers,
        )


def _unwrap_error_detail(error_message: str) -> str:
    """
    Surface the human-readable message inside Nimble's error envelopes.

    Falls back to the raw body for anything else (CDN HTML pages, plain text, other shapes).
    """
    try:
        body: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return body.detail or body.message or error_message


def _domain_filters(search_domain_filter: object) -> Mapping[str, object]:
    """
    Split the unified `search_domain_filter` into Nimble's include/exclude lists.

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
                ("include_domains", tuple(d for d in domains if d and not d.startswith("-"))),
                ("exclude_domains", tuple(d[1:] for d in domains if d.startswith("-") and len(d) > 1)),
            )
            if value
        }
    )


def _publish_date(additional_data: object) -> str | None:
    try:
        return _AdditionalData.model_validate(additional_data).publish_date
    except ValidationError:
        return None

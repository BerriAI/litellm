"""
Calls Keenable's /v1/search endpoint to search the web.

Keenable works without an API key: keyless requests go to /v1/search/public, which is rate
limited per IP and identifies the calling app by the X-Keenable-Title header. With an API key
the same body goes to /v1/search, which has higher limits.

Keenable API Reference: https://docs.keenable.ai/api-reference/search
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_KEENABLE_DOCS_URL: Final = "https://docs.keenable.ai/api-reference/search"

# Sent on every request. The keyless endpoint rejects a request without it (400), and it
# names the calling software only, never the user.
_APP_TITLE: Final = "litellm"


class _KeenableResult(LiteLLMBaseModel):
    """One entry of Keenable's `results` array. Every field is optional so a single degraded
    result degrades to empty strings instead of failing the whole call."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str | None = None
    url: str | None = None
    # Usually empty. The page text is in `snippet`.
    description: str | None = None
    snippet: str | None = None
    published_at: str | None = None
    acquired_at: str | None = None


class _KeenableSearchResponse(LiteLLMBaseModel):
    """Keenable's /v1/search response envelope."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Required: a search with no hits returns `[]`, so a null or absent `results` means the
    # body is not a search response and must not be reported as a successful empty search.
    results: tuple[_KeenableResult, ...]


class _ErrorEnvelope(LiteLLMBaseModel):
    """Keenable reports errors as `{"error": "Invalid parameter", "message": "..."}`."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    error: str | None = None
    message: str | None = None


_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])

# Set on the logging object's optional_params from the endpoint that answered, and read by the
# search cost calculator: keyless searches are free, keyed ones are billed per query.
KEENABLE_KEYLESS_PARAM: Final = "_keenable_keyless"

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


def _optional(key: str, value: object) -> Mapping[str, object]:
    """A one-entry mapping to spread into a payload, or nothing when the value is absent."""
    return MappingProxyType({key: value}) if value is not None else _NOTHING


class KeenableSearchConfig(BaseSearchConfig):
    KEENABLE_API_BASE = "https://api.keenable.ai/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Keenable"

    def _resolve_api_key(self, api_key: str | None, api_base: str | None) -> str | None:
        """The caller's key, else KEENABLE_API_KEY, else None for a keyless call."""
        return self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("KEENABLE_API_KEY",),
            base_env_var=None,
            default_api_base=self.KEENABLE_API_BASE,
        )

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        """
        Validate environment and return headers. No API key is required.

        Returns a new dict rather than mutating ``headers``: the http handler calls this
        a second time after ``litellm/search/main.py`` already did, so it has to be idempotent.
        """
        resolved_api_key: Final = self._resolve_api_key(api_key, api_base)
        return {
            **headers,
            "Content-Type": "application/json",
            "X-Keenable-Title": _APP_TITLE,
            **_optional("Authorization", f"Bearer {resolved_api_key}" if resolved_api_key else None),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        """
        The keyed and keyless endpoints take the same body and differ only by path, so the key
        decides it: /search with a key, /search/public without. ``api_base`` is the version
        root (https://api.keenable.ai/v1); a base that already ends in either path is accepted.
        """
        caller_api_key: Final = kwargs.get("api_key")
        resolved_api_key: Final = self._resolve_api_key(
            caller_api_key if isinstance(caller_api_key, str) else None, api_base
        )
        root: Final = (
            (api_base or self.KEENABLE_API_BASE).rstrip("/").removesuffix("/search/public").removesuffix("/search")
        )
        return f"{root}/search" if resolved_api_key else f"{root}/search/public"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as the JSON body
        """
        Transform Search request to Keenable API format.

        - query -> query (a list is joined with spaces; Keenable takes a single string)
        - max_results -> max_results (sent unclamped so Keenable's own 1-50 validation reports the error)
        - search_domain_filter -> `site` for a single domain, `site:` / `-site:` clauses in the query
          otherwise, following the Perplexity unified spec where a `-` prefix excludes a domain
        - country, max_tokens_per_page -> dropped (no Keenable equivalent)

        Everything else is forwarded as-is, so the rest of Keenable's surface (date filters,
        snippet_max_length) stays reachable without LiteLLM tracking it.
        """
        unified_params: Final = self.get_supported_perplexity_optional_params()
        include, exclude = _split_domains(optional_params.get("search_domain_filter"))
        site: Final = include[0] if len(include) == 1 else None

        # Spread after the derived `site` so an explicitly supplied one wins.
        # The keyless flag is reserved for pricing and never sent to Keenable.
        passthrough: Final = MappingProxyType(
            {
                param: value
                for param, value in optional_params.items()
                if param not in unified_params and param != KEENABLE_KEYLESS_PARAM
            }
        )

        return {
            **_optional("site", site),
            **passthrough,
            "query": _query_with_domain_clauses(
                " ".join(query) if isinstance(query, list) else query,
                include=() if site else include,
                exclude=exclude,
            ),
            **_optional("max_results", optional_params.get("max_results")),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        Transform Keenable API response to LiteLLM unified SearchResponse format.

        `snippet` carries the page text and `description` is usually empty, so the snippet comes
        first. `date` is the page's `published_at` and `last_updated` is `acquired_at`, when
        Keenable last fetched the page. A body that does not match the documented schema raises
        an attributed error rather than being reported as a successful empty search.

        Records whether the keyless endpoint answered, so the cost calculator prices only keyed
        searches. Written unconditionally, so a caller-supplied value never sets its own cost.
        """
        try:
            parsed: Final = _KeenableSearchResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the documented /v1/search schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        logging_obj.optional_params = {
            **logging_obj.optional_params,
            KEENABLE_KEYLESS_PARAM: _answered_keyless(raw_response),
        }

        return SearchResponse(
            results=[
                SearchResult(
                    title=result.title or "",
                    url=result.url or "",
                    snippet=result.snippet or result.description or "",
                    date=result.published_at,
                    last_updated=result.acquired_at,
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
        # Keyless calls share one per-IP budget, which a proxy serving many users exhausts first.
        hint: Final = (
            " Without an API key, requests share a per-IP limit; set KEENABLE_API_KEY to raise it."
            if status_code == 429
            else ""
        )
        return BaseLLMException(
            status_code=status_code,
            message=f"Keenable Search: {detail}.{hint} See {_KEENABLE_DOCS_URL} for details.",
            headers=headers,
        )


def _answered_keyless(raw_response: httpx.Response) -> bool:
    """True when /search/public answered. An unknown endpoint counts as keyed, so it is billed."""
    try:
        return raw_response.request.url.path.endswith("/search/public")
    except RuntimeError:  # a response built without its request
        return False


def _unwrap_error_detail(error_message: str) -> str:
    """
    Surface the human-readable message inside Keenable's error envelope.

    Falls back to the raw body for anything else (CDN HTML pages, plain text, other shapes).
    """
    try:
        body: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return body.message or body.error or error_message


def _split_domains(search_domain_filter: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Split the unified `search_domain_filter` into included and excluded domains.

    Anything that is not a list of strings is ignored rather than raising, since it only
    ever narrows a search that is otherwise valid.
    """
    try:
        domains: Final = _DomainListAdapter.validate_python(search_domain_filter)
    except ValidationError:
        return (), ()
    return (
        tuple(d for d in domains if d and not d.startswith("-")),
        tuple(d[1:] for d in domains if d.startswith("-") and len(d) > 1),
    )


def _query_with_domain_clauses(query: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> str:
    """Append `(site:a OR site:b)` and `-site:c` clauses, which Keenable honors in the query."""
    included: Final = (f"({' OR '.join(f'site:{d}' for d in include)})",) if include else ()
    excluded: Final = tuple(f"-site:{d}" for d in exclude)
    return " ".join((query, *included, *excluded))

"""
cloro Google Search API.
Endpoint: POST https://api.cloro.dev/v1/monitor/google
Docs: https://cloro.dev/docs/api-reference/endpoint/monitor-google?utm_source=litellm&utm_medium=byok
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(Sequence[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_STR_LIST: Final = TypeAdapter(list[str])

CLORO_DEFAULT_COUNTRY: Final = "US"
_CLORO_GOOGLE_ROUTE: Final = "v1/monitor/google"
# The cost map prices the base call: one sync results page without the AI Overview, 5 credits.
CLORO_BASE_QUERY_CREDITS: Final = 5
# Reserved optional_params key that carries the credits cloro charged (X-Credits-Charged) to cost tracking.
CLORO_CREDITS_CHARGED_PARAM: Final = "_cloro_credits_charged"
_CREDITS_CHARGED_HEADER: Final = "X-Credits-Charged"


class CloroSearchConfig(BaseSearchConfig):
    CLORO_API_BASE = "https://api.cloro.dev"

    @staticmethod
    def ui_friendly_name() -> str:
        return "cloro (Google Search)"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("CLORO_API_KEY",),
            base_env_var="CLORO_API_BASE",
            default_api_base=self.CLORO_API_BASE,
        )
        if not resolved_key:
            raise ValueError("CLORO_API_KEY is not set. Set `CLORO_API_KEY` environment variable.")
        return {**headers, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str("CLORO_API_BASE") or self.CLORO_API_BASE
        return ensure_api_route(resolved_base, _CLORO_GOOGLE_ROUTE)

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler sends this as the JSON body
        """
        Unified params -> cloro body: ``query`` (list joined with spaces, ``search_domain_filter``
        folded in as ``site:`` and ``-site:`` operators), ``country`` upper-cased and sent as ``US`` when absent,
        because cloro requires it. ``max_results`` has no cloro equivalent: one call returns one
        Google results page, and the response is cut to ``max_results``. ``max_tokens_per_page`` is
        dropped. Everything outside the unified spec (``include``, ``device``, ``location``,
        ``pages``, ``hl``) is forwarded as-is.
        """
        joined_query: Final = " ".join(query) if isinstance(query, list) else query
        domains: Final = _domain_filter(optional_params.get("search_domain_filter"))
        country: Final = optional_params.get("country")
        unified: Final = self.get_supported_perplexity_optional_params()
        # The reserved credits key is dropped: a caller-supplied value would set the cost of its own request
        passthrough: Final = {
            k: v for k, v in optional_params.items() if k not in unified and k != CLORO_CREDITS_CHARGED_PARAM
        }
        return {
            **passthrough,
            "query": _with_site_filters(joined_query, domains),
            "country": country.upper() if isinstance(country, str) and country else CLORO_DEFAULT_COUNTRY,
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        ``result.organicResults[]`` -> results, without rows whose ``link`` is empty or a Google
        ``/goto`` redirect, cut to ``max_results``. Every other key of ``result``
        (``peopleAlsoAsk``, ``relatedSearches``, ``ads``, and ``aioverview`` when
        ``include={"aioverview": True}`` was sent) -> ``SearchResponse.provider_metadata``.
        The ``X-Credits-Charged`` header goes to cost tracking, so ``pages`` and the AI Overview are billed.
        """
        # Written also when the header is absent, so a caller-supplied value never prices the request
        logging_obj.optional_params = {  # rebind-ok: cost tracking reads the reported credits from here
            **logging_obj.optional_params,  # pyright: ignore[reportUnknownMemberType]  # Logging types it as a bare dict
            CLORO_CREDITS_CHARGED_PARAM: _credits_charged(raw_response.headers.get(_CREDITS_CHARGED_HEADER)),
        }
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())
        result: Final = _JSON_OBJECT.validate_python(response_json.get("result", {}))
        rows: Final = _JSON_OBJECTS.validate_python(result.get("organicResults", ()))
        results: Final = [_to_search_result(row) for row in rows if _is_direct_link(row.get("link"))]
        limit: Final = _max_results(kwargs.get("optional_params"))
        return SearchResponse.model_validate(
            {
                "results": results[:limit] if limit is not None else results,
                "object": "search",
                "provider_metadata": {key: value for key, value in result.items() if key != "organicResults"},
            }
        )


def cloro_search_cost(price_per_query: float, credits_charged: object) -> float | None:
    """Cost of the credits cloro reported, at the per-credit rate of the per-query price. None when not reported."""
    if isinstance(credits_charged, bool) or not isinstance(credits_charged, int) or credits_charged < 0:
        return None
    return credits_charged * price_per_query / CLORO_BASE_QUERY_CREDITS


def _credits_charged(header: str | None) -> int | None:
    if header is None:
        return None
    try:
        credits: Final = int(header.strip())
    except ValueError:
        return None
    return credits if credits >= 0 else None


def _domain_filter(raw: object) -> tuple[str, ...]:
    try:
        return tuple(_STR_LIST.validate_python(raw))
    except ValidationError:
        return ()


def _with_site_filters(query: str, domains: tuple[str, ...]) -> str:
    """A leading ``-`` excludes a domain, as in Perplexity's ``search_domain_filter``. Included
    domains are joined with ``OR``; each excluded domain is a separate ``-site:`` term."""
    if not domains:
        return query
    included: Final = [d for d in domains if not d.startswith("-")]
    excluded: Final = [d[1:] for d in domains if d.startswith("-") and len(d) > 1]
    clauses: Final = [f"({query})"]
    if included:
        clauses.append("(" + " OR ".join(f"site:{d}" for d in included) + ")")
    clauses.extend(f"-site:{d}" for d in excluded)
    return " ".join(clauses)


def _max_results(optional_params: object) -> int | None:
    try:
        params: Final = _JSON_OBJECT.validate_python(optional_params)
    except ValidationError:
        return None
    raw: Final = params.get("max_results")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return max(1, raw)


def _is_direct_link(link: object) -> bool:
    if not isinstance(link, str) or not link:
        return False
    parts: Final = urlsplit(link)
    host: Final = (parts.hostname or "").lower()
    is_google_host: Final = host == "google.com" or host.endswith(".google.com")
    return not (is_google_host and parts.path.startswith("/goto"))


def _to_search_result(row: Mapping[str, object]) -> SearchResult:
    return SearchResult.model_validate(
        {
            "title": _as_str(row.get("title")),
            "url": _as_str(row.get("link")),
            "snippet": _as_str(row.get("snippet")),
            "date": None,
            "last_updated": None,
        }
    )


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""

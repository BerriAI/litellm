"""
DeepSeek Search API.
Endpoint: POST https://api.deepseek.com/v1/tools/search
Docs: https://api-docs.deepseek.com/guides/search/
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(Sequence[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_STR_LIST: Final = TypeAdapter(list[str])

DEEPSEEK_SEARCH_MAX_COUNT: Final = 50
DEEPSEEK_SEARCH_MAX_QUERY_CHARS: Final = 128
_DEEPSEEK_SEARCH_PATH: Final = "/v1/tools/search"
# DeepSeek chat defaults DEEPSEEK_API_BASE to the /beta prefix; the search route lives at the root.
_CHAT_ONLY_BASE_SUFFIXES: Final = ("/beta", "/v1")


class DeepSeekSearchConfig(BaseSearchConfig):
    DEEPSEEK_API_BASE = "https://api.deepseek.com"

    @staticmethod
    def ui_friendly_name() -> str:
        return "DeepSeek"

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
            key_env_vars=("DEEPSEEK_API_KEY",),
            base_env_var="DEEPSEEK_API_BASE",
            default_api_base=self.DEEPSEEK_API_BASE,
        )
        if not resolved_key:
            raise ValueError("DEEPSEEK_API_KEY is not set. Set `DEEPSEEK_API_KEY` environment variable.")
        return {**headers, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str("DEEPSEEK_API_BASE") or self.DEEPSEEK_API_BASE
        clean_base: Final = resolved_base.rstrip("/")
        if clean_base.endswith(_DEEPSEEK_SEARCH_PATH):
            return clean_base
        return f"{_strip_chat_suffix(clean_base)}{_DEEPSEEK_SEARCH_PATH}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler sends this as the JSON body
        """
        Unified params -> DeepSeek body: ``query`` (list joined with spaces, ``search_domain_filter``
        folded in as ``site:`` operators, the whole string capped at 128 characters by DeepSeek),
        ``max_results`` -> ``count`` clamped to [1, 50].
        ``country`` and ``max_tokens_per_page`` have no DeepSeek equivalent and are dropped.
        Everything outside the unified spec (``need_content``, ``ori_question``, ``localization``)
        is forwarded as-is.
        """
        joined_query: Final = " ".join(query) if isinstance(query, list) else query
        domains: Final = _domain_filter(optional_params.get("search_domain_filter"))
        count: Final = _count(optional_params.get("max_results"))
        unified: Final = self.get_supported_perplexity_optional_params()
        passthrough: Final = {k: v for k, v in optional_params.items() if k not in unified}
        return {
            **passthrough,
            "query": _checked_query(_with_site_filters(joined_query, domains), domains),
            **({"count": count} if count is not None else {}),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())
        raw_results: Final = _JSON_OBJECTS.validate_python(response_json.get("results", ()))
        return SearchResponse(
            results=[_to_search_result(item) for item in raw_results],
            object="search",
        )


def _strip_chat_suffix(base: str) -> str:
    return next(
        (base.removesuffix(suffix) for suffix in _CHAT_ONLY_BASE_SUFFIXES if base.endswith(suffix)),
        base,
    )


def _domain_filter(raw: object) -> tuple[str, ...]:
    try:
        return tuple(_STR_LIST.validate_python(raw))
    except ValidationError:
        return ()


def _with_site_filters(query: str, domains: tuple[str, ...]) -> str:
    if not domains:
        return query
    site_clause: Final = " OR ".join(f"site:{domain}" for domain in domains)
    return f"{query} {site_clause}"


def _checked_query(query: str, domains: tuple[str, ...]) -> str:
    if len(query) <= DEEPSEEK_SEARCH_MAX_QUERY_CHARS:
        return query
    filter_note: Final = (
        f" ({len(domains)} search_domain_filter entries are sent as site: operators inside the query)"
        if domains
        else ""
    )
    raise BadRequestError(
        message=(
            f"DeepSeek Search accepts a query of at most {DEEPSEEK_SEARCH_MAX_QUERY_CHARS} characters, "
            f"got {len(query)}{filter_note}. Shorten the query or the domain filter."
        ),
        model="deepseek/search",
        llm_provider="deepseek",
    )


def _count(raw: object) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return max(1, min(raw, DEEPSEEK_SEARCH_MAX_COUNT))


def _to_search_result(item: Mapping[str, object]) -> SearchResult:
    date: Final = item.get("date")
    return SearchResult.model_validate(
        {
            **item,
            "title": _as_str(item.get("title")),
            "url": _as_str(item.get("url")),
            "snippet": _as_str(item.get("snippet")),
            "date": date if isinstance(date, str) else None,
            "last_updated": None,
        }
    )


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""

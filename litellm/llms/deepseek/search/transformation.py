"""
DeepSeek Search and Fetch APIs.
Search: POST https://api.deepseek.com/v1/tools/search
Fetch: POST https://api.deepseek.com/v1/tools/fetch
Docs: https://api-docs.deepseek.com/guides/search/
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import urlsplit

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
DEEPSEEK_FETCH_MAX_URL_CHARS: Final = 8192
_DEEPSEEK_SEARCH_PATH: Final = "/v1/tools/search"
_DEEPSEEK_FETCH_PATH: Final = "/v1/tools/fetch"
FETCH_URL_PARAM: Final = "fetch_url"
_FETCH_SCHEMES: Final = frozenset({"http", "https"})
# DeepSeek chat defaults DEEPSEEK_API_BASE to the /beta prefix; the tool routes live at the root.
_CHAT_ONLY_BASE_SUFFIXES: Final = ("/beta", "/v1")
_TOOL_PATHS: Final = (_DEEPSEEK_SEARCH_PATH, _DEEPSEEK_FETCH_PATH)


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
        path: Final = _DEEPSEEK_FETCH_PATH if _fetch_url(optional_params) is not None else _DEEPSEEK_SEARCH_PATH
        if clean_base.endswith(path):
            return clean_base
        without_tool: Final = _without_tool_path(clean_base)
        return f"{_strip_chat_suffix(without_tool)}{path}"

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
        fetch_url: Final = _fetch_url(optional_params)
        if fetch_url is not None:
            return {"url": _checked_fetch_url(fetch_url)}
        joined_query: Final = " ".join(query) if isinstance(query, list) else query
        domains: Final = _domain_filter(optional_params.get("search_domain_filter"))
        count: Final = _count(optional_params.get("max_results"))
        unified: Final = self.get_supported_perplexity_optional_params()
        passthrough: Final = {k: v for k, v in optional_params.items() if k not in unified and k != FETCH_URL_PARAM}
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
        page_url: Final = _fetch_url(_optional_mapping(kwargs.get("optional_params")))
        if page_url is not None or ("results" not in response_json and "content" in response_json):
            return _fetch_page_response(response_json, page_url or "")
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


def _without_tool_path(base: str) -> str:
    stripped: Final = next((base.removesuffix(path) for path in _TOOL_PATHS if base.endswith(path)), base)
    return stripped.rstrip("/")


def _optional_mapping(raw: object) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_python(raw)
    except ValidationError:
        return {}


def _fetch_url(optional_params: Mapping[str, object]) -> str | None:
    raw: Final = optional_params.get(FETCH_URL_PARAM)
    return raw if isinstance(raw, str) else None


def _checked_fetch_url(url: str) -> str:
    parsed: Final = urlsplit(url)
    if (
        not url
        or len(url) > DEEPSEEK_FETCH_MAX_URL_CHARS
        or parsed.scheme not in _FETCH_SCHEMES
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise BadRequestError(
            message=(
                "DeepSeek Fetch needs an http or https URL with a host, no username or password, "
                f"and at most {DEEPSEEK_FETCH_MAX_URL_CHARS} characters."
            ),
            model="deepseek/fetch",
            llm_provider="deepseek",
        )
    return url


def _fetch_page_response(payload: Mapping[str, object], url: str) -> SearchResponse:
    title: Final = payload.get("title")
    content: Final = payload.get("content")
    title_text: Final = title if isinstance(title, str) else ""
    content_text: Final = content if isinstance(content, str) else ""
    return SearchResponse(
        results=[
            SearchResult.model_validate(
                {"title": title_text, "url": url, "snippet": content_text, "content": content_text}
            )
        ],
        object="fetch",
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

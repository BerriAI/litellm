"""
SpecStream Search API.
Endpoint: GET https://api.specstream.ai/v1/web-search?q=<the question>
Docs: https://specstream.ai/search-api
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))
_PARAMS_KEY: Final = "_specstream_params"


class SpecStreamSearchConfig(BaseSearchConfig):
    SPECSTREAM_API_BASE = "https://api.specstream.ai/v1/web-search"

    @staticmethod
    def ui_friendly_name() -> str:
        return "SpecStream"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

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
            key_env_vars=("SPECSTREAM_API_KEY",),
            base_env_var="SPECSTREAM_API_BASE",
            default_api_base=self.SPECSTREAM_API_BASE,
        )
        if not resolved_key:
            raise ValueError("SPECSTREAM_API_KEY is not set. Set `SPECSTREAM_API_KEY` environment variable.")
        return {**headers, "Authorization": f"Bearer {resolved_key}", "Accept": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str("SPECSTREAM_API_BASE") or self.SPECSTREAM_API_BASE
        if isinstance(data, dict) and isinstance(data.get(_PARAMS_KEY), dict):
            return f"{resolved_base}?{urlencode(data[_PARAMS_KEY])}"
        return resolved_base

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: get_complete_url reads this to build the GET query string
        """
        Unified params -> SpecStream query string: ``query`` (a list is joined with spaces) becomes ``q``,
        written in everyday words, 1 to 2,000 characters.
        ``max_results`` trims the returned results; ``search_domain_filter``, ``country`` and
        ``max_tokens_per_page`` have no SpecStream equivalent and are dropped.
        """
        joined_query: Final = " ".join(query) if isinstance(query, list) else query
        return {_PARAMS_KEY: {"q": joined_query}}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        """
        Each source page the answer was read from becomes one result (title, url, snippet), in the
        order SpecStream ranks them. ``site`` is kept as an extra field.
        """
        body: Final = _JSON_OBJECT.validate_python(raw_response.json())
        data: Final = _JSON_OBJECT.validate_python(body.get("data") or {})
        results: Final = [
            _to_search_result(item)
            for item in _JSON_VALUES.validate_python(data.get("sources") or ())
            if isinstance(item, Mapping)
        ]
        optional_params: Final = kwargs.get("optional_params")
        limit: Final = optional_params.get("max_results") if isinstance(optional_params, dict) else None
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return SearchResponse(results=results[:limit], object="search")
        return SearchResponse(results=results, object="search")


def _to_search_result(item: Mapping[str, object]) -> SearchResult:
    site: Final = item.get("site")
    return SearchResult.model_validate(
        {
            "title": _as_str(item.get("title")),
            "url": _as_str(item.get("url")),
            "snippet": _as_str(item.get("snippet")),
            "date": None,
            "last_updated": None,
            **({"site": site} if isinstance(site, str) else {}),
        }
    )


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""

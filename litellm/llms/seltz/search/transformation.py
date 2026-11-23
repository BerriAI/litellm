"""
Seltz Search API.

Endpoint: POST https://api.seltz.ai/v1/search
Docs: https://docs.seltz.ai/api-reference/search?utm_source=litellm&utm_medium=integration&utm_campaign=marketplace
OpenAPI: https://docs.seltz.ai/api-reference/openapi.json?utm_source=litellm&utm_medium=integration&utm_campaign=marketplace
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse, SearchResult
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


_SEARCH_ROUTE: Final = "/v1/search"
_BODY_API_KEY: Final = "api_key"
_DOCUMENT_FIELDS: Final = frozenset({"url", "content", "published_date", "snippets", "properties"})
_RESPONSE_FIELDS: Final = frozenset({"documents", "request_id", "results", "object"})


class _SeltzSnippet(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    text: str | None = None


class _SeltzDocument(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    url: str | None = None
    content: str | None = None
    published_date: str | None = None
    snippets: tuple[_SeltzSnippet, ...] = ()
    properties: dict[str, JsonValue] | None = None


class _SeltzSearchResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    documents: tuple[_SeltzDocument, ...] = ()
    request_id: str | None = None


def _extra_fields(model: BaseModel, reserved: frozenset[str]) -> MappingProxyType[str, object]:
    extra: Final[Mapping[str, object]] = model.model_extra or {}
    return MappingProxyType({key: value for key, value in extra.items() if key not in reserved})


def _complete_search_url(api_base: str) -> str:
    base: Final = api_base.rstrip("/")
    if base.endswith(_SEARCH_ROUTE):
        return base
    return f"{base}{_SEARCH_ROUTE}"


def _mapped_fields(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    max_results: Final = (("max_results", params["max_results"]),) if "max_results" in params else ()
    domains: Final = (("include_domains", params["search_domain_filter"]),) if "search_domain_filter" in params else ()
    return max_results + domains


def _snippet(document: _SeltzDocument) -> str:
    if document.content:
        return document.content
    passages: Final = tuple(snippet.text for snippet in document.snippets if snippet.text)
    return "\n".join(passages)


def _search_result(document: _SeltzDocument) -> SearchResult:
    snippets: Final = [snippet.model_dump() for snippet in document.snippets]
    content: Final = {"content": document.content} if isinstance(document.content, str) else {}
    properties: Final = {"properties": document.properties} if document.properties is not None else {}
    passages: Final = {"snippets": snippets} if snippets else {}
    return SearchResult.model_validate(
        {
            **_extra_fields(document, _DOCUMENT_FIELDS),
            **content,
            **properties,
            **passages,
            "title": "",
            "url": document.url or "",
            "snippet": _snippet(document),
            "date": document.published_date,
            "last_updated": None,
        }
    )


class SeltzSearchConfig(BaseSearchConfig):
    SELTZ_API_BASE: Final = "https://api.seltz.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Seltz"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SELTZ_API_KEY",),
            base_env_var="SELTZ_API_BASE",
            default_api_base=self.SELTZ_API_BASE,
        )
        if not resolved_key:
            raise ValueError("SELTZ_API_KEY is not set. Set `SELTZ_API_KEY` environment variable.")
        return {**headers, "x-api-key": resolved_key, "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> str:
        resolved_base: Final = api_base or get_secret_str("SELTZ_API_BASE") or self.SELTZ_API_BASE
        return _complete_search_url(resolved_base)

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig interface
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, object]:  # mutable-ok: the http handler sends this as the JSON body
        """
        Unified params -> Seltz body: ``query`` (a list is joined with spaces), ``max_results`` kept,
        ``search_domain_filter`` -> ``include_domains``. ``country`` and ``max_tokens_per_page`` have no
        Seltz field and are dropped. Native params (``scope``, ``tier``, ``exclude_domains``, ``from_date``,
        ``to_date``, ``fields``, ``include_domains``) pass through and override a mapped key of the same name.
        ``tier`` is forwarded as written. Seltz matches it without case or surrounding whitespace.
        """
        query_text: Final = " ".join(query) if isinstance(query, list) else query
        blocked: Final = self.get_supported_perplexity_optional_params() | {_BODY_API_KEY}
        native: Final = {key: value for key, value in optional_params.items() if key not in blocked}
        mapped: Final = {key: value for key, value in _mapped_fields(optional_params) if key not in native}
        return {**mapped, **native, "query": query_text}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> SearchResponse:
        try:
            parsed: Final = _SeltzSearchResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise BaseLLMException(
                status_code=502,
                message=f"Seltz returned an invalid search response: {exc}",
                headers=dict(raw_response.headers),
            ) from exc
        request_id: Final = {"request_id": parsed.request_id} if parsed.request_id is not None else {}
        return SearchResponse.model_validate(
            {
                **_extra_fields(parsed, _RESPONSE_FIELDS),
                **request_id,
                "object": "search",
                "results": [_search_result(document) for document in parsed.documents],
            }
        )

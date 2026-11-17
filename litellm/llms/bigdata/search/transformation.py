"""
Bigdata.com document search: https://docs.bigdata.com/api-reference/search/search-documents
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse, SearchResult
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_TOP_LEVEL_FIELDS: Final = frozenset(("search_mode", "include_audit", "response_detail"))


class _BigdataChunk(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    text: str


class _BigdataDocument(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    headline: str
    url: str | None = None
    timestamp: str | None = None
    chunks: tuple[_BigdataChunk, ...]


class _BigdataResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    results: tuple[_BigdataDocument, ...]
    external_results: Mapping[str, tuple[_BigdataDocument, ...]]


def _extra_fields(model: BaseModel, reserved: Mapping[str, object]) -> MappingProxyType[str, object]:
    extra: Final[Mapping[str, object]] = model.model_extra or {}
    return MappingProxyType({key: value for key, value in extra.items() if key not in reserved})


def _search_result(document: _BigdataDocument) -> SearchResult:
    return SearchResult.model_validate(
        {
            **_extra_fields(document, SearchResult.model_fields),
            "title": document.headline,
            "url": document.url or "",
            "snippet": "\n\n".join(chunk.text for chunk in document.chunks),
            "date": document.timestamp,
            "chunks": [chunk.model_dump() for chunk in document.chunks],
        }
    )


class BigdataSearchConfig(BaseSearchConfig):
    DEFAULT_API_BASE: Final = "https://api.bigdata.com/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Bigdata.com"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, str]:  # mutable-ok: HTTP handler requires a dict
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("BIGDATA_API_KEY",),
            base_env_var="BIGDATA_API_BASE",
            default_api_base=self.DEFAULT_API_BASE,
        )
        if not resolved_key:
            raise ValueError("Set BIGDATA_API_KEY or pass api_key for Bigdata search")
        return {
            **headers,
            "X-API-KEY": resolved_key,
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> str:
        return ensure_api_route(api_base or get_secret_str("BIGDATA_API_BASE") or self.DEFAULT_API_BASE, "search")

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig interface
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, object]:  # mutable-ok: HTTP handler requires a JSON dict
        if optional_params.get("search_domain_filter"):
            raise BadRequestError(
                message="Bigdata search cannot filter by domain. Drop search_domain_filter and narrow results with "
                "Bigdata `filters` such as `source` instead.",
                model="bigdata/search",
                llm_provider="bigdata",
            )
        native: Final = MappingProxyType(
            {
                key: value
                for key, value in optional_params.items()
                if key not in self.get_supported_perplexity_optional_params()
            }
        )
        max_results: Final = optional_params.get("max_results")
        return {
            "search_mode": "fast",
            **MappingProxyType({key: value for key, value in native.items() if key in _TOP_LEVEL_FIELDS}),
            "query": {
                **MappingProxyType({} if max_results is None else {"max_chunks": max_results}),
                **MappingProxyType({key: value for key, value in native.items() if key not in _TOP_LEVEL_FIELDS}),
                "text": " ".join(query) if isinstance(query, list) else query,
            },
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> SearchResponse:
        try:
            parsed: Final = _BigdataResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise BaseLLMException(
                status_code=502,
                message=f"Bigdata returned an invalid search response: {exc}",
                headers=dict(raw_response.headers),
            ) from exc
        documents: Final = chain(parsed.results, chain.from_iterable(parsed.external_results.values()))
        return SearchResponse.model_validate(
            {
                "results": [_search_result(document) for document in documents],
                "provider_metadata": dict(_extra_fields(parsed, MappingProxyType({}))),
            }
        )

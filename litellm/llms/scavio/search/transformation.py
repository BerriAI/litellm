"""
Calls Scavio's /api/v2/google endpoint to search Google.

Scavio API Reference: https://scavio.dev/docs/search-api
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, StrictInt, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SCAVIO_DOCS_URL: Final = "https://scavio.dev/docs/search-api"
_SEARCH_PATH: Final = "/api/v2/google"


class _ScavioOrganicResult(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str | None = None
    link: str | None = None
    snippet: str | None = None


class _ScavioSearchResponse(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    organic_results: tuple[_ScavioOrganicResult, ...]


class _ErrorEnvelope(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error: str | None = None


class _UnifiedParams(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    max_results: StrictInt | None = None


_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


def _optional(key: str, value: object) -> Mapping[str, object]:
    return MappingProxyType({key: value}) if value is not None else _NOTHING


class ScavioSearchConfig(BaseSearchConfig):
    SCAVIO_API_BASE = "https://api.scavio.dev"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Scavio"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        resolved_api_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("SCAVIO_API_KEY",),
            base_env_var=None,
            default_api_base=self.SCAVIO_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("SCAVIO_API_KEY is not set. Set `SCAVIO_API_KEY` environment variable.")
        return {
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
        resolved_base: Final = (api_base or self.SCAVIO_API_BASE).rstrip("/")
        if resolved_base.endswith(_SEARCH_PATH):
            return resolved_base
        return f"{resolved_base}{_SEARCH_PATH}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as the JSON body
        """
        Scavio returns one Google page per call and has no result-count param, so `max_results` is
        applied to the parsed response instead. `resolve_ai_overview` defaults to false because only
        organic results are mapped and resolving a deferred AI Overview costs an extra upstream round trip
        """
        unified_params: Final = self.get_supported_perplexity_optional_params()
        country: Final = optional_params.get("country")
        joined_query: Final = " ".join(query) if isinstance(query, list) else query

        passthrough: Final = MappingProxyType(
            {param: value for param, value in optional_params.items() if param not in unified_params}
        )

        return {
            "resolve_ai_overview": False,
            **_optional("gl", country.lower() if isinstance(country, str) else None),
            **passthrough,
            "query": _with_domain_filter(joined_query, optional_params.get("search_domain_filter")),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        try:
            parsed: Final = _ScavioSearchResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the documented /api/v2/google schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        max_results: Final = _requested_max_results(kwargs.get("optional_params"))
        organic: Final = parsed.organic_results[:max_results] if max_results is not None else parsed.organic_results

        return SearchResponse(
            results=[
                SearchResult(
                    title=result.title or "",
                    url=result.link or "",
                    snippet=result.snippet or "",
                    date=None,
                    last_updated=None,
                )
                for result in organic
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
            message=f"Scavio Search: {detail}. See {_SCAVIO_DOCS_URL} for details.",
            headers=headers,
        )


def _unwrap_error_detail(error_message: str) -> str:
    try:
        body: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return body.error or error_message


def _with_domain_filter(query: str, search_domain_filter: object) -> str:
    """
    Google ignores the `site:` filter on `(query) (site:a)` and on an ungrouped `site:a OR site:b`,
    so the query stays bare, one include is `site:a`, and two or more are grouped as `(site:a OR site:b)`
    """
    try:
        domains: Final = _DomainListAdapter.validate_python(search_domain_filter)
    except ValidationError:
        return query
    included: Final = tuple(d for d in domains if d and not d.startswith("-"))
    excluded: Final = tuple(d[1:] for d in domains if d.startswith("-") and len(d) > 1)
    include_sites: Final = tuple(f"site:{d}" for d in included)
    include_clause: Final = f"({' OR '.join(include_sites)})" if len(include_sites) > 1 else "".join(include_sites)
    operators: Final = tuple(clause for clause in (include_clause, *(f"-site:{d}" for d in excluded)) if clause)
    return " ".join((query, *operators))


def _requested_max_results(optional_params: object) -> int | None:
    try:
        max_results: Final = _UnifiedParams.model_validate(optional_params).max_results
    except ValidationError:
        return None
    return max_results if max_results is not None and max_results > 0 else None

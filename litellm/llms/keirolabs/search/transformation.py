"""
Calls Keirolabs' lite search endpoint.

Keirolabs API Reference: https://platform.keirolabs.cloud/docs/
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse, SearchResult
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _KeirolabsResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str = ""
    url: str
    snippet: str | None = None


class _KeirolabsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    results: tuple[_KeirolabsResult, ...]


_DOMAINS: Final = TypeAdapter(tuple[str, ...])


def _query_with_domains(query: str, domains: object) -> str:
    parsed: Final = _DOMAINS.validate_python(domains) if domains is not None else ()
    clauses: Final = tuple(f"site:{domain}" for domain in parsed if domain)
    return f"({query}) ({' OR '.join(clauses)})" if clauses else query


class KeirolabsSearchConfig(BaseSearchConfig):
    DEFAULT_API_BASE: Final = "https://api.keirolabs.cloud"
    SEARCH_ROUTE: Final = "/api/v2/search/lite"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Keirolabs"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, str]:  # mutable-ok: HTTP handler requires a dict
        effective_base: Final = api_base or get_secret_str("KEIROLABS_API_BASE") or self.DEFAULT_API_BASE
        if urlsplit(effective_base).scheme != "https":
            raise ValueError("Keirolabs search requires an HTTPS api_base, so the API key is never sent in plaintext")
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("KEIROLABS_API_KEY",),
            base_env_var="KEIROLABS_API_BASE",
            default_api_base=self.DEFAULT_API_BASE,
        )
        if not resolved_key:
            raise ValueError("KEIROLABS_API_KEY is not set. Set `KEIROLABS_API_KEY` environment variable.")
        return {
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> str:
        base: Final = (api_base or get_secret_str("KEIROLABS_API_BASE") or self.DEFAULT_API_BASE).rstrip("/")
        return base if base.endswith(self.SEARCH_ROUTE) else f"{base}{self.SEARCH_ROUTE}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig interface
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, object]:  # mutable-ok: HTTP handler requires a JSON dict
        max_results: Final = optional_params.get("max_results")
        native: Final = MappingProxyType(
            {
                key: value
                for key, value in optional_params.items()
                if key not in self.get_supported_perplexity_optional_params()
            }
        )
        return {
            **({"maxResults": max_results} if max_results is not None else {}),
            **native,
            "query": _query_with_domains(
                " ".join(query) if isinstance(query, list) else query,
                optional_params.get("search_domain_filter"),
            ),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> SearchResponse:
        try:
            parsed: Final = _KeirolabsResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise BaseLLMException(
                status_code=502,
                message=f"Keirolabs returned an invalid search response: {exc}",
                headers=dict(raw_response.headers),
            ) from exc
        # The lite endpoint returns no publication date, so date and last_updated stay None.
        return SearchResponse(
            object="search",
            results=[
                SearchResult(title=result.title, url=result.url, snippet=result.snippet or "")
                for result in parsed.results
            ],
        )

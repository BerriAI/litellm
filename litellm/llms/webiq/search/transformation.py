from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse, SearchResult

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _WebIQResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    title: str
    url: str
    content: str
    lastUpdatedAt: str | None = None


class _WebIQResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    webResults: tuple[_WebIQResult, ...]


_DOMAINS: Final = TypeAdapter(tuple[str, ...])


def _query_with_domains(query: str, domains: object) -> str:
    if domains is None:
        return query
    parsed: Final = _DOMAINS.validate_python(domains)
    included: Final = tuple(f"site:{domain}" for domain in parsed if domain and not domain.startswith("-"))
    excluded: Final = tuple(f"-site:{domain[1:]}" for domain in parsed if domain.startswith("-") and len(domain) > 1)
    include_clause: Final = f"({' OR '.join(included)})" if included else ""
    if not included and not excluded:
        return query
    return " ".join(part for part in (f"({query})", include_clause, *excluded) if part)


class WebIQSearchConfig(BaseSearchConfig):
    DEFAULT_API_BASE: Final = "https://api.microsoft.ai/v3"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Microsoft Web IQ"

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
            key_env_vars=("WEBIQ_API_KEY",),
            base_env_var=None,
            default_api_base=self.DEFAULT_API_BASE,
        )
        if not resolved_key:
            raise ValueError("Set WEBIQ_API_KEY or pass api_key for Microsoft Web IQ search")
        return {  # mutable-ok: HTTP handler requires a plain headers dict
            **headers,
            "x-apikey": resolved_key,
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> str:
        base: Final = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        return base if base.endswith("/search/web") else f"{base}/search/web"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig interface
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, object]:  # mutable-ok: HTTP handler requires a JSON dict
        country: Final = optional_params.get("country")
        mapped: Final = MappingProxyType(
            {
                "maxResults": optional_params.get("max_results"),
                "region": country.upper() if isinstance(country, str) else None,
            }
        )
        native: Final = MappingProxyType(
            {
                key: value
                for key, value in optional_params.items()
                if key not in self.get_supported_perplexity_optional_params()
            }
        )
        return {  # mutable-ok: HTTP handler serializes a dict as JSON
            "contentFormat": "passage",
            "maxLength": 5000,
            **MappingProxyType({key: value for key, value in mapped.items() if value is not None}),
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
            parsed: Final = _WebIQResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise self.get_error_class(
                error_message=f"Microsoft Web IQ returned an invalid search response: {exc}",
                status_code=502,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class interface
            ) from exc
        return SearchResponse(
            results=[  # mutable-ok: SearchResponse requires list[SearchResult]
                SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.content,
                    date=result.lastUpdatedAt or None,
                    last_updated=result.lastUpdatedAt or None,
                    **MappingProxyType(
                        {
                            key: value
                            for key, value in (result.model_extra or MappingProxyType({})).items()
                            if key not in SearchResult.model_fields
                        }
                    ),
                )
                for result in parsed.webResults
            ],
            **MappingProxyType(
                {
                    key: value
                    for key, value in (parsed.model_extra or MappingProxyType({})).items()
                    if key not in SearchResponse.model_fields
                }
            ),
        )

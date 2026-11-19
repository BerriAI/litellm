from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse, SearchResult
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _ClawifiResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    url: str
    title: str | None = None
    snippet: str | None = None
    published_at: str | None = None
    modified_at: str | None = None


class _ClawifiError(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    message: str


class _ClawifiResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    success: bool
    results: tuple[_ClawifiResult, ...]
    errors: tuple[_ClawifiError, ...] = ()


_DOMAINS: Final = TypeAdapter(tuple[str, ...])


def _domain_filters(domains: object) -> Mapping[str, tuple[str, ...]]:
    parsed: Final = _DOMAINS.validate_python(() if domains is None else domains)
    included: Final = tuple(domain for domain in parsed if domain and not domain.startswith("-"))
    excluded: Final = tuple(domain[1:] for domain in parsed if domain.startswith("-") and len(domain) > 1)
    return MappingProxyType(
        {key: value for key, value in (("include_domains", included), ("exclude_domains", excluded)) if value}
    )


def _extra_fields(model: BaseModel) -> Mapping[str, object]:
    extra: Final[Mapping[str, object]] = model.model_extra or {}
    return extra


class ClawifiSearchConfig(BaseSearchConfig):
    DEFAULT_API_BASE: Final = "https://api.clawifi.dev"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Clawifi"

    def _api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("CLAWIFI_API_BASE") or self.DEFAULT_API_BASE

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig interface
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, str]:  # mutable-ok: HTTP handler requires a dict
        if urlsplit(self._api_base(api_base)).scheme != "https":
            raise ValueError("Clawifi search requires an HTTPS api_base")
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("CLAWIFI_API_KEY",),
            base_env_var="CLAWIFI_API_BASE",
            default_api_base=self.DEFAULT_API_BASE,
        )
        if not resolved_key:
            raise ValueError("Set CLAWIFI_API_KEY or pass api_key for Clawifi search")
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
        return ensure_api_route(self._api_base(api_base), "search/free")

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig interface
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig interface
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> dict[str, object]:  # mutable-ok: HTTP handler requires a JSON dict
        unified: Final = self.get_supported_perplexity_optional_params()
        mapped: Final = MappingProxyType(
            {
                "max_results": optional_params.get("max_results"),
                "country": optional_params.get("country"),
            }
        )
        return {
            **_domain_filters(optional_params.get("search_domain_filter")),
            **MappingProxyType({key: value for key, value in optional_params.items() if key not in unified}),
            **MappingProxyType({key: value for key, value in mapped.items() if value is not None}),
            "query": " ".join(query) if isinstance(query, list) else query,
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig interface
    ) -> SearchResponse:
        try:
            parsed: Final = _ClawifiResponse.model_validate_json(raw_response.content)
        except ValidationError as exc:
            raise BaseLLMException(
                status_code=502,
                message=f"Clawifi returned an invalid search response: {exc}",
                headers=dict(raw_response.headers),
            ) from exc
        if not parsed.success:
            reasons: Final = "; ".join(error.message for error in parsed.errors) or "no engine answered"
            raise BaseLLMException(
                status_code=502,
                message=f"Clawifi search failed: {reasons}",
                headers=dict(raw_response.headers),
            )
        return SearchResponse.model_validate(
            {
                "results": [
                    SearchResult.model_validate(
                        {
                            **_extra_fields(result),
                            "title": result.title or "",
                            "url": result.url,
                            "snippet": result.snippet or "",
                            "date": result.published_at,
                            "last_updated": result.modified_at,
                        }
                    )
                    for result in parsed.results
                ],
                "provider_metadata": parsed.model_dump(mode="json", exclude={"results"}),
            }
        )

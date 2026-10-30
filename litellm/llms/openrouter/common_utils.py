from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import TokenLimit, get_cached_model_limits
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

_DEFAULT_API_BASE: Final = "https://openrouter.ai/api/v1"
_CHAT_CATALOG_ROUTE: Final = "models/user"
_CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
_BATCH_ONLY_SUFFIX: Final = ":batch"
_CATALOG_TIMEOUT: Final = httpx.Timeout(10.0)
_NO_LIMITS: Final[Mapping[str, int]] = MappingProxyType({})


class OpenRouterException(BaseLLMException):
    pass


def _mapping_or_none(value: object) -> object:
    return value if issubclass(type(value), Mapping) else None


class _TopProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_completion_tokens: TokenLimit = None


class _CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    context_length: TokenLimit = None
    top_provider: Annotated[_TopProvider | None, BeforeValidator(_mapping_or_none)] = None

    def token_limits(self) -> Mapping[str, int]:
        output_limit: Final = self.top_provider.max_completion_tokens if self.top_provider is not None else None
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    ("max_tokens", output_limit or self.context_length),
                    ("max_input_tokens", self.context_length),
                    ("max_output_tokens", output_limit),
                )
                if value is not None
            }
        )


class _Catalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_CatalogModel, ...]


def _chat_catalog_url(api_base: str) -> str:
    return join_api_base(api_base.rstrip("/").removesuffix(_CHAT_COMPLETIONS_PATH), _CHAT_CATALOG_ROUTE)


def _limits_by_catalog_id(content: bytes) -> Mapping[str, Mapping[str, int]]:
    catalog: Final = _Catalog.model_validate_json(content)
    return MappingProxyType({model.id: model.token_limits() for model in catalog.data})


async def aget_openrouter_catalog_limits(
    *, api_base: str, headers: Mapping[str, str], client: AsyncHTTPHandler, cache: InMemoryCache
) -> Mapping[str, Mapping[str, int]]:
    return await get_cached_model_limits(
        url=_chat_catalog_url(api_base),
        headers=headers,
        parse=_limits_by_catalog_id,
        client=client,
        cache=cache,
        timeout=_CATALOG_TIMEOUT,
    )


class OpenRouterModelInfo(BaseLLMModelInfo):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return (
            api_key
            or litellm.api_key
            or litellm.openrouter_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or _DEFAULT_API_BASE

    @staticmethod
    def get_base_model(model: str) -> str:
        return model.removeprefix("openrouter/")

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseLLMModelInfo.validate_environment returns a dict
        resolved_key: Final = self.get_api_key(api_key)
        if resolved_key is None:
            return {**headers}
        return {**headers, "Authorization": f"Bearer {resolved_key}"}

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        resolved_key: Final = self.get_api_key(api_key)
        if resolved_key is None:
            raise ValueError("Set OPENROUTER_API_KEY or the deployment's api_key to list OpenRouter models")
        headers: Final = self.validate_environment(
            headers={}, model="", messages=(), optional_params={}, litellm_params={}, api_key=resolved_key
        )
        response: Final = litellm.module_level_client.get(  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP wrapper has untyped query parameters
            url=_chat_catalog_url(self.get_api_base(api_base)), headers=headers, timeout=_CATALOG_TIMEOUT
        )
        response.raise_for_status()
        return [
            f"openrouter/{model.id}"
            for model in _Catalog.model_validate_json(response.content).data
            if not model.id.endswith(_BATCH_ONLY_SUFFIX)
        ]


def openrouter_catalog_limits_for_model(catalog: Mapping[str, Mapping[str, int]], model: str) -> Mapping[str, int]:
    catalog_id: Final = OpenRouterModelInfo.get_base_model(model)
    candidates: Final = (catalog_id, model, catalog_id.partition(":")[0])
    return next((limits for limits in map(catalog.get, candidates) if limits), _NO_LIMITS)

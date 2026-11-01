import hashlib
import json
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypeAlias

import httpx
from pydantic import BeforeValidator, ConfigDict

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.types.llms.base import LiteLLMBaseModel


class _MetadataCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, *, ttl: int) -> None: ...


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


_TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _InstanceConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    context_length: _TokenLimit = None


class _Instance(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    config: _InstanceConfig


class _ModelV1(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    loaded_instances: tuple[_Instance, ...] = ()


class _ModelsV1(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_ModelV1, ...]

    def context_length(self, model: str) -> int | None:
        instances: Final = tuple(chain.from_iterable(card.loaded_instances for card in self.models))
        exact: Final = tuple(instance.config.context_length for instance in instances if instance.id == model)
        matching: Final = next((card.loaded_instances for card in self.models if card.key == model), ())
        contexts: Final = exact or tuple(instance.config.context_length for instance in matching)
        if not contexts or any(context is None for context in contexts):
            return None
        return min(context for context in contexts if context is not None)


class _ModelV0(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    state: str
    loaded_context_length: _TokenLimit = None


class _ModelsV0(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelV0, ...]

    def context_length(self, model: str) -> int | None:
        card: Final = next((card for card in self.data if card.id == model), None)
        return card.loaded_context_length if card is not None and card.state == "loaded" else None


async def get_lm_studio_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: _MetadataCache,
) -> Mapping[str, int]:
    try:
        base: Final = httpx.URL(api_base)
    except (ValueError, httpx.InvalidURL):
        return MappingProxyType({})
    if base.scheme not in ("http", "https") or not base.host or base.username or base.password:
        return MappingProxyType({})
    path: Final = base.path.rstrip("/")
    suffix: Final = next((suffix for suffix in ("/api/v1", "/api/v0", "/v1") if path.endswith(suffix)), "")
    prefix: Final = path[: -len(suffix)] if suffix else path
    url: Final = str(base.copy_with(path=prefix + "/api/v1/models", fragment=None))
    cache_key: Final = (
        "lm_studio_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, (_ModelsV1, _ModelsV0)):
        return _token_limits(cached.context_length(model))
    try:
        response: Final = await client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped options
            url=url,
            headers=dict(headers),
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        if response.status_code in (404, 405):
            legacy_response: Final = await client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped options
                url=str(base.copy_with(path=prefix + "/api/v0/models", fragment=None)),
                headers=dict(headers),
                timeout=httpx.Timeout(5.0),
                follow_redirects=False,
                max_response_bytes=2 * 1024 * 1024,
            )
            legacy_response.raise_for_status()
            models: Final = _ModelsV0.model_validate_json(legacy_response.content)
            cache.set_cache(cache_key, models, ttl=MODEL_INFO_REFRESH_SECONDS)
            return _token_limits(models.context_length(model))
        response.raise_for_status()
        current_models: Final = _ModelsV1.model_validate_json(response.content)
        cache.set_cache(cache_key, current_models, ttl=MODEL_INFO_REFRESH_SECONDS)
        return _token_limits(current_models.context_length(model))
    except Exception:  # noqa: BLE001  # optional metadata must not interrupt proxy refresh
        verbose_logger.debug("Could not discover LM Studio model token limits")
        cache.set_cache(cache_key, _ModelsV1(models=()), ttl=60)
        return MappingProxyType({})


def _token_limits(context: int | None) -> Mapping[str, int]:
    return (
        MappingProxyType({"max_tokens": context, "max_input_tokens": context, "max_output_tokens": context})
        if context is not None
        else MappingProxyType({})
    )

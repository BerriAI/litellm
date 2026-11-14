import hashlib
import json
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias, TypeVar

import httpx
from pydantic import BeforeValidator, ConfigDict, Field, ValidationError

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.base import LiteLLMBaseModel

RESPONSE_REUSE_SECONDS: Final = 60


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
    kind: str | None = Field(default=None, alias="type")
    loaded_instances: tuple[_Instance, ...]


class _ModelsV1(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_ModelV1, ...]

    def token_limits(self, model: str) -> Mapping[str, int] | None:
        chat_cards: Final = tuple(card for card in self.models if card.kind != "embedding")
        instances: Final = tuple(chain.from_iterable(card.loaded_instances for card in chat_cards))
        named: Final = next((instance for instance in instances if instance.id == model), None)
        first_loaded: Final = next(
            (card.loaded_instances[0] for card in chat_cards if card.key == model and card.loaded_instances), None
        )
        serving: Final = named if named is not None else first_loaded
        if serving is None:
            return MappingProxyType({})
        return _token_limits(serving.config.context_length)


class _ModelV0(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: str | None = Field(default=None, alias="type")
    state: Literal["loaded", "not-loaded"]
    loaded_context_length: _TokenLimit = None


class _ModelsV0(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelV0, ...]

    def token_limits(self, model: str) -> Mapping[str, int] | None:
        card: Final = next((card for card in self.data if card.id == model), None)
        if card is None or card.state != "loaded" or card.kind == "embeddings":
            return MappingProxyType({})
        return _token_limits(card.loaded_context_length)


class _Unavailable(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)


_Models: Final = TypeVar("_Models", _ModelsV1, _ModelsV0)


async def get_lm_studio_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: _MetadataCache,
) -> Mapping[str, int] | None:
    try:
        base: Final = httpx.URL(api_base)
    except (ValueError, httpx.InvalidURL):
        return None
    if base.scheme not in ("http", "https") or not base.host or base.username or base.password:
        return None
    raw_path, separator, query = base.raw_path.partition(b"?")
    path: Final = raw_path.rstrip(b"/")
    suffix: Final = next((suffix for suffix in (b"/api/v1", b"/api/v0", b"/v1") if path.endswith(suffix)), b"")
    prefix: Final = path[: -len(suffix)] if suffix else path
    current_url: Final = base.copy_with(raw_path=prefix + b"/api/v1/models" + separator + query, fragment=None)
    cache_key: Final = (
        "lm_studio_model_info:"
        + hashlib.sha256(json.dumps((str(current_url), sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, _Unavailable):
        return None
    if isinstance(cached, (_ModelsV1, _ModelsV0)):
        return cached.token_limits(model)
    models: Final = await _fetch_models(
        client=client,
        current_url=current_url,
        legacy_url=base.copy_with(raw_path=prefix + b"/api/v0/models" + separator + query, fragment=None),
        headers=headers,
    )
    if models is None:
        verbose_logger.debug("Could not discover LM Studio model token limits")
        cache.set_cache(cache_key, _Unavailable(), ttl=RESPONSE_REUSE_SECONDS)
        return None
    cache.set_cache(cache_key, models, ttl=RESPONSE_REUSE_SECONDS)
    return models.token_limits(model)


async def _fetch_models(
    *, client: AsyncHTTPHandler, current_url: httpx.URL, legacy_url: httpx.URL, headers: Mapping[str, str]
) -> _ModelsV1 | _ModelsV0 | None:
    try:
        current: Final = await _get(client=client, url=current_url, headers=headers)
        current_models: Final = _parse(current, _ModelsV1)
        if current_models is not None or not (current.is_success or current.status_code in (404, 405)):
            return current_models
        return _parse(await _get(client=client, url=legacy_url, headers=headers), _ModelsV0)
    except Exception:  # noqa: BLE001  # optional metadata must not interrupt proxy refresh
        return None


async def _get(*, client: AsyncHTTPHandler, url: httpx.URL, headers: Mapping[str, str]) -> httpx.Response:
    return await client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped options
        url=str(url),
        headers=dict(headers),
        timeout=httpx.Timeout(5.0),
        follow_redirects=False,
        max_response_bytes=2 * 1024 * 1024,
    )


def _parse(response: httpx.Response, shape: type[_Models]) -> _Models | None:
    if not response.is_success:
        return None
    try:
        return shape.model_validate_json(response.content)
    except ValidationError:
        return None


def _token_limits(context: int | None) -> Mapping[str, int] | None:
    return (
        MappingProxyType({"max_tokens": context, "max_input_tokens": context, "max_output_tokens": context})
        if context is not None
        else None
    )

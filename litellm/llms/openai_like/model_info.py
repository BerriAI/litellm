import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import httpx
from pydantic import BeforeValidator, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.utils import add_path_to_api_base

UPSTREAM_MODEL_INFO_FIELDS: Final = "upstream_model_info_fields"
MODEL_INFO_REFRESH_SECONDS: Final = 300
MODEL_INFO_REFRESH_CONCURRENCY: Final = 8
MODEL_INFO_DISCOVERY_PROVIDERS: Final = frozenset({"hosted_vllm", "openai", "text-completion-openai", "openai_like"})
_EMPTY_LIMITS: Final[Mapping[str, int]] = MappingProxyType({})
_METADATA_FIELDS: Final = TypeAdapter(Sequence[object])


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


_TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _ModelCard(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    max_model_len: _TokenLimit = None
    context_length: _TokenLimit = None
    max_input_tokens: _TokenLimit = None
    max_output_tokens: _TokenLimit = None
    name: JsonValue = None
    description: JsonValue = None
    meta: JsonValue = None

    def token_limits(self) -> Mapping[str, int]:
        context: Final = self.max_model_len or self.context_length
        input_limit: Final = self.max_input_tokens or context
        output_limit: Final = self.max_output_tokens or context
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    ("max_tokens", context),
                    ("max_input_tokens", min(input_limit, context) if input_limit and context else input_limit),
                    ("max_output_tokens", min(output_limit, context) if output_limit and context else output_limit),
                )
                if value is not None
            }
        )

    def metadata(self, fields: tuple[str, ...]) -> Mapping[str, JsonValue]:
        display: Final = {"name": self.name, "description": self.description}
        meta_paths: Final = tuple(tuple(field.split(".")[1:]) for field in fields if field.startswith("meta."))
        meta: Final = _select_metadata(self.meta, meta_paths)
        return {
            **{key: value for key, value in display.items() if key in fields and isinstance(value, str)},
            **({"meta": meta} if meta else {}),
        }


def _select_metadata(value: JsonValue, paths: tuple[tuple[str, ...], ...]) -> JsonValue:
    if not isinstance(value, dict):
        return {}
    keys: Final = frozenset(path[0] for path in paths if path and path[0] in value)
    selected: Final = {
        key: value[key]
        if (key,) in paths
        else _select_metadata(value[key], tuple(path[1:] for path in paths if path[0] == key))
        for key in keys
    }
    return {key: item for key, item in selected.items() if (key,) in paths or item}


def upstream_model_info_fields(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        fields: Final = _METADATA_FIELDS.validate_python(value, strict=True)
    except ValidationError:
        return ()
    return tuple(field for field in fields if isinstance(field, str) and "*" not in field and all(field.split(".")))


class _ModelList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelCard, ...] = ()


async def get_openai_compatible_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, int]:
    limits, _ = await discover_openai_compatible_model_info(
        model=model, api_base=api_base, headers=headers, client=client, cache=cache, fields=()
    )
    return limits


async def discover_openai_compatible_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    fields: tuple[str, ...],
) -> tuple[Mapping[str, int], Mapping[str, JsonValue] | None]:
    models: Final = await _get_model_list(api_base=api_base, headers=headers, client=client, cache=cache)
    if models is None:
        return _EMPTY_LIMITS, None
    card: Final = next((card for card in models.data if card.id == model), None)
    if card is None:
        return _EMPTY_LIMITS, {}
    return card.token_limits(), card.metadata(fields) if fields else {}


async def _get_model_list(
    *,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> _ModelList | None:
    url: Final = add_path_to_api_base(api_base, "/v1/models")
    cache_key: Final = (
        "upstream_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, _ModelList):
        return cached
    if cached is False:
        return None

    try:
        response: Final = await client.get(
            url=url,
            headers=dict(headers),
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        models: Final = _ModelList.model_validate_json(response.content)
    except Exception:  # noqa: BLE001  # optional upstream metadata must not interrupt proxy refresh
        verbose_logger.debug("Could not discover upstream model token limits")
        cache.set_cache(cache_key, False, ttl=60)
        return None

    cache.set_cache(cache_key, models, ttl=MODEL_INFO_REFRESH_SECONDS)
    return models

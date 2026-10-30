import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import httpx
from pydantic import BeforeValidator, ConfigDict

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.utils import add_path_to_api_base

MODEL_INFO_REFRESH_SECONDS: Final = 300
MODEL_INFO_REFRESH_CONCURRENCY: Final = 8
MODEL_INFO_DISCOVERY_PROVIDERS: Final = frozenset({"hosted_vllm", "openai", "text-completion-openai", "openai_like"})
_EMPTY_LIMITS: Final[Mapping[str, int]] = MappingProxyType({})
_DISCOVERY_TIMEOUT: Final = httpx.Timeout(5.0)


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _ModelCard(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    max_model_len: TokenLimit = None
    context_length: TokenLimit = None
    max_input_tokens: TokenLimit = None
    max_output_tokens: TokenLimit = None

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


class _ModelList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelCard, ...] = ()


@dataclass(frozen=True, slots=True)
class _CachedModelLimits:
    by_model: Mapping[str, Mapping[str, int]]


def _limits_of_first_card_per_id(content: bytes) -> Mapping[str, Mapping[str, int]]:
    cards: Final = _ModelList.model_validate_json(content).data
    return MappingProxyType({card.id: card.token_limits() for card in reversed(cards)})


async def get_cached_model_limits(
    *,
    url: str,
    headers: Mapping[str, str],
    parse: Callable[[bytes], Mapping[str, Mapping[str, int]]],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    timeout: httpx.Timeout = _DISCOVERY_TIMEOUT,
) -> Mapping[str, Mapping[str, int]]:
    cache_key: Final = (
        "upstream_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, _CachedModelLimits):
        return cached.by_model

    try:
        response: Final = await client.get(
            url=url,
            headers=dict(headers),
            timeout=timeout,
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        limits: Final = parse(response.content)
    except Exception:  # noqa: BLE001  # optional upstream metadata must not interrupt proxy refresh
        verbose_logger.debug("Could not discover upstream model token limits")
        cache.set_cache(cache_key, _CachedModelLimits(MappingProxyType({})), ttl=60)
        return MappingProxyType({})

    cache.set_cache(cache_key, _CachedModelLimits(limits), ttl=MODEL_INFO_REFRESH_SECONDS)
    return limits


async def get_openai_compatible_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, int]:
    limits: Final = await get_cached_model_limits(
        url=add_path_to_api_base(api_base, "/v1/models"),
        headers=headers,
        parse=_limits_of_first_card_per_id,
        client=client,
        cache=cache,
    )
    return limits.get(model, _EMPTY_LIMITS)

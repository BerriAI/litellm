import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import _add_path_to_api_base  # pyright: ignore[reportPrivateUsage]  # shared provider URL helper

MODEL_INFO_REFRESH_SECONDS: Final = 300
MODEL_INFO_REFRESH_CONCURRENCY: Final = 8
_EMPTY_LIMITS: Final[Mapping[str, int]] = MappingProxyType({})


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


_TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _ModelCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    max_model_len: _TokenLimit = None
    context_length: _TokenLimit = None
    max_input_tokens: _TokenLimit = None
    max_output_tokens: _TokenLimit = None

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


class _ModelList(BaseModel):
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
    url: Final = _add_path_to_api_base(api_base, "/v1/models")
    cache_key: Final = (
        "upstream_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, _ModelList):
        return next((card.token_limits() for card in cached.data if card.id == model), _EMPTY_LIMITS)

    try:
        response: Final = await client.get(
            url=url,
            headers=dict(headers),  # mutable-ok: AsyncHTTPHandler requires a concrete dict
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        models: Final = _ModelList.model_validate_json(response.content)
    except Exception:  # noqa: BLE001  # optional upstream metadata must not interrupt proxy refresh
        verbose_logger.debug("Could not discover upstream model token limits")
        cache.set_cache(cache_key, _ModelList(), ttl=60)
        return _EMPTY_LIMITS

    cache.set_cache(cache_key, models, ttl=MODEL_INFO_REFRESH_SECONDS)
    return next((card.token_limits() for card in models.data if card.id == model), _EMPTY_LIMITS)

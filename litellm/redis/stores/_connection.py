from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm._redis import _merge_redis_kwargs
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    REDIS_CIRCUIT_BREAKER_ENABLED,
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
)
from litellm.rust_bridge._native import NativeRedisClient

_KWARGS: Final = TypeAdapter(dict[str, object])


def _redis_kwargs(connection: RedisCache) -> Mapping[str, object]:
    return _merge_redis_kwargs(
        _KWARGS.validate_python(
            connection.redis_kwargs  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped legacy dict
        )
    )


def _breaker() -> tuple[int, float, float] | None:
    if not REDIS_CIRCUIT_BREAKER_ENABLED:
        return None
    return (
        REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        float(REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT),
        REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
    )


def native_client(connection: RedisCache | None) -> NativeRedisClient | None:
    if connection is None:
        return None
    return NativeRedisClient.connect(_redis_kwargs(connection), namespace=connection.namespace, breaker=_breaker())

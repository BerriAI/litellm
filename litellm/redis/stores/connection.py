from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.constants import (
    REDIS_CIRCUIT_BREAKER_ENABLED,
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
)
from litellm.rust_bridge._native import NativeRedisClient

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

_KWARGS: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class RedisConnection:
    """A Redis connection the stores share. Pass one to every store that should ride the same client."""

    client: NativeRedisClient


def _breaker() -> tuple[int, float, float] | None:
    if not REDIS_CIRCUIT_BREAKER_ENABLED:
        return None
    return (
        REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        float(REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT),
        REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
    )


def connect(redis_cache: RedisCache | None) -> RedisConnection | None:
    settings: Final = None if redis_cache is None else redis_cache.store_settings
    if settings is None:
        return None
    from litellm._redis import (  # redis-py, which importing litellm must not require
        _merge_redis_kwargs,  # pyright: ignore[reportPrivateUsage]  # Reuse the shared internal kwargs normalizer
    )

    client: Final = NativeRedisClient.connect(
        _merge_redis_kwargs(_KWARGS.validate_python(settings.redis_kwargs)),
        default_ttl_seconds=settings.default_ttl,
        namespace=settings.namespace,
        breaker=_breaker(),
    )
    return None if client is None else RedisConnection(client)


def client_of(connection: RedisConnection | None) -> NativeRedisClient | None:
    return None if connection is None else connection.client

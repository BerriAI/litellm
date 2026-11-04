from typing import Final

from pydantic import TypeAdapter

from litellm._redis import _merge_redis_kwargs
from litellm.constants import (
    REDIS_CIRCUIT_BREAKER_ENABLED,
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
)
from litellm.redis.native_settings import NativeRedisSettings
from litellm.rust_bridge._native import NativeRedisClient

_KWARGS: Final = TypeAdapter(dict[str, object])


def _breaker() -> tuple[int, float, float] | None:
    if not REDIS_CIRCUIT_BREAKER_ENABLED:
        return None
    return (
        REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        float(REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT),
        REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
    )


def native_client(settings: NativeRedisSettings | None) -> NativeRedisClient | None:
    if settings is None:
        return None
    return NativeRedisClient.connect(
        _merge_redis_kwargs(_KWARGS.validate_python(settings.redis_kwargs)),
        namespace=settings.namespace,
        breaker=_breaker(),
    )

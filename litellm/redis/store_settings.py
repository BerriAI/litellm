from collections.abc import Mapping
from dataclasses import dataclass

from litellm.constants import (
    REDIS_CIRCUIT_BREAKER_ENABLED,
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
)


@dataclass(frozen=True, slots=True)
class RedisBreakerSettings:
    failure_threshold: int
    recovery_timeout: float
    timeout_min_duration: float


@dataclass(frozen=True, slots=True)
class RedisStoreSettings:
    """What `connect` takes from a `RedisCache`, captured once when the cache is built."""

    redis_kwargs: Mapping[str, object]
    namespace: str | None
    default_ttl: int
    breaker: RedisBreakerSettings | None


def breaker_settings() -> RedisBreakerSettings | None:
    if not REDIS_CIRCUIT_BREAKER_ENABLED:
        return None
    return RedisBreakerSettings(
        failure_threshold=REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        timeout_min_duration=REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
    )

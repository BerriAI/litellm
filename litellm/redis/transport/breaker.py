from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

from litellm.caching.redis_cache import (
    RedisCircuitBreaker,
    RedisCircuitBreakerOpenError,
    _is_redis_health_failure,  # pyright: ignore[reportPrivateUsage]  # same health classification as RedisCache
    _run_under_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same admission and health signal as RedisCache
    is_redis_error,
    is_redis_timeout_failure,
)
from litellm.redis.transport.contract import Admit, RedisFailure

_T: Final = TypeVar("_T")


async def run_under_breaker(
    breaker: RedisCircuitBreaker, name: str, admit: Admit, call: Callable[[], Awaitable[_T]]
) -> _T | RedisFailure:
    try:
        return await _run_under_circuit_breaker(breaker, name, lambda: admit(call))
    except Exception as e:
        failure: Final = failure_from(e)
        if failure is None:
            raise
        return failure


def failure_from(exc: BaseException) -> RedisFailure | None:
    if isinstance(exc, RedisCircuitBreakerOpenError):
        return RedisFailure(kind="breaker_open", message=str(exc))
    if _is_redis_health_failure(exc):
        return RedisFailure(kind="transport", message=str(exc), timeout=is_redis_timeout_failure(exc))
    if is_redis_error(exc):
        return RedisFailure(kind="server", message=str(exc))
    return None

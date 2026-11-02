"""Redis call and circuit breaker events from the native Redis client, reported the way ``RedisCache`` reports them."""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Literal, TypeAlias

from litellm._service_logger import ServiceLogging
from litellm.types.services import ServiceTypes

Failure: TypeAlias = Literal["timeout", "transport", "server", "decode", "breaker_open"]
BreakerState: TypeAlias = Literal["closed", "open", "half_open"]

_HEALTH_FAILURE_CLASSES: Final = {"timeout": "timeout", "transport": "connectivity"}


class RedisCallError(Exception):
    """The failure class of a native Redis call, the only detail the Rust client hands over."""


@lru_cache(maxsize=1)
def _service_logging() -> ServiceLogging:
    return ServiceLogging()


def record_redis_call(call_type: str, duration: float, failure: Failure | None, caller: str | None) -> None:
    if failure == "breaker_open":
        return
    logger: Final = _service_logging()
    if failure is None:
        logger.service_success_hook(service=ServiceTypes.REDIS, duration=duration, call_type=call_type, caller=caller)
        return
    logger.service_failure_hook(
        service=ServiceTypes.REDIS,
        duration=duration,
        error=RedisCallError(failure),
        call_type=call_type,
        caller=caller,
    )
    failure_class: Final = _HEALTH_FAILURE_CLASSES.get(failure)
    if failure_class is not None:
        logger.redis_breaker_hook(transitions_to=None, failure_class=failure_class)


def record_redis_breaker(state: BreakerState) -> None:
    _service_logging().redis_breaker_hook(transitions_to=state, failure_class=None)

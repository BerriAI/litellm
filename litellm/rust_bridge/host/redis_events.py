"""Redis call and circuit breaker events from the native Redis client, reported the way ``RedisCache`` reports them."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import lru_cache
from typing import Final, Literal, TypeAlias

from redis import exceptions as redis_exceptions
from typing_extensions import assert_never

from litellm._internal_context import current_request_parent_otel_span
from litellm._logging import verbose_logger
from litellm._service_logger import ServiceLogging
from litellm.types.services import ServiceTypes

CallFailure: TypeAlias = Literal["timeout", "transport", "server", "decode"]
Failure: TypeAlias = CallFailure | Literal["breaker_open"]
BreakerState: TypeAlias = Literal["closed", "open", "half_open"]
HealthFailure: TypeAlias = Literal["timeout", "connectivity"]
Hook: TypeAlias = Callable[[], Coroutine[object, object, None]]


@lru_cache(maxsize=1)
def _service_logging() -> ServiceLogging:
    return ServiceLogging()


def _redis_error(failure: CallFailure) -> redis_exceptions.RedisError:
    """The redis-py exception ``RedisCache`` would have caught, so ``error_class`` keeps its labels."""
    match failure:
        case "timeout":
            return redis_exceptions.TimeoutError(failure)
        case "transport":
            return redis_exceptions.ConnectionError(failure)
        case "server":
            return redis_exceptions.ResponseError(failure)
        case "decode":
            return redis_exceptions.InvalidResponse(failure)
        case _:
            assert_never(failure)


def _health_failure(failure: CallFailure) -> HealthFailure | None:
    match failure:
        case "timeout":
            return "timeout"
        case "transport":
            return "connectivity"
        case "server" | "decode":
            return None
        case _:
            assert_never(failure)


async def _guarded(hook: Hook) -> None:
    try:
        await hook()
    except Exception as e:  # noqa: BLE001  # Service-event hooks must not fail Redis operations
        verbose_logger.exception("Error emitting a Redis service event - %s", e)


def _spawn(hook: Hook) -> None:
    asyncio.get_running_loop().create_task(_guarded(hook))


def _emit(event_loop: asyncio.AbstractEventLoop | None, hook: Hook) -> None:
    """Schedule ``hook`` as a task on ``event_loop`` in the current context, as ``RedisCache`` did with
    ``asyncio.create_task``; with no open loop, run it here the way ``ServiceLogging``'s sync hooks do."""
    if event_loop is None or event_loop.is_closed():
        asyncio.run(_guarded(hook))
        return
    event_loop.call_soon_threadsafe(_spawn, hook)


def record_redis_call(
    event_loop: asyncio.AbstractEventLoop | None,
    call_type: str,
    start_time: float,
    end_time: float,
    failure: Failure | None,
) -> None:
    logger: Final = _service_logging()
    duration: Final = end_time - start_time
    parent_otel_span: Final = current_request_parent_otel_span()
    match failure:
        case "breaker_open":
            return
        case None:
            _emit(
                event_loop,
                lambda: logger.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    call_type=call_type,
                    duration=duration,
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
        case "timeout" | "transport" | "server" | "decode":
            error: Final = _redis_error(failure)
            _emit(
                event_loop,
                lambda: logger.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=duration,
                    error=error,
                    call_type=call_type,
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
            health: Final = _health_failure(failure)
            if health is not None:
                logger.redis_breaker_hook(transitions_to=None, failure_class=health)
        case _:
            assert_never(failure)


def record_redis_breaker(state: BreakerState) -> None:
    _service_logging().redis_breaker_hook(transitions_to=state, failure_class=None)

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, TypeVar

from litellm._internal_context import current_service_target
from litellm._logging import verbose_logger
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import (
    _get_call_stack_info,  # pyright: ignore[reportPrivateUsage]  # same caller attribution as RedisCache events
    _log_redis_failure_value,  # pyright: ignore[reportPrivateUsage]  # same timeout throttle as RedisCache logging
)
from litellm.redis.transport.contract import RedisFailure
from litellm.types.services import ServiceTypes

_T: Final = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RedisServiceEvents:
    service_logging: ServiceLogging

    async def observe(self, call_type: str, call: Callable[[], Awaitable[_T | RedisFailure]]) -> _T | RedisFailure:
        if not self.service_logging.has_sinks():
            result: Final = await call()
            _log_failure(call_type, result)
            return result
        caller: Final = _get_call_stack_info()
        start_time: Final = time.time()
        observed: Final = await call()
        end_time: Final = time.time()
        _log_failure(call_type, observed)
        await self._emit(call_type, observed, caller=caller, start_time=start_time, end_time=end_time)
        return observed

    async def _emit(self, call_type: str, result: object, *, caller: str, start_time: float, end_time: float) -> None:
        if not isinstance(result, RedisFailure):
            await ServiceLogging.emit_guarded(
                lambda: self.service_logging.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    call_type=call_type,
                    duration=end_time - start_time,
                    start_time=start_time,
                    end_time=end_time,
                    caller=caller,
                )
            )
            return
        if result.kind == "breaker_open":
            return
        await ServiceLogging.emit_guarded(
            lambda: self.service_logging.async_service_failure_hook(
                service=ServiceTypes.REDIS,
                duration=end_time - start_time,
                error=result.message,
                call_type=call_type,
                start_time=start_time,
                end_time=end_time,
                caller=caller,
            )
        )


def _log_failure(call_type: str, result: object) -> None:
    if not isinstance(result, RedisFailure):
        return
    target: Final = current_service_target()
    _log_redis_failure_value(
        verbose_logger, logging.WARNING, f"Redis {call_type} failed" + (f" for {target}" if target else ""), result
    )

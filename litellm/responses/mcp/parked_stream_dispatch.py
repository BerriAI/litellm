from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Final, Protocol

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import UncachedPrice

PARKED_DISPATCH: Final = "_deferred_stream_complete_args"
DISPATCH_RELEASE: Final = "_on_deferred_stream_complete"
SERVED_FROM_CACHE: Final = UncachedPrice(cost=0.0, cost_breakdown=None)


class DispatchesSuccessHandlers(Protocol):
    async def dispatch_success_handlers(
        self,
        result: object = None,
        start_time: object = None,
        end_time: object = None,
        cache_hit: object = None,
        prefer_async_handlers: bool = False,
    ) -> None: ...


class DeferredStreamRelease(Protocol):
    def __call__(self, *parked: object) -> Coroutine[object, object, None]: ...


def success_dispatcher(logger: LiteLLMLoggingObj) -> DispatchesSuccessHandlers:
    return logger


@dataclass(frozen=True, slots=True)
class ParkedSuccess:
    assembled: object
    cache_hit: object
    logged_on: LiteLLMLoggingObj | None = None

    async def dispatch(self, request_logger: LiteLLMLoggingObj) -> None:
        logger: Final = request_logger if self.logged_on is None else self.logged_on
        await success_dispatcher(logger).dispatch_success_handlers(
            self.assembled, cache_hit=self.cache_hit, start_time=None, end_time=None, prefer_async_handlers=True
        )


def parked_success(parked: tuple[object, ...]) -> ParkedSuccess | None:
    match parked:
        case (assembled, cache_hit):
            return ParkedSuccess(assembled, cache_hit)
        case (assembled, cache_hit, LiteLLMLoggingObj() as logged_on):
            return ParkedSuccess(assembled, cache_hit, logged_on)
        case _:
            return None

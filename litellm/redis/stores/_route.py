from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Final, Generic, TypeVar, final

from litellm._internal_context import current_service_target, service_target
from litellm.caching.redis_batch import Batch, NativePending, RequestRedisBatches, active_request_redis_batches
from litellm.rust_bridge._native import NativeRequestRedis, NativeSend

_T = TypeVar("_T")
_U = TypeVar("_U")


@final
class Reply(Generic[_T]):
    """A store reply that builds its coroutine only once awaited, so one declared on a batch and never awaited,
    such as an armed prefetch the request no longer needs, leaves no coroutine behind to warn when collected."""

    __slots__ = ("_start",)

    def __init__(self, start: Callable[[], Awaitable[_T]]) -> None:
        self._start: Final = start

    def __await__(self) -> Generator[object, None, _T]:
        return self._start().__await__()

    def map(self, convert: Callable[[_T], _U]) -> Reply[_U]:
        return Reply(lambda: _converted(self, convert))


async def _converted(reply: Reply[_T], convert: Callable[[_T], _U]) -> _U:
    return convert(await reply)


async def _ready(value: _T) -> _T:
    return value


def ready(value: _T) -> Reply[_T]:
    """A reply already known, awaited without yielding to the event loop."""
    return Reply(lambda: _ready(value))


@dataclass(frozen=True, slots=True)
class Route:
    """The batch a native store op rides, and how its reply is awaited. A pre-call read awaits through
    ``native_result``, so the pre-call flush hooks run first and late readers join the same trip. The reply is
    reported under the service target the op was declared with, however late it is awaited."""

    send: NativeSend
    request: RequestRedisBatches | None

    def result(self, pending: NativePending[_T]) -> Reply[_T]:
        request: Final = self.request
        if request is None:
            resolved: Final = pending.resolve()
            return Reply(lambda: resolved)
        target: Final = current_service_target()
        return Reply(lambda: _resolved_under(target, request.native_result(pending)))


async def _resolved_under(target: str | None, reply: Awaitable[_T]) -> _T:
    with service_target(target):
        return await reply


def as_tuple(reply: Reply[Iterable[_T]]) -> Reply[tuple[_T, ...]]:
    return reply.map(tuple)


def route(batch: Batch) -> Route:
    request: Final = active_request_redis_batches()
    match batch:
        case Batch.NONE:
            return Route(NativeSend.Now(), None)
        case Batch.PRE_CALL | Batch.POST_CALL:
            return Route(NativeSend.Now(), None) if request is None else Route(request.native_send(batch), request)


def post_call_request() -> NativeRequestRedis | None:
    """The current request's native batches for a post-call write nobody awaits, None outside a request."""
    request: Final = active_request_redis_batches()
    return None if request is None else request.native_post_call()

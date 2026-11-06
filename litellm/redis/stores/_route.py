from __future__ import annotations

from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from typing import Final, TypeVar

from litellm._internal_context import current_service_target, service_target
from litellm.caching.redis_batch import Batch, NativePending, RequestRedisBatches, active_request_redis_batches
from litellm.rust_bridge._native import NativeRequestRedis, NativeSend

_T: Final = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Route:
    """The batch a native store op rides, and how its reply is awaited. A pre-call read awaits through
    ``native_result``, so the pre-call flush hooks run first and late readers join the same trip. The reply is
    reported under the service target the op was declared with, however late it is awaited."""

    send: NativeSend
    request: RequestRedisBatches | None

    def result(self, pending: NativePending[_T]) -> Awaitable[_T]:
        if self.request is None:
            return pending.resolve()
        return _resolved_under(current_service_target(), self.request.native_result(pending))


async def _resolved_under(target: str | None, reply: Awaitable[_T]) -> _T:
    with service_target(target):
        return await reply


async def as_tuple(reply: Awaitable[Iterable[_T]]) -> tuple[_T, ...]:
    return tuple(await reply)


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

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Final, TypeVar

from litellm.caching.redis_batch import Batch, NativePending, RequestRedisBatches, active_request_redis_batches
from litellm.rust_bridge._native import NativeRequestRedis, NativeSend

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Route:
    """The batch a native store op rides, and how its reply is awaited. A pre-call read awaits through
    ``native_result``, so the pre-call flush hooks run first and late readers join the same trip."""

    send: NativeSend
    request: RequestRedisBatches | None

    def result(self, pending: NativePending[_T]) -> Awaitable[_T]:
        return pending.resolve() if self.request is None else self.request.native_result(pending)


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

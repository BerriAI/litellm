from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import BatchResult, active_request_redis_batches

_PREFETCH_SLOT: Final = "routing_read"
ROUTER_USAGE_TARGET: Final = "router_usage"


@dataclass(frozen=True, slots=True)
class TierRead:
    cache: DualCache
    keys: tuple[str, ...]


def selector_read(cache: DualCache, keys: Sequence[str]) -> TierRead:
    return TierRead(cache, tuple(keys))


@dataclass(frozen=True, slots=True)
class ArmedRead:
    read: TierRead
    fetched: frozenset[str]
    result: BatchResult[Mapping[str, object]]
    previous: Mapping[str, float | None]

    def release(self) -> None:
        self.read.cache._rollback_redis_batch_key_reservations(  # pyright: ignore[reportPrivateUsage]  # the selector owns this tier
            dict(self.previous)
        )

    async def settle(self, future: asyncio.Future[Mapping[str, object]]) -> None:
        if future.cancelled() or future.exception() is not None:
            self.release()
            return
        try:
            pending: Final = await self.read.cache._prepare_batch_get(  # pyright: ignore[reportPrivateUsage]  # the selector owns this tier
                list(self.read.keys), local_only=True
            )
            values: Final = future.result()
            missed: Final = {
                key: values[key]
                for key, local in zip(self.read.keys, pending.result)
                if local is None and values.get(key) is not None
            }
            await self.read.cache._apply_batch_get(pending, missed)  # pyright: ignore[reportPrivateUsage]  # the selector owns this tier
        except Exception:
            self.release()
            raise


def arm(read: TierRead) -> None:
    request: Final = active_request_redis_batches()
    redis: Final = read.cache.redis_cache
    if request is None or redis is None or _PREFETCH_SLOT in request.prefetched:
        return
    due, previous = read.cache.reserve_redis_batch_reads(read.keys)
    if not due:
        return
    with service_target(ROUTER_USAGE_TARGET):
        result: Final = request.batch(redis).mget(tuple(due))
    armed_read: Final = ArmedRead(read=read, fetched=frozenset(due), result=result, previous=previous)
    result.on_settled(armed_read.settle)
    request.prefetched[_PREFETCH_SLOT] = armed_read


def armed() -> bool:
    request: Final = active_request_redis_batches()
    return request is not None and _PREFETCH_SLOT in request.prefetched


def _take(read: TierRead) -> ArmedRead | None:
    request: Final = active_request_redis_batches()
    if request is None:
        return None
    taken: Final = request.prefetched.pop(_PREFETCH_SLOT, None)
    if isinstance(taken, ArmedRead) and set(taken.read.keys).issuperset(read.keys):
        return taken
    if isinstance(taken, ArmedRead):
        taken.release()
    return None


async def _read_prefetched(read: TierRead) -> Sequence[object | None] | None:
    prefetch: Final = _take(read)
    if prefetch is None:
        return None
    try:
        values: Final = await prefetch.result
    except Exception as e:  # noqa: BLE001  # the read falls back to the tier's own Redis read
        verbose_router_logger.debug("routing prefetch failed, reading again: %s", e)
        return None
    pending: Final = await read.cache._prepare_batch_get(  # pyright: ignore[reportPrivateUsage]  # the selector owns this tier
        list(read.keys), local_only=True
    )
    if any(key not in prefetch.fetched for key, value in zip(read.keys, pending.result) if value is None):
        return None
    return await read.cache._apply_batch_get(  # pyright: ignore[reportPrivateUsage]  # the selector owns this tier
        pending, {key: values.get(key) for key, value in zip(read.keys, pending.result) if value is None}
    )


async def read(read: TierRead, parent_otel_span: Span | None) -> Sequence[object | None] | None:
    with service_target(ROUTER_USAGE_TARGET):
        prefetched: Final = await _read_prefetched(read)
        if prefetched is not None:
            return prefetched
        results: Final = await DualCache.async_batch_get_cache_shared(
            ((read.cache, list(read.keys)),), parent_otel_span=parent_otel_span
        )
        return results[0]

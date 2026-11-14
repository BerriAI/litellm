from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Final, TypeVar

from typing_extensions import Never

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch, NativePending, active_request_redis_batches
from litellm.redis.stores._route import Reply, Route, as_tuple, post_call_request, route
from litellm.redis.stores.rate_limits.handle import RateLimitsHandle
from litellm.redis.stores.rate_limits.types import (
    RATE_LIMITS_TARGET,
    AcquireReply,
    AtomicApplied,
    AtomicCounter,
    AtomicOverLimit,
    AtomicReply,
    CounterIncrement,
    CounterKind,
    GaugeOverLimit,
    GaugesAcquired,
    GuardedReply,
    LocalValue,
    RateLimitCounter,
    RateLimitEntry,
    RateLimitGauge,
    RateLimitScope,
    RateLimitWindow,
    WindowCount,
)

__all__ = [
    "RATE_LIMITS_TARGET",
    "AcquireReply",
    "AtomicApplied",
    "AtomicCounter",
    "AtomicOverLimit",
    "AtomicReply",
    "CounterIncrement",
    "CounterKind",
    "GaugeOverLimit",
    "GaugesAcquired",
    "GuardedReply",
    "LocalValue",
    "RateLimitCounter",
    "RateLimitEntry",
    "RateLimitGauge",
    "RateLimitScope",
    "RateLimitStore",
    "RateLimitWindow",
    "WindowCount",
]

_T = TypeVar("_T")


async def _no_redis() -> Never:
    raise RuntimeError("the rate limiter has no Redis to run its scripts on")


def _script(pending: NativePending[_T] | None, target: Route) -> Reply[_T]:
    """A script's reply; one declared without Redis raises where it is awaited, like a failed script."""
    return Reply(_no_redis) if pending is None else target.result(pending)


class RateLimitStore:
    """The v3 limiter's keyspace on the Rust store. Each script is declared on the batch its ``Batch`` picks when
    called, so several declared before any is awaited share one trip; awaiting the result sends it."""

    def __init__(self, handle: RateLimitsHandle) -> None:
        self._handle: Final = handle

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    @property
    def has_request_batch(self) -> bool:
        """Whether ``Batch.PRE_CALL`` scripts share the request's pre-call trip and ``*_post_call`` writes ride its
        post-call batch, rather than each go out on its own or be refused."""
        return self.has_redis and active_request_redis_batches() is not None

    def slot_groups(self, scopes: Sequence[RateLimitScope], force: bool) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(group) for group in self._handle.slot_groups(scopes, force))

    def local(self, entries: Sequence[RateLimitEntry]) -> tuple[LocalValue | None, ...]:
        return tuple(self._handle.get_local(entry) for entry in entries)

    def set_local(self, entry: RateLimitEntry, value: LocalValue, ttl: float | None) -> None:
        self._handle.set_local(entry, value, ttl)

    @with_service_target(RATE_LIMITS_TARGET)
    async def get(self, entry: RateLimitEntry, backfill_ttl: float | None = None) -> LocalValue | None:
        """``DualCache.async_get_cache``: memory, else Redis written back to memory. A memory hit, or a miss
        without Redis, answers without yielding to the event loop, as ``DualCache`` did."""
        local: Final = self._handle.get_local(entry)
        if local is not None or not self.has_redis:
            return local
        return await self._handle.get(entry, backfill_ttl_seconds=backfill_ttl)

    @with_service_target(RATE_LIMITS_TARGET)
    async def get_from_redis(self, counter: RateLimitCounter, backfill_ttl: float | None = None) -> LocalValue | None:
        """Redis, never memory, for a count every worker must agree on: ``DualCache.async_get_cache`` answers from
        memory first, and a worker's own copy carries just the usage it served itself. The count read is written
        back to a memory that holds nothing for ``backfill_ttl``, as ``DualCache`` backfilled a miss; a copy the limiter
        keeps for the whole window is left alone, so a Redis outage still falls back to it. Without Redis memory is
        all there is."""
        if not self.has_redis:
            return self._handle.get_local(counter)
        _window, count = await self.read_counters([counter], batch=Batch.NONE)
        if count is None:
            return None
        number: Final = float(count)
        value: Final[LocalValue] = int(number) if number.is_integer() else number
        if self._handle.get_local(counter) is None:
            self._handle.set_local(counter, value, backfill_ttl)
        return value

    @with_service_target(RATE_LIMITS_TARGET)
    def sliding_window(
        self, counters: Sequence[RateLimitCounter], now: int, window_size: int, *, batch: Batch
    ) -> Awaitable[tuple[WindowCount, ...]]:
        target: Final = route(batch)
        return as_tuple(_script(self._handle.sliding_window(counters, now, window_size, target.send), target))

    @with_service_target(RATE_LIMITS_TARGET)
    def read_counters(self, counters: Sequence[RateLimitCounter], *, batch: Batch) -> Awaitable[tuple[str | None, ...]]:
        """Each counter's window start then its count, as ``BATCH_COUNTER_READ_SCRIPT`` reads them."""
        target: Final = route(batch)
        return as_tuple(_script(self._handle.read_counters(counters, target.send), target))

    @with_service_target(RATE_LIMITS_TARGET)
    def check_and_increment(self, counters: Sequence[AtomicCounter], *, batch: Batch) -> Awaitable[AtomicReply]:
        target: Final = route(batch)
        return _script(self._handle.check_and_increment(counters, target.send), target)

    @with_service_target(RATE_LIMITS_TARGET)
    def guarded_increment(
        self,
        counter: RateLimitCounter,
        expected_window_start: str,
        amount: float,
        ttl: int | None,
        *,
        batch: Batch,
    ) -> Awaitable[GuardedReply]:
        target: Final = route(batch)
        return _script(self._handle.guarded_increment(counter, expected_window_start, amount, ttl, target.send), target)

    @with_service_target(RATE_LIMITS_TARGET)
    def parallel_acquire(
        self, gauges: Sequence[tuple[RateLimitScope, int]], slot_ttl: int, slot_id: str, *, batch: Batch
    ) -> Awaitable[AcquireReply]:
        target: Final = route(batch)
        return _script(self._handle.parallel_acquire(gauges, slot_ttl, slot_id, target.send), target)

    @with_service_target(RATE_LIMITS_TARGET)
    def parallel_release(
        self, gauges: Sequence[RateLimitScope], slot_id: str, *, batch: Batch
    ) -> Awaitable[tuple[int, ...]]:
        target: Final = route(batch)
        return as_tuple(_script(self._handle.parallel_release(gauges, slot_id, target.send), target))

    @with_service_target(RATE_LIMITS_TARGET)
    def parallel_count(
        self, gauges: Sequence[RateLimitScope], slot_ttl: int, *, batch: Batch
    ) -> Awaitable[tuple[int, ...]]:
        target: Final = route(batch)
        return as_tuple(_script(self._handle.parallel_count(gauges, slot_ttl, target.send), target))

    @with_service_target(RATE_LIMITS_TARGET)
    def token_increment(self, increments: Sequence[CounterIncrement], *, batch: Batch) -> Awaitable[tuple[float, ...]]:
        target: Final = route(batch)
        return as_tuple(_script(self._handle.token_increment(increments, target.send), target))

    @with_service_target(RATE_LIMITS_TARGET)
    def token_increment_post_call(self, increments: Sequence[CounterIncrement]) -> bool:
        """``TOKEN_INCREMENT_SCRIPT`` on the request's post-call batch, falling back to ``increment_many`` when
        the script fails; False outside a request scope or without Redis."""
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            return False
        return self._handle.token_increment_post_call(request, increments)

    @with_service_target(RATE_LIMITS_TARGET)
    def parallel_release_post_call(self, gauges: Sequence[RateLimitScope], slot_id: str) -> bool:
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            return False
        return self._handle.parallel_release_post_call(request, gauges, slot_id)

    @with_service_target(RATE_LIMITS_TARGET)
    async def increment(self, increment: CounterIncrement) -> int | float:
        """``DualCache.async_increment_cache``: memory, then Redis keeping an existing TTL."""
        return await self._handle.increment(increment)

    @with_service_target(RATE_LIMITS_TARGET)
    async def increment_many(self, increments: Sequence[CounterIncrement]) -> None:
        """``DualCache.async_increment_cache_pipeline``: memory, then one Redis pipeline refreshing each TTL."""
        await self._handle.increment_many(increments)

    @with_service_target(RATE_LIMITS_TARGET)
    async def increment_many_post_call(self, increments: Sequence[CounterIncrement]) -> None:
        """Memory now; the Redis increments ride the request's post-call batch, or go out at once without one."""
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            await self._handle.increment_many(increments)
            return
        self._handle.increment_many_post_call(request, increments)

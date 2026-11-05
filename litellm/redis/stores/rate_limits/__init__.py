from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Final, Never, TypeVar

from litellm.caching.redis_batch import Batch, NativePending, active_request_redis_batches
from litellm.redis.stores._route import Route, post_call_request, route
from litellm.redis.stores.rate_limits.handle import HandleFactory, RateLimitsHandle
from litellm.redis.stores.rate_limits.types import (
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
    NativeAtomicCounter,
    NativeCounter,
    NativeEntry,
    NativeIncrement,
    NativeOverLimit,
    NativeScope,
    RateLimitCounter,
    RateLimitEntry,
    RateLimitGauge,
    RateLimitScope,
    RateLimitWindow,
    WindowCount,
    native_atomic,
    native_counter,
    native_entry,
    native_increment,
    native_scope,
)
from litellm.rust_bridge._native import NativeRateLimits, NativeRedisClient

__all__ = [
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
    "HandleFactory",
    "LocalValue",
    "NativeAtomicCounter",
    "NativeCounter",
    "NativeEntry",
    "NativeIncrement",
    "NativeScope",
    "RateLimitCounter",
    "RateLimitEntry",
    "RateLimitGauge",
    "RateLimitScope",
    "RateLimitStore",
    "RateLimitWindow",
    "RateLimitsHandle",
    "WindowCount",
]

_T = TypeVar("_T")


async def _no_redis() -> Never:
    raise RuntimeError("the rate limiter has no Redis to run its scripts on")


def _script(pending: NativePending[_T] | None, target: Route) -> Awaitable[_T]:
    """A script's reply; one declared without Redis raises where it is awaited, like a failed script."""
    return _no_redis() if pending is None else target.result(pending)


async def _window_counts(reply: Awaitable[list[tuple[str, int]]]) -> tuple[WindowCount, ...]:
    return tuple(WindowCount(start, count) for start, count in await reply)


async def _values(reply: Awaitable[list[str | None]]) -> tuple[str | None, ...]:
    return tuple(await reply)


async def _counts(reply: Awaitable[list[int]]) -> tuple[int, ...]:
    return tuple(await reply)


async def _totals(reply: Awaitable[list[float]]) -> tuple[float, ...]:
    return tuple(await reply)


async def _guarded(reply: Awaitable[tuple[bool, int]]) -> GuardedReply:
    applied, count = await reply
    return GuardedReply(applied=applied, count=count)


async def _atomic_reply(reply: Awaitable[tuple[NativeOverLimit | None, list[tuple[int, int]]]]) -> AtomicReply:
    over_limit, applied = await reply
    if over_limit is None:
        return AtomicApplied(tuple(applied))
    index, current, limit = over_limit
    return AtomicOverLimit(index=index, current=current, limit=limit)


async def _acquire_reply(reply: Awaitable[tuple[NativeOverLimit | None, list[int]]]) -> AcquireReply:
    over_limit, in_flight = await reply
    if over_limit is None:
        return GaugesAcquired(tuple(in_flight))
    index, count, limit = over_limit
    return GaugeOverLimit(index=index, in_flight=count, limit=limit)


class RateLimitStore:
    """The v3 limiter's keyspace on the Rust store. Each script is declared on the batch its ``Batch`` picks when
    called, so several declared before any is awaited share one trip; awaiting the result sends it."""

    def __init__(
        self,
        client: NativeRedisClient | None,
        backfill_ttl: float | None = None,
        redis_ttl: int = 60,
        *,
        handle: HandleFactory = NativeRateLimits,
    ) -> None:
        self._handle: Final = handle(client, backfill_ttl, redis_ttl)

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    @property
    def has_pre_call_batch(self) -> bool:
        """Whether ``Batch.PRE_CALL`` scripts share the request's pre-call trip rather than each go out on its own."""
        return self.has_redis and active_request_redis_batches() is not None

    @property
    def has_post_call_batch(self) -> bool:
        """Whether a ``*_post_call`` write would ride the request's post-call batch rather than be refused."""
        return self.has_redis and active_request_redis_batches() is not None

    def slot_groups(self, scopes: Sequence[RateLimitScope], force: bool) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(group) for group in self._handle.slot_groups(tuple(map(native_scope, scopes)), force))

    def local(self, entries: Sequence[RateLimitEntry]) -> tuple[LocalValue | None, ...]:
        return tuple(self._handle.get_local(native_entry(entry)) for entry in entries)

    def set_local(self, entry: RateLimitEntry, value: LocalValue, ttl: float | None) -> None:
        self._handle.set_local(native_entry(entry), value, ttl)

    async def get(self, entry: RateLimitEntry, backfill_ttl: float | None = None) -> LocalValue | None:
        """``DualCache.async_get_cache``: memory, else Redis written back to memory. A memory hit, or a miss
        without Redis, answers without yielding to the event loop, as ``DualCache`` did."""
        native: Final = native_entry(entry)
        local: Final = self._handle.get_local(native)
        if local is not None or not self.has_redis:
            return local
        return await self._handle.get(native, backfill_ttl_seconds=backfill_ttl)

    def sliding_window(
        self, counters: Sequence[RateLimitCounter], now: int, window_size: int, *, batch: Batch
    ) -> Awaitable[tuple[WindowCount, ...]]:
        target: Final = route(batch)
        native: Final = tuple(map(native_counter, counters))
        return _window_counts(_script(self._handle.sliding_window(native, now, window_size, target.send), target))

    def read_counters(self, counters: Sequence[RateLimitCounter], *, batch: Batch) -> Awaitable[tuple[str | None, ...]]:
        """Each counter's window start then its count, as ``BATCH_COUNTER_READ_SCRIPT`` reads them."""
        target: Final = route(batch)
        native: Final = tuple(map(native_counter, counters))
        return _values(_script(self._handle.read_counters(native, target.send), target))

    def check_and_increment(self, counters: Sequence[AtomicCounter], *, batch: Batch) -> Awaitable[AtomicReply]:
        target: Final = route(batch)
        native: Final = tuple(map(native_atomic, counters))
        return _atomic_reply(_script(self._handle.check_and_increment(native, target.send), target))

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
        native: Final = native_counter(counter)
        return _guarded(
            _script(self._handle.guarded_increment(native, expected_window_start, amount, ttl, target.send), target)
        )

    def parallel_acquire(
        self, gauges: Sequence[tuple[RateLimitScope, int]], slot_ttl: int, slot_id: str, *, batch: Batch
    ) -> Awaitable[AcquireReply]:
        target: Final = route(batch)
        native: Final = tuple((native_scope(scope), limit) for scope, limit in gauges)
        return _acquire_reply(_script(self._handle.parallel_acquire(native, slot_ttl, slot_id, target.send), target))

    def parallel_release(
        self, gauges: Sequence[RateLimitScope], slot_id: str, *, batch: Batch
    ) -> Awaitable[tuple[int, ...]]:
        target: Final = route(batch)
        native: Final = tuple(map(native_scope, gauges))
        return _counts(_script(self._handle.parallel_release(native, slot_id, target.send), target))

    def parallel_count(
        self, gauges: Sequence[RateLimitScope], slot_ttl: int, *, batch: Batch
    ) -> Awaitable[tuple[int, ...]]:
        target: Final = route(batch)
        native: Final = tuple(map(native_scope, gauges))
        return _counts(_script(self._handle.parallel_count(native, slot_ttl, target.send), target))

    def token_increment(self, increments: Sequence[CounterIncrement], *, batch: Batch) -> Awaitable[tuple[float, ...]]:
        target: Final = route(batch)
        native: Final = tuple(map(native_increment, increments))
        return _totals(_script(self._handle.token_increment(native, target.send), target))

    def token_increment_post_call(self, increments: Sequence[CounterIncrement]) -> bool:
        """``TOKEN_INCREMENT_SCRIPT`` on the request's post-call batch, falling back to ``increment_many`` when
        the script fails; False outside a request scope or without Redis."""
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            return False
        return self._handle.token_increment_post_call(request, tuple(map(native_increment, increments)))

    def parallel_release_post_call(self, gauges: Sequence[RateLimitScope], slot_id: str) -> bool:
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            return False
        return self._handle.parallel_release_post_call(request, tuple(map(native_scope, gauges)), slot_id)

    async def increment(self, increment: CounterIncrement) -> int | float:
        """``DualCache.async_increment_cache``: memory, then Redis keeping an existing TTL."""
        return await self._handle.increment(native_increment(increment))

    async def increment_many(self, increments: Sequence[CounterIncrement]) -> None:
        """``DualCache.async_increment_cache_pipeline``: memory, then one Redis pipeline refreshing each TTL."""
        await self._handle.increment_many(tuple(map(native_increment, increments)))

    async def increment_many_post_call(self, increments: Sequence[CounterIncrement]) -> None:
        """Memory now; the Redis increments ride the request's post-call batch, or go out at once without one."""
        native: Final = tuple(map(native_increment, increments))
        request: Final = post_call_request() if self.has_redis else None
        if request is None:
            await self._handle.increment_many(native)
            return
        self._handle.increment_many_post_call(request, native)

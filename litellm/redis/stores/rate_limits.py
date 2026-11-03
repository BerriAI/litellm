from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypeVar

from pydantic import TypeAdapter

from litellm.caching.redis_batch import active_request_redis_batches
from litellm.rust_bridge._native import NativeRateLimits, NativeRedisClient, NativeRequestRedis

_R = TypeVar("_R")
CounterKind: TypeAlias = Literal["requests", "tokens"]
LocalValue: TypeAlias = int | float | str | dict[str, float]


@dataclass(frozen=True, slots=True, order=True)
class RateLimitScope:
    """A rate-limit descriptor; every key of its window shares the ``{key:value}`` hash tag."""

    descriptor_key: str
    descriptor_value: str

    def counter(self, kind: CounterKind) -> RateLimitCounter:
        return RateLimitCounter(self, kind)

    @property
    def window(self) -> RateLimitWindow:
        return RateLimitWindow(self)

    @property
    def gauge(self) -> RateLimitGauge:
        return RateLimitGauge(self)


@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    scope: RateLimitScope


@dataclass(frozen=True, slots=True, order=True)
class RateLimitCounter:
    scope: RateLimitScope
    kind: CounterKind

    @property
    def window(self) -> RateLimitWindow:
        return RateLimitWindow(self.scope)


@dataclass(frozen=True, slots=True)
class RateLimitGauge:
    scope: RateLimitScope


RateLimitEntry: TypeAlias = RateLimitWindow | RateLimitCounter | RateLimitGauge


@dataclass(frozen=True, slots=True)
class CounterIncrement:
    counter: RateLimitCounter
    amount: int | float
    ttl: int | None


@dataclass(frozen=True, slots=True)
class AtomicCounter:
    counter: RateLimitCounter
    limit: int
    increment: int
    ttl: int
    window_size: int


@dataclass(frozen=True, slots=True)
class WindowCount:
    window_start: str
    count: int


@dataclass(frozen=True, slots=True)
class AtomicApplied:
    """Per counter, the count after the increment and the start of the window it landed in."""

    counters: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AtomicOverLimit:
    index: int
    current: int
    limit: int


AtomicReply: TypeAlias = AtomicApplied | AtomicOverLimit


@dataclass(frozen=True, slots=True)
class GaugesAcquired:
    in_flight: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GaugeOverLimit:
    index: int
    in_flight: int
    limit: int


AcquireReply: TypeAlias = GaugesAcquired | GaugeOverLimit


@dataclass(frozen=True, slots=True)
class GuardedReply:
    applied: bool
    count: int


_WINDOW_COUNTS: Final = TypeAdapter(list[tuple[str, int]])
_VALUES: Final = TypeAdapter(list[str | None])
_COUNTS: Final = TypeAdapter(list[int])
_TOTALS: Final = TypeAdapter(list[float])
_GUARDED: Final = TypeAdapter(tuple[bool, int])
_ATOMIC: Final = TypeAdapter(tuple[tuple[int, int, int] | None, list[tuple[int, int]]])
_ACQUIRE: Final = TypeAdapter(tuple[tuple[int, int, int] | None, list[int]])

NativeScope: TypeAlias = tuple[str, str]
NativeCounter: TypeAlias = tuple[str, str, CounterKind]
NativeEntry: TypeAlias = tuple[Literal["window", "requests", "tokens", "max_parallel_requests"], str, str]
NativeIncrement: TypeAlias = tuple[NativeCounter, int | float, int | None]
NativeAtomicCounter: TypeAlias = tuple[NativeCounter, int, int, int, int]


class PendingScript(Protocol):
    def resolve(self) -> Awaitable[object]: ...


class RateLimitsHandle(Protocol):
    def has_redis(self) -> bool: ...
    def slot_groups(self, scopes_: Sequence[NativeScope], force: bool) -> list[list[int]]: ...
    def local(self, entries: Sequence[NativeEntry]) -> list[LocalValue | None]: ...
    def set_local(self, entry: NativeEntry, value: LocalValue, ttl_seconds: float | None) -> None: ...
    def get(self, entry: NativeEntry, *, backfill_ttl_seconds: float | None = None) -> Awaitable[LocalValue | None]: ...
    def sliding_window(
        self,
        counters_: Sequence[NativeCounter],
        now: int,
        window_size: int,
        *,
        request: NativeRequestRedis | None = None,
    ) -> PendingScript: ...
    def read_counters(
        self, counters_: Sequence[NativeCounter], *, request: NativeRequestRedis | None = None
    ) -> PendingScript: ...
    def check_and_increment(
        self, counters_: Sequence[NativeAtomicCounter], *, request: NativeRequestRedis | None = None
    ) -> PendingScript: ...
    def guarded_increment(
        self,
        counter: NativeCounter,
        expected_window_start: str,
        amount: float,
        ttl_seconds: int | None,
        *,
        request: NativeRequestRedis | None = None,
    ) -> PendingScript: ...
    def parallel_acquire(
        self,
        gauges: Sequence[tuple[NativeScope, int]],
        slot_ttl_seconds: int,
        slot_id: str,
        *,
        request: NativeRequestRedis | None = None,
    ) -> PendingScript: ...
    def parallel_release(
        self, gauges: Sequence[NativeScope], slot_id: str, *, request: NativeRequestRedis | None = None
    ) -> PendingScript: ...
    def parallel_count(
        self, gauges: Sequence[NativeScope], slot_ttl_seconds: int, *, request: NativeRequestRedis | None = None
    ) -> PendingScript: ...
    def token_increment(
        self, increments_: Sequence[NativeIncrement], *, request: NativeRequestRedis | None = None
    ) -> PendingScript: ...
    def token_increment_post_call(
        self, request: NativeRequestRedis, increments_: Sequence[NativeIncrement]
    ) -> Awaitable[None]: ...
    def parallel_release_post_call(
        self, request: NativeRequestRedis, gauges: Sequence[NativeScope], slot_id: str
    ) -> Awaitable[None]: ...
    def increment(self, increment: NativeIncrement) -> Awaitable[int | float]: ...
    def increment_many(self, increments_: Sequence[NativeIncrement]) -> Awaitable[None]: ...
    def increment_many_post_call(
        self, request: NativeRequestRedis, increments_: Sequence[NativeIncrement]
    ) -> Awaitable[None]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None, float | None, int], RateLimitsHandle]


def _scope(scope: RateLimitScope) -> NativeScope:
    return (scope.descriptor_key, scope.descriptor_value)


def _counter(counter: RateLimitCounter) -> NativeCounter:
    return (counter.scope.descriptor_key, counter.scope.descriptor_value, counter.kind)


def _entry(entry: RateLimitEntry) -> NativeEntry:
    match entry:
        case RateLimitWindow(scope):
            return ("window", scope.descriptor_key, scope.descriptor_value)
        case RateLimitGauge(scope):
            return ("max_parallel_requests", scope.descriptor_key, scope.descriptor_value)
        case RateLimitCounter(scope, kind):
            return (kind, scope.descriptor_key, scope.descriptor_value)


def _increment(increment: CounterIncrement) -> NativeIncrement:
    return (_counter(increment.counter), increment.amount, increment.ttl)


def _atomic(counter: AtomicCounter) -> NativeAtomicCounter:
    return (_counter(counter.counter), counter.limit, counter.increment, counter.ttl, counter.window_size)


async def _window_counts(pending: Awaitable[object]) -> tuple[WindowCount, ...]:
    return tuple(WindowCount(start, count) for start, count in _WINDOW_COUNTS.validate_python(await pending))


async def _values(pending: Awaitable[object]) -> tuple[str | None, ...]:
    return tuple(_VALUES.validate_python(await pending))


async def _counts(pending: Awaitable[object]) -> tuple[int, ...]:
    return tuple(_COUNTS.validate_python(await pending))


async def _totals(pending: Awaitable[object]) -> tuple[float, ...]:
    return tuple(_TOTALS.validate_python(await pending))


async def _guarded(pending: Awaitable[object]) -> GuardedReply:
    applied, count = _GUARDED.validate_python(await pending)
    return GuardedReply(applied=applied, count=count)


async def _atomic_reply(pending: Awaitable[object]) -> AtomicReply:
    over_limit, applied = _ATOMIC.validate_python(await pending)
    if over_limit is None:
        return AtomicApplied(tuple(applied))
    index, current, limit = over_limit
    return AtomicOverLimit(index=index, current=current, limit=limit)


async def _acquire_reply(pending: Awaitable[object]) -> AcquireReply:
    over_limit, in_flight = _ACQUIRE.validate_python(await pending)
    if over_limit is None:
        return GaugesAcquired(tuple(in_flight))
    index, count, limit = over_limit
    return GaugeOverLimit(index=index, in_flight=count, limit=limit)


class RateLimitStore:
    """The v3 limiter's keyspace on the Rust store. Inside a request scope every script rides the request's
    native pre-call batch, and ``declare_*`` returns it without awaiting so several share one trip (None
    outside a scope, where the plain method runs on a batch of its own)."""

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
        return self._handle.has_redis()

    @property
    def has_post_call_batch(self) -> bool:
        """Whether a ``*_post_call`` write would ride the request's post-call batch rather than be refused."""
        return self.has_redis and active_request_redis_batches() is not None

    def slot_groups(self, scopes: Sequence[RateLimitScope], force: bool) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(group) for group in self._handle.slot_groups(tuple(map(_scope, scopes)), force))

    def local(self, entries: Sequence[RateLimitEntry]) -> tuple[LocalValue | None, ...]:
        return tuple(self._handle.local(tuple(map(_entry, entries))))

    def set_local(self, entry: RateLimitEntry, value: LocalValue, ttl: float | None) -> None:
        self._handle.set_local(_entry(entry), value, ttl)

    async def get(self, entry: RateLimitEntry, backfill_ttl: float | None = None) -> LocalValue | None:
        """``DualCache.async_get_cache``: memory, else Redis written back to memory. A memory hit, or a miss
        without Redis, answers without yielding to the event loop, as ``DualCache`` did."""
        native: Final = _entry(entry)
        local: Final = self._handle.local((native,))[0]
        if local is not None or not self.has_redis:
            return local
        return await self._handle.get(native, backfill_ttl_seconds=backfill_ttl)

    def declare_sliding_window(
        self, counters: Sequence[RateLimitCounter], now: int, window_size: int
    ) -> Awaitable[tuple[WindowCount, ...]] | None:
        return self._declare(
            lambda request: self._handle.sliding_window(
                tuple(map(_counter, counters)), now, window_size, request=request
            ),
            _window_counts,
        )

    async def sliding_window(
        self, counters: Sequence[RateLimitCounter], now: int, window_size: int
    ) -> tuple[WindowCount, ...]:
        return await _window_counts(
            self._handle.sliding_window(tuple(map(_counter, counters)), now, window_size).resolve()
        )

    async def read_counters(self, counters: Sequence[RateLimitCounter]) -> tuple[str | None, ...]:
        """Each counter's window start then its count, as ``BATCH_COUNTER_READ_SCRIPT`` reads them."""
        native: Final = tuple(map(_counter, counters))
        return await _values(self._handle.read_counters(native).resolve())

    def declare_check_and_increment(self, counters: Sequence[AtomicCounter]) -> Awaitable[AtomicReply] | None:
        native: Final = tuple(map(_atomic, counters))
        return self._declare(lambda request: self._handle.check_and_increment(native, request=request), _atomic_reply)

    async def check_and_increment(self, counters: Sequence[AtomicCounter]) -> AtomicReply:
        return await _atomic_reply(self._handle.check_and_increment(tuple(map(_atomic, counters))).resolve())

    async def guarded_increment(
        self, counter: RateLimitCounter, expected_window_start: str, amount: float, ttl: int | None
    ) -> GuardedReply:
        native: Final = _counter(counter)
        return await _guarded(self._handle.guarded_increment(native, expected_window_start, amount, ttl).resolve())

    async def parallel_acquire(
        self, gauges: Sequence[tuple[RateLimitScope, int]], slot_ttl: int, slot_id: str
    ) -> AcquireReply:
        native: Final = tuple((_scope(scope), limit) for scope, limit in gauges)
        return await _acquire_reply(self._handle.parallel_acquire(native, slot_ttl, slot_id).resolve())

    async def parallel_release(self, gauges: Sequence[RateLimitScope], slot_id: str) -> tuple[int, ...]:
        native: Final = tuple(map(_scope, gauges))
        return await _counts(self._handle.parallel_release(native, slot_id).resolve())

    async def parallel_count(self, gauges: Sequence[RateLimitScope], slot_ttl: int) -> tuple[int, ...]:
        native: Final = tuple(map(_scope, gauges))
        return await _counts(self._handle.parallel_count(native, slot_ttl).resolve())

    async def token_increment(self, increments: Sequence[CounterIncrement]) -> tuple[float, ...]:
        return await _totals(self._handle.token_increment(tuple(map(_increment, increments))).resolve())

    async def token_increment_post_call(self, increments: Sequence[CounterIncrement]) -> bool:
        """``TOKEN_INCREMENT_SCRIPT`` on the request's post-call batch, falling back to ``increment_many`` when
        the script fails; False outside a request scope or without Redis."""
        request: Final = self._post_call_request()
        if request is None:
            return False
        await self._handle.token_increment_post_call(request, tuple(map(_increment, increments)))
        return True

    async def parallel_release_post_call(self, gauges: Sequence[RateLimitScope], slot_id: str) -> bool:
        request: Final = self._post_call_request()
        if request is None:
            return False
        await self._handle.parallel_release_post_call(request, tuple(map(_scope, gauges)), slot_id)
        return True

    async def increment(self, increment: CounterIncrement) -> int | float:
        """``DualCache.async_increment_cache``: memory, then Redis keeping an existing TTL."""
        return await self._handle.increment(_increment(increment))

    async def increment_many(self, increments: Sequence[CounterIncrement]) -> None:
        """``DualCache.async_increment_cache_pipeline``: memory, then one Redis pipeline refreshing each TTL."""
        await self._handle.increment_many(tuple(map(_increment, increments)))

    async def increment_many_post_call(self, increments: Sequence[CounterIncrement]) -> None:
        """Memory now; the Redis increments ride the request's post-call batch, or go out at once without one."""
        request: Final = self._post_call_request()
        native: Final = tuple(map(_increment, increments))
        if request is None:
            await self._handle.increment_many(native)
            return
        await self._handle.increment_many_post_call(request, native)

    def _post_call_request(self) -> NativeRequestRedis | None:
        batches: Final = active_request_redis_batches()
        if batches is None or not self.has_redis:
            return None
        return batches.native_post_call()

    def _declare(
        self,
        declare: Callable[[NativeRequestRedis], PendingScript],
        decode: Callable[[Awaitable[object]], Awaitable[_R]],
    ) -> Awaitable[_R] | None:
        batches: Final = active_request_redis_batches()
        if batches is None or not self.has_redis:
            return None
        return decode(batches.native_result(declare(batches.native())))

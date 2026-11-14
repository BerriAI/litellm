"""A stand-in for ``NativeRouterUsage`` with the native store's semantics: memory in front of a Redis dict.
``redis`` None is no Redis, ``breaker_open`` refuses every Redis round trip and ``redis_down`` fails it, and every
call the store makes is logged in ``calls``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar

from litellm.redis.stores.router_usage import CounterIncrement, RouterUsageStore, UsageCounter
from litellm.router_python_host import PythonHost
from litellm.rust_bridge._native import NativeRedisClient
from tests.unit.caching.redis_batch_double import fake_pending

T = TypeVar("T")


def _done(value: T) -> asyncio.Future[T]:
    future: Final[asyncio.Future[T]] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@dataclass
class FakeUsageHandle:
    local: dict[UsageCounter, float] = field(default_factory=dict)
    redis: dict[UsageCounter, float] | None = None
    breaker_open: bool = False
    redis_down: bool = False
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    @property
    def has_redis(self) -> bool:
        return self.redis is not None

    def attach(self, client: NativeRedisClient, /) -> bool:
        attached: Final = self.redis is None
        self.redis = {} if self.redis is None else self.redis
        return attached

    def _reachable(self) -> dict[UsageCounter, float] | None:
        return None if self.breaker_open or self.redis_down else self.redis

    def _increment(self, increment: CounterIncrement) -> float:
        self.local[increment.counter] = self.local.get(increment.counter, 0) + increment.amount
        redis: Final = self._reachable()
        if redis is None:
            return self.local[increment.counter]
        redis[increment.counter] = redis.get(increment.counter, 0) + increment.amount
        return redis[increment.counter]

    def increment_sync(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float:
        self.calls.append(("increment", (counter, value, ttl_seconds)))
        return self._increment(CounterIncrement(counter, value, ttl_seconds))

    def increment(self, counter: UsageCounter, value: float, ttl_seconds: int) -> Awaitable[float]:
        return _done(self.increment_sync(counter, value, ttl_seconds))

    def increment_pipeline(self, increments: Sequence[CounterIncrement]) -> asyncio.Future[list[float]]:
        self.calls.append(("pipeline", tuple(increments)))
        return _done([self._increment(increment) for increment in increments])

    def increment_post_call(self, request: object, increments: Sequence[CounterIncrement]) -> None:
        """Memory now; the Redis half is the request's, so only the declaration is logged."""
        self.calls.append(("post_call", tuple(increments)))
        for increment in increments:
            self.local[increment.counter] = self.local.get(increment.counter, 0) + increment.amount

    def increment_local(self, increment: CounterIncrement) -> float:
        self.local[increment.counter] = self.local.get(increment.counter, 0) + increment.amount
        return self.local[increment.counter]

    def get_local(self, counter: UsageCounter) -> float | None:
        return self.local.get(counter)

    def get_local_many(self, counters: Sequence[UsageCounter]) -> list[float | None]:
        return [self.local.get(counter) for counter in counters]

    def set_local(self, counter: UsageCounter, value: float) -> None:
        self.local[counter] = value

    def increment_remote(self, increments: Sequence[CounterIncrement]) -> asyncio.Future[list[float] | None]:
        self.calls.append(("remote", tuple(increments)))
        redis: Final = self._reachable()
        if redis is None:
            return _done(None)
        totals: Final = [redis.get(i.counter, 0) + i.amount for i in increments]
        redis.update({i.counter: total for i, total in zip(increments, totals)})
        return _done(totals)

    def current_sync(self, counter: UsageCounter, limit: float) -> float | None:
        self.calls.append(("current", (counter, limit)))
        local: Final = self.local.get(counter)
        if self.redis is None or self.breaker_open or (local is not None and local >= limit):
            return local
        return None if self.redis_down else self.redis.get(counter)

    def current(self, counter: UsageCounter, limit: float) -> asyncio.Future[float | None]:
        return _done(self.current_sync(counter, limit))

    def shared_or_local(self, counter: UsageCounter) -> asyncio.Future[float | None]:
        self.calls.append(("shared_or_local", (counter,)))
        redis: Final = self._reachable()
        shared: Final = None if redis is None else redis.get(counter)
        return _done(self.local.get(counter) if shared is None else shared)

    def _read(self, counters: Sequence[UsageCounter]) -> list[float | None]:
        redis: Final = self._reachable()
        return [self.local.get(c, None if redis is None else redis.get(c)) for c in counters]

    def get_many_sync(self, counters: Sequence[UsageCounter]) -> list[float | None]:
        self.calls.append(("get_many", tuple(counters)))
        return self._read(counters)

    def get_many(self, counters: Sequence[UsageCounter]) -> asyncio.Future[list[float | None]]:
        return _done(self.get_many_sync(counters))

    def usage(self, counters: Sequence[UsageCounter], send: object) -> FakeUsageRead:
        """Declared on the fake request's batch, or alone for ``NativeSend.Now()``."""
        self.calls.append(("usage", tuple(counters)))
        return FakeUsageRead(
            fake_pending(send, lambda: self._read(counters)),
            lambda: self.calls.append(("release", tuple(counters))),
        )


@dataclass(frozen=True, slots=True)
class FakeUsageRead:
    read: object
    on_release: Callable[[], None]

    def release(self) -> None:
        self.on_release()


def usage_store(handle: FakeUsageHandle) -> RouterUsageStore:
    return RouterUsageStore(handle)  # pyright: ignore[reportArgumentType]  # implements what the tests call


class UsageHost(PythonHost):
    """The Python host with a router's usage store counting in ``handle``."""

    __slots__ = ("_usage",)

    def __init__(self, handle: FakeUsageHandle) -> None:
        self._usage: Final = usage_store(handle)

    def router_usage(self, redis: object) -> RouterUsageStore:
        return self._usage

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.redis.stores._route import post_call_request, ready, route
from litellm.redis.stores.router_usage.handle import RouterUsageHandle
from litellm.redis.stores.router_usage.types import (
    ROUTER_USAGE_TARGET,
    CounterIncrement,
    CounterKind,
    CounterScope,
    UsageCounter,
)
from litellm.rust_bridge._native import NativeRedisStoreError

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = [
    "ROUTER_USAGE_TARGET",
    "CounterIncrement",
    "CounterKind",
    "CounterScope",
    "RouterUsage",
    "RouterUsageStore",
    "UsageCounter",
]


class RouterUsage(Protocol):
    def attach_redis(self, redis: RedisStores) -> None: ...
    def flush_local_counters(self) -> None: ...
    async def increment(self, counter: UsageCounter, value: float, ttl: int) -> float | None: ...
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    async def get_many(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
    def increment_sync(self, counter: UsageCounter, value: float, ttl: int) -> float | None: ...
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    def get_many_sync(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
    @property
    def has_redis(self) -> bool: ...
    def usage(self, counters: Sequence[UsageCounter], *, batch: Batch) -> Awaitable[Sequence[float | None]]: ...
    async def increment_pipeline(self, increments: Sequence[CounterIncrement]) -> Sequence[float] | None: ...
    async def increment_post_call(self, increments: Sequence[CounterIncrement]) -> None: ...
    async def increment_remote(self, increments: Sequence[CounterIncrement]) -> Sequence[float] | None: ...
    def increment_local(self, increment: CounterIncrement) -> float: ...
    def get_local(self, counter: UsageCounter) -> float | None: ...
    def get_local_many(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
    def set_local(self, counter: UsageCounter, value: float) -> None: ...
    async def current(self, counter: UsageCounter, limit: float) -> float | None: ...
    def current_sync(self, counter: UsageCounter, limit: float) -> float | None: ...
    async def shared_or_local(self, counter: UsageCounter) -> float | None: ...
    def snapshot(self) -> Mapping[str, float]: ...


class RouterUsageStore:
    def __init__(self, handle: RouterUsageHandle) -> None:
        self._handle: Final = handle

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    def attach_redis(self, redis: RedisStores) -> None:
        """Adds the Redis tier once, keeping this router's memory counts, as ``DualCache.attach_redis_cache`` does."""
        if redis.client is not None:
            self._handle.attach(redis.client)

    def flush_local_counters(self) -> None:
        self._handle.flush_local_counters()

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment(self, counter: UsageCounter, value: float, ttl: int) -> float | None:
        """Without Redis every async method answers from memory without suspending, as ``DualCache``'s did."""
        if not self.has_redis:
            return self.increment_sync(counter, value, ttl)
        return await self._handle.increment(counter, value, ttl)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None:
        if not self.has_redis:
            self.increment_deployment_sync(deployment_id, model, minute, tokens, requests)
            return
        await self._handle.increment_deployment(deployment_id, model, minute, tokens, requests)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def get_many(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]:
        if not self.has_redis:
            return self.get_many_sync(counters)
        return await self._handle.get_many(counters)

    @with_service_target(ROUTER_USAGE_TARGET)
    def increment_sync(self, counter: UsageCounter, value: float, ttl: int) -> float | None:
        """``increment`` for a sync call, which has no event loop to await on."""
        return self._handle.increment_sync(counter, value, ttl)

    @with_service_target(ROUTER_USAGE_TARGET)
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None:
        self._handle.increment_deployment_sync(deployment_id, model, minute, tokens, requests)

    @with_service_target(ROUTER_USAGE_TARGET)
    def get_many_sync(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]:
        return self._handle.get_many_sync(counters)

    @with_service_target(ROUTER_USAGE_TARGET)
    def usage(self, counters: Sequence[UsageCounter], *, batch: Batch) -> Awaitable[Sequence[float | None]]:
        """``get_many`` declared now on the batch ``batch`` picks, so a pick's read can share the request's
        pre-call trip. Without Redis it is answered from memory without suspending."""
        if not self.has_redis:
            return ready(self.get_many_sync(counters))
        target: Final = route(batch)
        return target.result(self._handle.usage(counters, target.send))

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_pipeline(self, increments: Sequence[CounterIncrement]) -> Sequence[float] | None:
        """``DualCache.async_increment_cache_pipeline``, on its own trip."""
        if not self.has_redis:
            return [self._handle.increment_local(increment) for increment in increments]
        return await self._handle.increment_pipeline(increments)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_post_call(self, increments: Sequence[CounterIncrement]) -> None:
        """``DualCache.async_increment_cache_pipeline_post_call``: memory now, Redis on the request's post-call
        pipeline, or on a trip of its own outside a request."""
        request: Final = post_call_request()
        if request is None or not self.has_redis:
            await self.increment_pipeline(increments)
            return
        self._handle.increment_post_call(request, increments)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_remote(self, increments: Sequence[CounterIncrement]) -> Sequence[float] | None:
        """``redis_cache.async_increment_pipeline``: Redis alone, None without it, raising what it raises there,
        so ``log_redis_failure`` keeps an open breaker at DEBUG and throttles timeouts."""
        try:
            return await self._handle.increment_remote(increments)
        except NativeRedisStoreError as e:
            match e.args[:1]:
                case ("breaker_open",):
                    raise RedisCircuitBreakerOpenError(*e.args[1:]) from e
                case ("timeout",):
                    raise TimeoutError(*e.args[1:]) from e
                case _:
                    raise

    def increment_local(self, increment: CounterIncrement) -> float:
        return self._handle.increment_local(increment)

    def get_local(self, counter: UsageCounter) -> float | None:
        return self._handle.get_local(counter)

    def get_local_many(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]:
        return self._handle.get_local_many(counters)

    def set_local(self, counter: UsageCounter, value: float) -> None:
        self._handle.set_local(counter, value)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def current(self, counter: UsageCounter, limit: float) -> float | None:
        """Memory alone once it reaches ``limit``, else Redis's count. No Redis and an open breaker serve memory,
        and any other Redis failure reads as None, as ``RedisCache.async_get_cache`` swallows it."""
        if not self.has_redis:
            return self._handle.get_local(counter)
        return await self._handle.current(counter, limit)

    @with_service_target(ROUTER_USAGE_TARGET)
    def current_sync(self, counter: UsageCounter, limit: float) -> float | None:
        return self._handle.current_sync(counter, limit)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def shared_or_local(self, counter: UsageCounter) -> float | None:
        """What every pod counted in Redis, else this router's memory count when Redis has none, is absent, refuses
        or fails."""
        if not self.has_redis:
            return self._handle.get_local(counter)
        return await self._handle.shared_or_local(counter)

    def snapshot(self) -> Mapping[str, float]:
        return self._handle.snapshot()

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from litellm._internal_context import with_service_target
from litellm.redis.stores.router_usage.handle import RouterUsageHandle
from litellm.redis.stores.router_usage.types import (
    ROUTER_USAGE_TARGET,
    CounterKind,
    UsageCounter,
)

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = [
    "ROUTER_USAGE_TARGET",
    "CounterKind",
    "RouterUsage",
    "RouterUsageStore",
    "UsageCounter",
]


class RouterUsage(Protocol):
    def attach_redis(self, redis: RedisStores) -> None: ...
    def flush_local_counters(self) -> None: ...
    async def increment(self, counter: UsageCounter, value: float, ttl: int) -> float: ...
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    async def get_many(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
    def increment_sync(self, counter: UsageCounter, value: float, ttl: int) -> float: ...
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    def get_many_sync(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
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
    async def increment(self, counter: UsageCounter, value: float, ttl: int) -> float:
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
    def increment_sync(self, counter: UsageCounter, value: float, ttl: int) -> float:
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

    def snapshot(self) -> Mapping[str, float]:
        return self._handle.snapshot()

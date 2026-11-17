from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch
from litellm.redis.stores._route import as_tuple, ready, route
from litellm.redis.stores.router_cooldowns.handle import RouterCooldownsHandle
from litellm.redis.stores.router_cooldowns.types import ROUTER_COOLDOWNS_TARGET, ActiveCooldowns, CooldownCacheValue

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = [
    "ROUTER_COOLDOWNS_TARGET",
    "ActiveCooldowns",
    "CooldownCacheValue",
    "RouterCooldownStore",
    "RouterCooldowns",
]


class RouterCooldowns(Protocol):
    def attach_redis(self, redis: RedisStores) -> None: ...
    @property
    def has_redis(self) -> bool: ...
    def flush_local_counters(self) -> None: ...
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    async def async_put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    async def async_increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    def active(self, model_ids: Sequence[str], *, batch: Batch) -> Awaitable[ActiveCooldowns]: ...
    def active_sync(self, model_ids: Sequence[str]) -> ActiveCooldowns: ...
    def minimum(self, model_ids: Sequence[str], default: float) -> float: ...
    def evict_local(self, model_id: str) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


class RouterCooldownStore:
    def __init__(self, handle: RouterCooldownsHandle) -> None:
        self._handle: Final = handle

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    def attach_redis(self, redis: RedisStores) -> None:
        """Adds the Redis tier once, keeping this router's memory, as ``DualCache.attach_redis_cache`` does."""
        if redis.client is not None:
            self._handle.attach(redis.client)

    def flush_local_counters(self) -> None:
        self._handle.flush_local_counters()

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None:
        self._handle.put(model_id, value, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int:
        return self._handle.increment_allowed_fails(model_id, suffix, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    async def async_put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None:
        await self._handle.async_put(model_id, value, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    async def async_increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int:
        return await self._handle.async_increment_allowed_fails(model_id, suffix, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def active(self, model_ids: Sequence[str], *, batch: Batch) -> Awaitable[ActiveCooldowns]:
        """Declared now on the batch ``batch`` picks, so several reads can share one trip before any is awaited.
        Without Redis it is answered from memory without suspending, as ``DualCache``'s memory read was."""
        if not self.has_redis:
            return ready(self.active_sync(model_ids))
        target: Final = route(batch)
        return as_tuple(target.result(self._handle.active(model_ids, target.send)))

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def active_sync(self, model_ids: Sequence[str]) -> ActiveCooldowns:
        return tuple(self._handle.active_sync(model_ids))

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def minimum(self, model_ids: Sequence[str], default: float) -> float:
        return self._handle.minimum(model_ids) or default

    def evict_local(self, model_id: str) -> None:
        self._handle.delete_local(model_id)

    def snapshot(self) -> Mapping[str, CooldownCacheValue]:
        return self._handle.snapshot()

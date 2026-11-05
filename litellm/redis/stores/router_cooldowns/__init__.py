from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch
from litellm.redis.stores._route import route
from litellm.redis.stores.router_cooldowns.handle import HandleFactory, RouterCooldownsHandle
from litellm.redis.stores.router_cooldowns.types import ROUTER_COOLDOWNS_TARGET, ActiveCooldowns, CooldownCacheValue
from litellm.rust_bridge._native import NativeRedisClient, NativeRouterCooldowns

__all__ = [
    "ROUTER_COOLDOWNS_TARGET",
    "ActiveCooldowns",
    "CooldownCacheValue",
    "HandleFactory",
    "RouterCooldownStore",
    "RouterCooldowns",
    "RouterCooldownsHandle",
]


class RouterCooldowns(Protocol):
    def attach_redis(self, client: NativeRedisClient | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    def active(self, model_ids: Sequence[str], *, batch: Batch) -> Awaitable[ActiveCooldowns]: ...
    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float: ...
    def evict_local(self, model_id: str) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


async def _tuple(pending: Awaitable[ActiveCooldowns]) -> ActiveCooldowns:
    return tuple(await pending)


class RouterCooldownStore:
    def __init__(self, client: NativeRedisClient | None, *, handle: HandleFactory = NativeRouterCooldowns) -> None:
        self._handle: Final = handle(client)

    def attach_redis(self, client: NativeRedisClient | None) -> None:
        """Adds the Redis tier once, keeping this router's memory, as ``DualCache.attach_redis_cache`` does."""
        if client is not None:
            self._handle.attach(client)

    def flush_local_counters(self) -> None:
        self._handle.flush_local_counters()

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None:
        self._handle.put(model_id, value, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int:
        return self._handle.increment_allowed_fails(model_id, suffix, ttl)

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def active(self, model_ids: Sequence[str], *, batch: Batch) -> Awaitable[ActiveCooldowns]:
        """Declared now on the batch ``batch`` picks, so several reads can share one trip before any is awaited."""
        target: Final = route(batch)
        return _tuple(target.result(self._handle.active(model_ids, target.send)))

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns:
        return tuple(self._handle.active_sync(model_ids))

    @with_service_target(ROUTER_COOLDOWNS_TARGET)
    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float:
        return self._handle.minimum(model_ids) or default

    def evict_local(self, model_id: str) -> None:
        self._handle.delete_local(model_id)

    def snapshot(self) -> Mapping[str, CooldownCacheValue]:
        return self._handle.snapshot()

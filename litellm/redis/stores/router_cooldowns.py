from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import service_target
from litellm.caching.redis_cache import RedisCache
from litellm.rust_bridge._native import NativeRouterCooldowns

from ._connection import redis_kwargs


class CooldownCacheValue(TypedDict):
    exception_received: ReadOnly[str]
    status_code: ReadOnly[str]
    timestamp: ReadOnly[float]
    cooldown_time: ReadOnly[float]


ROUTER_COOLDOWNS_TARGET: Final = "router_cooldowns"
ActiveCooldowns: TypeAlias = Sequence[tuple[str, CooldownCacheValue]]


class RouterCooldowns(Protocol):
    def attach_redis(self, connection: RedisCache | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    async def active(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float: ...
    def evict_local(self, model_id: str) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


class CooldownsHandle(Protocol):
    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int: ...
    def active(self, model_ids: Sequence[str]) -> Sequence[tuple[str, CooldownCacheValue]]: ...
    def aactive(self, model_ids: Sequence[str]) -> Awaitable[Sequence[tuple[str, CooldownCacheValue]]]: ...
    def minimum(self, model_ids: Sequence[str]) -> float | None: ...
    def evict_local(self, model_id: str) -> None: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


HandleFactory: TypeAlias = Callable[[RedisCache | None], CooldownsHandle]


def native_handle(connection: RedisCache | None) -> CooldownsHandle:
    if connection is None:
        return NativeRouterCooldowns(None)
    return NativeRouterCooldowns(redis_kwargs(connection), namespace=connection.namespace)


class RouterCooldownStore:
    def __init__(self, connection: RedisCache | None, *, handle: HandleFactory = native_handle) -> None:
        self._connected = connection is not None
        self._build: Final = handle
        self._handle = handle(connection)

    def attach_redis(self, connection: RedisCache | None) -> None:
        if connection is None or self._connected:
            return
        self._connected = True
        self._handle = self._build(connection)

    def flush_local_counters(self) -> None:
        self._handle.flush_local_counters()

    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            self._handle.put(model_id, value, ttl)

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            return self._handle.increment_allowed_fails(model_id, suffix, ttl)

    async def active(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            pending: Final = self._handle.aactive(model_ids)
        return tuple(await pending)

    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            return tuple(self._handle.active(model_ids))

    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            return self._handle.minimum(model_ids) or default

    def evict_local(self, model_id: str) -> None:
        self._handle.evict_local(model_id)

    def snapshot(self) -> Mapping[str, CooldownCacheValue]:
        return self._handle.snapshot()

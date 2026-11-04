from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import service_target
from litellm.caching.redis_batch import NativeBatchResult, NativePending, RequestRedisBatches
from litellm.rust_bridge._native import NativeRedisClient, NativeRequestRedis, NativeRouterCooldowns


class CooldownCacheValue(TypedDict):
    exception_received: ReadOnly[str]
    status_code: ReadOnly[str]
    timestamp: ReadOnly[float]
    cooldown_time: ReadOnly[float]


ROUTER_COOLDOWNS_TARGET: Final = "router_cooldowns"
ActiveCooldowns: TypeAlias = Sequence[tuple[str, CooldownCacheValue]]


class RouterCooldowns(Protocol):
    def attach_redis(self, client: NativeRedisClient | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    async def active(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def declare_active(
        self, request: RequestRedisBatches, model_ids: Sequence[str]
    ) -> NativeBatchResult[ActiveCooldowns] | None: ...
    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float: ...
    def evict_local(self, model_id: str) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


class CooldownsHandle(Protocol):
    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int: ...
    def active(self, model_ids: Sequence[str]) -> Sequence[tuple[str, CooldownCacheValue]]: ...
    def aactive(self, model_ids: Sequence[str]) -> Awaitable[Sequence[tuple[str, CooldownCacheValue]]]: ...
    def declare_active(
        self, request: NativeRequestRedis, model_ids: Sequence[str]
    ) -> NativePending[ActiveCooldowns]: ...
    def minimum(self, model_ids: Sequence[str]) -> float | None: ...
    def evict_local(self, model_id: str) -> None: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> Mapping[str, CooldownCacheValue]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None], CooldownsHandle]


class RouterCooldownStore:
    def __init__(self, client: NativeRedisClient | None, *, handle: HandleFactory = NativeRouterCooldowns) -> None:
        self._connected = client is not None
        self._build: Final = handle
        self._handle = handle(client)

    def attach_redis(self, client: NativeRedisClient | None) -> None:
        if client is None or self._connected:
            return
        self._connected = True
        self._handle = self._build(client)

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

    def declare_active(
        self, request: RequestRedisBatches, model_ids: Sequence[str]
    ) -> NativeBatchResult[ActiveCooldowns] | None:
        """``active`` declared on the request's native pre-call batch; None when there is no Redis to read."""
        if not self._connected:
            return None
        with service_target(ROUTER_COOLDOWNS_TARGET):
            pending: Final = self._handle.declare_active(request.native(), model_ids)
        return request.native_result(pending)

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

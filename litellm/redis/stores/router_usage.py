from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import with_service_target
from litellm.rust_bridge._native import NativeRedisClient, NativeRouterUsage

ROUTER_USAGE_TARGET: Final = "router_usage"
CounterKind: TypeAlias = Literal["tpm", "rpm"]


@dataclass(frozen=True, slots=True)
class UsageCounter:
    deployment_id: str
    model: str | None
    minute: str
    kind: CounterKind


class RouterUsage(Protocol):
    def attach_redis(self, client: NativeRedisClient | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    async def async_increment(
        self, counter: UsageCounter, value: float, ttl: int, parent_otel_span: Span | None = None
    ) -> float | None: ...
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
        parent_otel_span: Span | None,
    ) -> None: ...
    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None: ...
    def snapshot(self) -> Mapping[str, float]: ...


NativeCounter: TypeAlias = tuple[str, str | None, str, CounterKind]


class UsageHandle(Protocol):
    def increment(self, counter: NativeCounter, value: float, ttl_seconds: int) -> Awaitable[float]: ...
    def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> Awaitable[None]: ...
    def get_many(self, counters: Sequence[NativeCounter]) -> Awaitable[Sequence[float | None]]: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> Mapping[str, float]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None], UsageHandle]


def _native(counter: UsageCounter) -> NativeCounter:
    return (counter.deployment_id, counter.model, counter.minute, counter.kind)


class RouterUsageStore:
    def __init__(self, client: NativeRedisClient | None, *, handle: HandleFactory = NativeRouterUsage) -> None:
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

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_increment(
        self, counter: UsageCounter, value: float, ttl: int, parent_otel_span: Span | None = None
    ) -> float | None:
        return await self._handle.increment(_native(counter), value, ttl)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
        parent_otel_span: Span | None,
    ) -> None:
        await self._handle.increment_deployment(deployment_id, model, minute, tokens, requests)

    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None:
        return await self._handle.get_many(tuple(map(_native, counters)))

    def snapshot(self) -> Mapping[str, float]:
        return self._handle.snapshot()

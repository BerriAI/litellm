from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, TypeAlias

from litellm.redis.stores.router_usage.types import NativeCounter
from litellm.rust_bridge._native import NativeRedisClient


class RouterUsageHandle(Protocol):
    def attach(self, client: NativeRedisClient, redis_ttl_seconds: int = 60, /) -> bool: ...
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


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None], RouterUsageHandle]

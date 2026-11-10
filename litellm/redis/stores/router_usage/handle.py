from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol

from litellm.redis.stores.router_usage.types import UsageCounter
from litellm.rust_bridge._native import NativeRedisClient


class RouterUsageHandle(Protocol):
    @property
    def has_redis(self) -> bool: ...
    def attach(self, client: NativeRedisClient, /) -> bool: ...
    def increment(self, counter: UsageCounter, value: float, ttl_seconds: int) -> Awaitable[float]: ...
    def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> Awaitable[None]: ...
    def get_many(self, counters: Sequence[UsageCounter]) -> Awaitable[Sequence[float | None]]: ...
    def increment_sync(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float: ...
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    def get_many_sync(self, counters: Sequence[UsageCounter]) -> Sequence[float | None]: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> Mapping[str, float]: ...

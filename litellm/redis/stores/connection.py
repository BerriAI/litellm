from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.rust_bridge._native import (
    NativeAuthCache,
    NativeDeploymentHealth,
    NativeRateLimits,
    NativeRedisClient,
    NativeRouterCooldowns,
    NativeRouterUsage,
    NativeSensitiveRoutePins,
    NativeSpendCounters,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.redis.stores.auth_cache.types import AuthCacheSettings


@dataclass(frozen=True, slots=True)
class RedisStores:
    """Builds every store family's handle on one shared client. Without a client the handles run memory-only."""

    client: NativeRedisClient | None

    def router_cooldowns(self) -> NativeRouterCooldowns:
        return NativeRouterCooldowns(self.client)

    def router_usage(self) -> NativeRouterUsage:
        return NativeRouterUsage(self.client)

    def spend_counters(self) -> NativeSpendCounters:
        return NativeSpendCounters(self.client)

    def rate_limits(self, backfill_ttl: float | None = None) -> NativeRateLimits:
        return NativeRateLimits(self.client, backfill_ttl)

    def auth_cache(self, settings: AuthCacheSettings) -> NativeAuthCache:
        return NativeAuthCache(
            self.client, memory_entries=settings.memory_entries, memory_ttl_seconds=settings.memory_ttl
        )

    def sensitive_route_pins(self) -> NativeSensitiveRoutePins:
        return NativeSensitiveRoutePins(self.client)

    def deployment_health(self) -> NativeDeploymentHealth:
        return NativeDeploymentHealth(self.client)


def connect(redis_cache: RedisCache | None) -> RedisStores:
    settings: Final = None if redis_cache is None else redis_cache.store_settings
    return RedisStores(None if settings is None else NativeRedisClient.connect(settings))

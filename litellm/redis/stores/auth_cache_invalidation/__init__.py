from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from litellm._internal_context import with_service_target
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import AUTH_OBJECTS_TARGET
from litellm.rust_bridge._native import NativeAuthCacheInvalidation

if TYPE_CHECKING:
    from litellm.redis.stores.auth_cache import AuthCacheStore
    from litellm.redis.stores.connection import RedisStores
    from litellm.redis.stores.spend_counters import SpendCounterStore

__all__ = ["AuthCacheInvalidation", "AuthCacheInvalidationHandle", "PythonCaches", "connect_invalidation"]


@dataclass(frozen=True, slots=True)
class PythonCaches:
    """This worker's caches the invalidation channel reaches that are not native yet. ``copies`` take a broadcast
    value as well as an eviction; ``evicted`` only drop the key."""

    copies: tuple[InMemoryCache, ...] = ()
    evicted: tuple[InMemoryCache, ...] = ()

    def apply(self, cache_key: str, new_value: float | None, ttl: float | None) -> None:
        if new_value is not None:
            for cache in self.copies:
                cache.set_cache(cache_key, new_value, ttl=ttl)
            return
        for cache in (*self.copies, *self.evicted):
            cache.delete_cache(cache_key)


class AuthCacheInvalidationHandle(Protocol):
    @property
    def channel(self) -> str: ...
    def publish(self, cache_key: str, new_value: float | None = None, ttl: float | None = None) -> bool: ...
    def broadcast_evictions(self, cache_keys: Sequence[str]) -> Awaitable[None]: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class AuthCacheInvalidation:
    """The ``litellm_proxy.auth_cache_invalidation`` channel every worker shares, all native: publishes go out in the
    background and never fail the caller, and the subscription applies each message to the native auth cache and
    spend counters, then to ``PythonCaches``."""

    def __init__(self, handle: AuthCacheInvalidationHandle) -> None:
        self._handle: Final = handle

    @property
    def channel(self) -> str:
        return self._handle.channel

    def publish(self, cache_key: str, new_value: float | None = None, ttl: float | None = None) -> bool:
        """A delete, or with ``new_value`` a SET that every worker, this one included, writes instead of dropping.
        False when it was dropped because too many publishes are already waiting on Redis."""
        return self._handle.publish(cache_key, new_value, ttl)

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def broadcast_evictions(self, cache_keys: Sequence[str]) -> None:
        """One pipelined round trip, waited for at most two seconds."""
        if cache_keys:
            await self._handle.broadcast_evictions(cache_keys)

    def start(self) -> None:
        self._handle.start()

    def stop(self) -> None:
        self._handle.stop()


def connect_invalidation(
    redis: RedisStores,
    namespace: str | None,
    auth_cache: AuthCacheStore,
    spend_counters: SpendCounterStore,
    python_caches: PythonCaches,
) -> AuthCacheInvalidation | None:
    """None without a Redis client. ``namespace`` is ``RedisCache.namespace``, which prefixes the channel; the native
    client's own namespace misses one assigned to the ``RedisCache`` after it was built. Both stores must be native,
    since Rust applies each message to them: one that is not raises, so a worker can never run with invalidation
    silently off."""
    if redis.client is None:
        return None
    return AuthCacheInvalidation(
        NativeAuthCacheInvalidation(redis.client, namespace, auth_cache.native, spend_counters.native, python_caches)
    )

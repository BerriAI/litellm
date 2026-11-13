from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch, active_request_redis_batches
from litellm.redis.stores._route import route
from litellm.redis.stores.auth_cache.handle import AuthCacheHandle
from litellm.redis.stores.auth_cache.types import AUTH_CACHE_TARGET, AuthCacheSettings, WriteBack
from litellm.rust_bridge._native import AuthObject, LocalSizes, NativeRequestRedis

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = [
    "AUTH_CACHE_TARGET",
    "AuthCacheSettings",
    "AuthCacheStore",
    "AuthObject",
    "WriteBack",
]


def _built_request() -> NativeRequestRedis | None:
    """The current request's native batches when a store op already built them; none built means none of its reads
    found a key absent yet."""
    request: Final = active_request_redis_batches()
    return None if request is None else request.existing_native()


def _redis_ttl(ttl: float | None) -> float | None:
    """``RedisCache.get_ttl``: a NaN TTL, which ``int()`` refuses, takes the default TTL."""
    return None if ttl is not None and math.isnan(ttl) else ttl


class AuthCacheStore:
    """The proxy's auth object cache: memory partitions, all native, in front of the Redis ``enable_redis_auth_cache``
    shares across workers once attached. Reads and writes name an ``AuthObject``; evictions carry wire keys. Values
    are plain data (what ``json.loads`` returns), so callers validate models themselves. Redis calls are reported
    under the caller's service target.

    Inside a request, a key a prefetch read found absent in Redis is answered as absent by ``get`` until something
    writes it, so per-key getters after a prefetch cost no round trip. The native request tracks those keys."""

    def __init__(self, handle: AuthCacheHandle) -> None:
        self._handle: Final = handle

    def attach_redis(self, redis: RedisStores) -> None:
        """Adds the Redis tier once, keeping memory, as ``DualCache.attach_redis_cache`` does."""
        if redis.client is not None:
            _ = self._handle.attach(redis.client)

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    def reconfigure(self, settings: AuthCacheSettings) -> None:
        """Both memory partitions keep what they hold, as ``update_cache_ttl`` and ``update_in_memory_max_size`` do."""
        self._handle.reconfigure(memory_entries=settings.memory_entries, memory_ttl_seconds=settings.memory_ttl)

    @with_service_target(AUTH_CACHE_TARGET)
    async def get(self, key: AuthObject) -> object:
        return await self._handle.get(key, _built_request())

    @with_service_target(AUTH_CACHE_TARGET)
    def get_sync(self, key: AuthObject) -> object:
        return self._handle.get_sync(key)

    @with_service_target(AUTH_CACHE_TARGET)
    async def set(self, key: AuthObject, value: object, ttl: float | None) -> None:
        await self._handle.set(key, value, _redis_ttl(ttl), _built_request())

    @with_service_target(AUTH_CACHE_TARGET)
    def set_sync(self, key: AuthObject, value: object, ttl: float | None) -> None:
        self._handle.set_sync(key, value, _redis_ttl(ttl), _built_request())

    @with_service_target(AUTH_CACHE_TARGET)
    def declare_set(self, key: AuthObject, value: object, ttl: float | None) -> Awaitable[None] | None:
        """Memory now and the SET on the request's pre-call batch, sent with its next flush; None outside a request,
        so the caller writes through."""
        target: Final = route(Batch.PRE_CALL)
        if target.request is None:
            return None
        return target.result(self._handle.declare_set(key, value, _redis_ttl(ttl), target.send))

    @with_service_target(AUTH_CACHE_TARGET)
    def declare_delete(self, key: str) -> Awaitable[None] | None:
        target: Final = route(Batch.PRE_CALL)
        if target.request is None:
            return None
        return target.result(self._handle.declare_delete(key, target.send))

    @with_service_target(AUTH_CACHE_TARGET)
    async def set_many(self, entries: Sequence[tuple[AuthObject, object]], ttl: float | None) -> None:
        await self._handle.set_many(entries, _redis_ttl(ttl), _built_request())

    @with_service_target(AUTH_CACHE_TARGET)
    async def write_back(self, entries: Sequence[WriteBack]) -> None:
        """Memory now; inside a request the SETs ride its pre-call batch and go out with the next flush, else in a
        round trip of their own. An entry without a TTL keeps ``default_in_memory_ttl`` in memory and the connection's
        default TTL in Redis."""
        await self._handle.write_back(
            tuple((key, value, _redis_ttl(ttl)) for key, value, ttl in entries), route(Batch.PRE_CALL).send
        )

    @with_service_target(AUTH_CACHE_TARGET)
    async def delete(self, keys: Sequence[str]) -> None:
        """Both memory partitions, then Redis; raises when Redis failed, with memory already dropped."""
        await self._handle.delete(keys)

    @with_service_target(AUTH_CACHE_TARGET)
    def delete_sync(self, keys: Sequence[str]) -> None:
        self._handle.delete_sync(keys)

    @with_service_target(AUTH_CACHE_TARGET)
    async def read_missing(self, keys: Sequence[AuthObject]) -> Mapping[AuthObject, object]:
        """One read of ``keys`` (already missing in memory), on the request's pre-call batch inside a request, which
        then notes the keys Redis lacks. Absent keys map to None; raises when Redis failed."""
        target: Final = route(Batch.PRE_CALL)
        pending: Final = self._handle.get_many(keys, target.send)
        if pending is None:
            return MappingProxyType({})
        return MappingProxyType(dict(zip(keys, await target.result(pending), strict=True)))

    @with_service_target(AUTH_CACHE_TARGET, preserve_existing=True)
    async def increment_if_present(self, key: AuthObject, amount: float) -> float | None:
        return await self._handle.increment_if_present(key, amount)

    def get_local(self, key: AuthObject) -> object:
        return self._handle.get_local(key)

    def set_local(self, key: AuthObject, value: object, ttl: float | None) -> None:
        self._handle.set_local(key, value, ttl)

    def delete_local(self, key: str) -> None:
        self._handle.delete_local(key)

    def flush_local(self) -> None:
        self._handle.flush_local()

    def len_local(self) -> LocalSizes:
        return self._handle.len_local()

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch, active_request_redis_batches
from litellm.redis.stores._route import route
from litellm.redis.stores.auth_cache.handle import AuthCacheHandle
from litellm.redis.stores.auth_cache.types import AUTH_CACHE_TARGET, AuthCacheSettings, WriteBack
from litellm.rust_bridge._native import LocalSizes

__all__ = [
    "AUTH_CACHE_TARGET",
    "AuthCacheSettings",
    "AuthCacheStore",
    "WriteBack",
]


def _redis_ttl(ttl: float | None) -> float | None:
    """``RedisCache.get_ttl``: a NaN TTL, which ``int()`` refuses, takes the default TTL."""
    return None if ttl is not None and math.isnan(ttl) else ttl


class AuthCacheStore:
    """The auth object cache ``enable_redis_auth_cache`` shares across workers: memory partitions in front of Redis,
    all native. Keys are ``UserApiKeyCache``'s logical cache keys; values are plain data (what ``json.loads``
    returns), so callers validate models themselves. Redis calls are reported under the caller's service target.

    Inside a request, a key a prefetch read found absent in Redis is answered as absent by ``get`` until something
    writes it, so per-key getters after a prefetch cost no round trip."""

    def __init__(self, handle: AuthCacheHandle) -> None:
        self._handle: Final = handle

    def reconfigure(self, settings: AuthCacheSettings) -> None:
        """Both memory partitions keep what they hold, as ``update_cache_ttl`` and ``update_in_memory_max_size`` do."""
        self._handle.reconfigure(memory_entries=settings.memory_entries, memory_ttl_seconds=settings.memory_ttl)

    @with_service_target(AUTH_CACHE_TARGET)
    async def get(self, key: str) -> object:
        request: Final = active_request_redis_batches()
        if request is not None and request.native_read_as_missing(key) and self._handle.get_local(key) is None:
            return None
        return await self._handle.get(key)

    @with_service_target(AUTH_CACHE_TARGET)
    def get_sync(self, key: str) -> object:
        return self._handle.get_sync(key)

    @with_service_target(AUTH_CACHE_TARGET)
    async def set(self, key: str, value: object, ttl: float | None) -> None:
        self._forget_miss(key)
        await self._handle.set(key, value, _redis_ttl(ttl))

    @with_service_target(AUTH_CACHE_TARGET)
    def set_sync(self, key: str, value: object, ttl: float | None) -> None:
        self._forget_miss(key)
        self._handle.set_sync(key, value, _redis_ttl(ttl))

    @with_service_target(AUTH_CACHE_TARGET)
    async def lease(self, key: str) -> str | None:
        """The generation of ``key`` every worker shares, read on a cache miss before the database read that will
        fill it. A delete of the key on any worker starts a new one."""
        return await self._handle.lease(key)

    @with_service_target(AUTH_CACHE_TARGET)
    async def set_leased(self, key: str, lease: str | None, value: object, ttl: float | None) -> bool:
        """The Redis SET alone, refused with False when ``key`` was deleted on any worker since ``lease`` was read.
        Memory is the caller's to write."""
        self._forget_miss(key)
        return await self._handle.set_leased(key, lease, value, _redis_ttl(ttl))

    @with_service_target(AUTH_CACHE_TARGET)
    def declare_set(self, key: str, value: object, ttl: float | None) -> Awaitable[None] | None:
        """Memory now and the SET on the request's pre-call batch, sent with its next flush; None outside a request,
        so the caller writes through."""
        target: Final = route(Batch.PRE_CALL)
        if target.request is None:
            return None
        pending: Final = self._handle.declare_set(key, value, _redis_ttl(ttl), target.send)
        target.request.forget_native_miss(key)
        return target.result(pending)

    @with_service_target(AUTH_CACHE_TARGET)
    def declare_delete(self, key: str) -> Awaitable[None] | None:
        target: Final = route(Batch.PRE_CALL)
        if target.request is None:
            return None
        pending: Final = self._handle.declare_delete(key, target.send)
        target.request.note_native_misses((key,))
        return target.result(pending)

    @with_service_target(AUTH_CACHE_TARGET)
    async def set_many(self, entries: Sequence[tuple[str, object]], ttl: float | None) -> None:
        for key, _value in entries:
            self._forget_miss(key)
        await self._handle.set_many(entries, _redis_ttl(ttl))

    @with_service_target(AUTH_CACHE_TARGET)
    async def write_back(self, entries: Sequence[WriteBack], *, batch: Batch) -> None:
        """Memory now; the SETs ride the batch ``batch`` picks, so a pre-call write inside a request goes out with the
        next flush, else in a round trip of its own. An entry without a TTL keeps ``default_in_memory_ttl`` in memory and the
        connection's default TTL in Redis."""
        for key, _value, _ttl in entries:
            self._forget_miss(key)
        await self._handle.write_back(
            tuple((key, value, _redis_ttl(ttl)) for key, value, ttl in entries), route(batch).send
        )

    @with_service_target(AUTH_CACHE_TARGET)
    async def delete(self, keys: Sequence[str]) -> None:
        """Memory, then Redis in one round trip; raises when Redis failed, with memory already dropped."""
        await self._handle.delete(keys)

    @with_service_target(AUTH_CACHE_TARGET)
    def delete_sync(self, keys: Sequence[str]) -> None:
        self._handle.delete_sync(keys)

    @with_service_target(AUTH_CACHE_TARGET)
    async def read_missing(self, keys: Sequence[str], *, batch: Batch) -> Mapping[str, object]:
        """One read of ``keys`` (already missing in memory) on the batch ``batch`` picks, noting the keys Redis lacks
        when it rode a request. Absent keys map to None; raises when Redis failed."""
        target: Final = route(batch)
        pending: Final = self._handle.get_many(keys, target.send)
        if pending is None:
            return MappingProxyType({})
        found: Final = MappingProxyType(dict(zip(keys, await target.result(pending), strict=True)))
        if target.request is not None:
            target.request.note_native_misses(tuple(key for key, value in found.items() if value is None))
        return found

    @with_service_target(AUTH_CACHE_TARGET, preserve_existing=True)
    async def increment_if_present(self, key: str, amount: float) -> float | None:
        return await self._handle.increment_if_present(key, amount)

    def get_local(self, key: str) -> object:
        return self._handle.get_local(key)

    def set_local(self, key: str, value: object, ttl: float | None) -> None:
        self._handle.set_local(key, value, ttl)

    def delete_local(self, key: str) -> None:
        self._handle.delete_local(key)

    def flush_local(self) -> None:
        self._handle.flush_local()

    def len_local(self) -> LocalSizes:
        return self._handle.len_local()

    @staticmethod
    def _forget_miss(key: str) -> None:
        request: Final = active_request_redis_batches()
        if request is not None:
            request.forget_native_miss(key)

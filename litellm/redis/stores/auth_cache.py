from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from litellm.caching.redis_batch import (
    NativeBatchResult,
    NativePending,
    active_request_redis_batches,
)
from litellm.rust_bridge._native import NativeAuthCache, NativeRedisClient, NativeRequestRedis

WriteBack: TypeAlias = tuple[str, object, float | None]


@dataclass(frozen=True, slots=True)
class AuthCacheSettings:
    """``UserApiKeyCache``'s memory configuration: each partition's size and ``default_in_memory_ttl``."""

    memory_entries: int
    memory_ttl: float | None


class AuthCacheHandle(Protocol):
    def get(self, key: str) -> Awaitable[object]: ...
    def get_sync(self, key: str) -> object: ...
    def set(self, key: str, value: object, ttl_seconds: float | None = None) -> Awaitable[None]: ...
    def set_sync(self, key: str, value: object, ttl_seconds: float | None = None) -> None: ...
    def set_pre_call(
        self, request: NativeRequestRedis, key: str, value: object, ttl_seconds: float | None = None
    ) -> NativePending[None] | None: ...
    def delete_pre_call(self, request: NativeRequestRedis, key: str) -> NativePending[None] | None: ...
    def set_many(self, entries: Sequence[tuple[str, object]], ttl_seconds: float | None = None) -> Awaitable[None]: ...
    def write_back(self, request: NativeRequestRedis | None, entries: Sequence[WriteBack]) -> Awaitable[None]: ...
    def delete(self, keys: Sequence[str]) -> Awaitable[None]: ...
    def delete_sync(self, keys: Sequence[str]) -> None: ...
    def declare_get_many(
        self, request: NativeRequestRedis, keys: Sequence[str]
    ) -> NativePending[Sequence[object]] | None: ...
    def get_many(self, keys: Sequence[str]) -> Awaitable[Sequence[object] | None]: ...
    def increment_if_present(self, key: str, amount: float) -> Awaitable[float | None]: ...
    def get_local(self, key: str) -> object: ...
    def set_local(self, key: str, value: object, ttl_seconds: float | None = None) -> None: ...
    def delete_local(self, key: str) -> None: ...
    def flush_local(self) -> None: ...
    def len_local(self) -> tuple[int, int]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None, AuthCacheSettings, int], AuthCacheHandle]


def native_handle(client: NativeRedisClient | None, settings: AuthCacheSettings, redis_ttl: int) -> AuthCacheHandle:
    return NativeAuthCache(
        client,
        memory_entries=settings.memory_entries,
        memory_ttl_seconds=settings.memory_ttl,
        redis_ttl_seconds=redis_ttl,
    )


class AuthCacheStore:
    """The auth object cache ``enable_redis_auth_cache`` shares across workers: memory partitions in front of Redis,
    all native. Keys are ``UserApiKeyCache``'s logical cache keys; values are plain data (what ``json.loads``
    returns), so callers validate models themselves. Redis calls are reported under the caller's service target.

    Inside a request, a key a prefetch read found absent in Redis is answered as absent by ``get`` until something
    writes it, so per-key getters after a prefetch cost no round trip."""

    def __init__(
        self,
        client: NativeRedisClient | None,
        settings: AuthCacheSettings,
        *,
        redis_ttl: int,
        handle: HandleFactory = native_handle,
    ) -> None:
        self._client: Final = client
        self._redis_ttl: Final = redis_ttl
        self._build: Final = handle
        self._settings = settings
        self._handle = handle(client, settings, redis_ttl)

    @property
    def settings(self) -> AuthCacheSettings:
        return self._settings

    def reconfigure(self, settings: AuthCacheSettings) -> None:
        """A changed memory configuration starts from an empty memory tier, as a resized partition would."""
        if settings == self._settings:
            return
        self._settings = settings
        self._handle = self._build(self._client, settings, self._redis_ttl)

    async def get(self, key: str) -> object:
        request: Final = active_request_redis_batches()
        if request is not None and request.native_read_as_missing(key) and self._handle.get_local(key) is None:
            return None
        return await self._handle.get(key)

    def get_sync(self, key: str) -> object:
        return self._handle.get_sync(key)

    async def set(self, key: str, value: object, ttl: float | None) -> None:
        self._forget_miss(key)
        await self._handle.set(key, value, ttl)

    def set_sync(self, key: str, value: object, ttl: float | None) -> None:
        self._forget_miss(key)
        self._handle.set_sync(key, value, ttl)

    def set_pre_call(self, key: str, value: object, ttl: float | None) -> NativeBatchResult[None] | None:
        """Memory now and the SET on the request's native pre-call batch, sent with its next flush; None outside a
        request, so the caller writes through."""
        request: Final = active_request_redis_batches()
        if request is None:
            return None
        pending: Final = self._handle.set_pre_call(request.native(), key, value, ttl)
        if pending is None:
            return None
        request.forget_native_miss(key)
        return request.native_result(pending)

    def delete_pre_call(self, key: str) -> NativeBatchResult[None] | None:
        request: Final = active_request_redis_batches()
        if request is None:
            return None
        pending: Final = self._handle.delete_pre_call(request.native(), key)
        if pending is None:
            return None
        request.note_native_misses((key,))
        return request.native_result(pending)

    async def set_many(self, entries: Sequence[tuple[str, object]], ttl: float | None) -> None:
        for key, _value in entries:
            self._forget_miss(key)
        await self._handle.set_many(entries, ttl)

    async def write_back(self, entries: Sequence[WriteBack]) -> None:
        """Memory now; inside a request the SETs ride its native pre-call batch, sent with the next flush, else they
        go out in a round trip of their own. An entry without a TTL keeps ``default_in_memory_ttl`` in memory and the
        connection's default TTL in Redis."""
        request: Final = active_request_redis_batches()
        for key, _value, _ttl in entries:
            self._forget_miss(key)
        await self._handle.write_back(None if request is None else request.native(), entries)

    async def delete(self, keys: Sequence[str]) -> None:
        """Memory, then Redis in one round trip; raises when Redis failed, with memory already dropped."""
        await self._handle.delete(keys)

    def delete_sync(self, keys: Sequence[str]) -> None:
        self._handle.delete_sync(keys)

    async def read_missing(self, keys: Sequence[str]) -> dict[str, object]:
        """One read of ``keys`` (already missing in memory): on the request's native pre-call batch inside a request,
        noting the keys Redis lacks, else a round trip of its own. Absent keys map to None; raises when Redis failed."""
        request: Final = active_request_redis_batches()
        if request is None:
            values: Final = await self._handle.get_many(keys)
            return {} if values is None else dict(zip(keys, values, strict=True))
        pending: Final = self._handle.declare_get_many(request.native(), keys)
        if pending is None:
            return {}
        found: Final = dict(zip(keys, await request.native_result(pending), strict=True))
        request.note_native_misses(tuple(key for key, value in found.items() if value is None))
        return found

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

    def len_local(self) -> tuple[int, int]:
        """(management objects, user-key objects) held in memory."""
        return self._handle.len_local()

    @staticmethod
    def _forget_miss(key: str) -> None:
        request: Final = active_request_redis_batches()
        if request is not None:
            request.forget_native_miss(key)

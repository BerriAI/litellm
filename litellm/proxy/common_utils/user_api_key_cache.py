from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, TypeVar, overload
from weakref import WeakValueDictionary

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import DEFAULT_MAX_SIZE_IN_MEMORY
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec, CacheValue
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject
from litellm.redis.stores.auth_cache.handle import TeamRoster
from litellm.redis.stores.connection import connect
from litellm.rust_bridge._native import NativeRedisStoreError

T = TypeVar("T", bound=BaseModel)

from litellm.constants import AUTH_OBJECTS_TARGET as AUTH_OBJECTS_TARGET  # noqa: E402, PLC0414  # public re-export

_CACHED_VALUE: Final = TypeAdapter(object)

AuthCacheFactory: TypeAlias = Callable[[AuthCacheSettings], AuthCacheStore]


def _native_auth_cache(settings: AuthCacheSettings) -> AuthCacheStore:
    return AuthCacheStore(connect(None).auth_cache(settings))


@dataclass(frozen=True, slots=True)
class _DecodedModel:
    payload: object
    model_type: type[BaseModel]
    model: BaseModel


class CacheLease:
    __slots__ = ("__weakref__", "key", "shared")

    def __init__(self, key: AuthObject, shared: str | None) -> None:
        self.key: Final = key
        self.shared: Final = shared


class UserApiKeyCache:
    """
    The proxy's auth objects (keys, users, teams, memberships, orgs, ...), always on the native ``AuthCacheStore``:
    memory only until ``enable_redis_auth_cache`` attaches the Redis every worker shares. Reads and writes name an
    ``AuthObject``; evictions carry wire keys, since a broadcast carries nothing else.

    Values are stored as a Redis-safe JSON payload so memory and Redis never drift in format. With ``model_type``,
    writes serialize through ``CacheCodec`` and reads return the model, or None on a miss or when the cached payload no
    longer validates (logged). Without it, the raw payload comes back. A read the store failed is a miss and a failed
    write is logged, as ``DualCache`` does. A read with a key that is not an ``AuthObject`` raises.
    """

    def __init__(
        self,
        default_in_memory_ttl: float | None = None,
        max_size_in_memory: int = DEFAULT_MAX_SIZE_IN_MEMORY,
        auth_cache: AuthCacheFactory = _native_auth_cache,
    ) -> None:
        self.default_in_memory_ttl: float | None = default_in_memory_ttl
        self.max_size_in_memory: int = max_size_in_memory
        self._store: Final = auth_cache(self._settings())
        self._redis_connection: RedisCache | None = None
        self._leases: Final[WeakValueDictionary[str, CacheLease]] = WeakValueDictionary()
        self._decoded_models: Final[dict[str, _DecodedModel]] = {}  # mutable-ok: per-key memo of validated reads

    @property
    def store(self) -> AuthCacheStore:
        return self._store

    @property
    def has_redis(self) -> bool:
        return self._store.has_redis

    @property
    def redis_connection(self) -> RedisCache | None:
        """The Redis the auth cache was attached to, for features that share it under keys of their own."""
        return self._redis_connection

    def _settings(self) -> AuthCacheSettings:
        return AuthCacheSettings(memory_entries=self.max_size_in_memory, memory_ttl=self.default_in_memory_ttl)

    def update_cache_ttl(self, default_in_memory_ttl: float | None) -> None:
        """Memory entries keep what they hold. A write without a TTL takes this one in Redis too."""
        self.default_in_memory_ttl = default_in_memory_ttl
        self._store.reconfigure(self._settings())

    def update_in_memory_max_size(self, max_size: int | None) -> None:
        """None restores the default size, as ``DualCache.update_in_memory_max_size`` does."""
        self.max_size_in_memory = DEFAULT_MAX_SIZE_IN_MEMORY if max_size is None else max_size
        self._store.reconfigure(self._settings())

    def attach_redis_cache(self, redis_cache: RedisCache) -> None:
        if self._redis_connection is not None:
            return
        self._redis_connection = redis_cache
        self._store.attach_redis(connect(redis_cache))

    def get_local(self, key: AuthObject) -> object:
        """This worker's memory copy alone."""
        return self._store.get_local(key)

    def has_local_team(self, key: AuthObject) -> bool:
        return self._store.has_local_team(key)

    async def get_team_fields(
        self, key: AuthObject, fields: Sequence[str] | None = None, *, user_id: str | None = None
    ) -> object:
        try:
            return await self._store.team_fields(key, fields, user_id)
        except NativeRedisStoreError as e:
            verbose_proxy_logger.error("UserApiKeyCache.get_team_fields failed for key=%r: %s", key.key, e)
            return None

    async def get_auth_team(self, key: AuthObject, user_id: str | None) -> LiteLLM_TeamTableCachedObj | None:
        cached: Final = await self.get_team_fields(key, user_id=user_id)
        if cached is None:
            return None
        decoded: Final = _validated(cached, key, LiteLLM_TeamTableCachedObj, "get_auth_team")
        return decoded if isinstance(decoded, LiteLLM_TeamTableCachedObj) else None

    async def get_team_snapshot(self, key: AuthObject) -> tuple[LiteLLM_TeamTableCachedObj, TeamRoster | None] | None:
        try:
            snapshot: Final = await self._store.team_snapshot(key)
        except NativeRedisStoreError as e:
            verbose_proxy_logger.error("UserApiKeyCache.get_team_snapshot failed for key=%r: %s", key.key, e)
            return None
        if snapshot is None:
            return None
        cached, roster = snapshot
        decoded: Final = _validated(cached, key, LiteLLM_TeamTableCachedObj, "get_team_snapshot")
        return (decoded, roster) if isinstance(decoded, LiteLLM_TeamTableCachedObj) else None

    def add_local_team_spend(self, key: AuthObject, delta: float) -> None:
        _ = self._store.add_local_team_spend(key, delta)

    def get_local_by_key(self, key: str) -> object:
        """This worker's memory copy under a wire key, for a hook that names no object."""
        return self._store.get_local_by_key(key)

    def set_local(self, key: AuthObject, value: object, ttl: float | None) -> None:
        """A memory write alone, with ``default_in_memory_ttl`` when ``ttl`` is None."""
        self._store.set_local(key, value, ttl)

    async def lease(self, key: AuthObject) -> CacheLease:
        """Taken on a cache miss, before the database read that will fill ``key``, and handed to
        ``async_set_leased``. Evicting the key voids it: the rows read under it may predate the write behind
        that eviction. With ``enable_redis_auth_cache`` an eviction on any worker voids it. Loads of one key
        on this worker share a lease until an eviction, and it is dropped once none holds it."""
        held: Final = self._leases.get(key.key)
        if held is not None:
            return held
        shared: Final = await self._store.lease(key)
        return self._leases.setdefault(key.key, CacheLease(key, shared))

    async def async_set_leased(self, lease: CacheLease, value: CacheValue, ttl: float | None) -> bool:
        """Caches ``value`` unless the lease's key was evicted since it was taken; False when refused."""
        wire_key: Final = lease.key.key
        if self._leases.get(wire_key) is not lease:
            return False
        payload: Final[object] = CacheCodec.serialize(value, model_type=None)
        written: Final = await self._store.set_leased(lease.key, lease.shared, payload, ttl)
        if self._leases.get(wire_key) is not lease:
            return False
        if not written:
            self._void_leases((wire_key,))
            return False
        self._store.set_local(lease.key, payload, ttl)
        return True

    def _decoded(self, cached: object, key: AuthObject, model_type: type[BaseModel] | None, method: str) -> object:
        """Without Redis, a read of a payload equal to the one a previous read validated returns a shallow copy of
        that model instead of validating again, so a caller that sets a field cannot change what the next reader
        gets. With Redis every read validates, since another worker may have written the entry."""
        if model_type is None or cached is None:
            return cached
        if self._store.has_redis:
            return _validated(cached, key, model_type, method)
        memo: Final = self._decoded_models.get(key.key)
        if memo is not None and memo.model_type is model_type and memo.payload == cached:
            return memo.model.model_copy()
        decoded: Final = _validated(cached, key, model_type, method)
        if decoded is None:
            return None
        _ = self._decoded_models.pop(key.key, None)
        self._decoded_models[key.key] = _DecodedModel(payload=cached, model_type=model_type, model=decoded)
        if len(self._decoded_models) > 2 * self.max_size_in_memory:
            _ = self._decoded_models.pop(next(iter(self._decoded_models)))
        return decoded.model_copy()

    def void_lease(self, key: str) -> None:
        """Another worker evicted ``key``; the store already dropped this worker's copy."""
        self._void_leases((key,))

    def _void_leases(self, keys: Sequence[str]) -> None:
        for key in keys:
            _ = self._leases.pop(key, None)
            _ = self._decoded_models.pop(key, None)

    def delete_local(self, key: str) -> None:
        """Drops this worker's memory copy alone, as a cross-worker eviction broadcast does."""
        self._void_leases((key,))
        self._store.delete_local(key)

    @overload
    def get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: type[T]) -> T | None: ...

    @overload
    def get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: None = None) -> Any: ...

    def get_cache(
        self, key: AuthObject, *, local_only: bool = False, model_type: type[BaseModel] | None = None
    ) -> object:
        try:
            cached: Final = self._store.get_local(key) if local_only else self._store.get_sync(key)
        except NativeRedisStoreError as e:
            verbose_proxy_logger.error("UserApiKeyCache.get_cache failed for key=%r: %s", key.key, e)
            return None
        return self._decoded(_CACHED_VALUE.validate_python(cached), key, model_type, "get_cache")

    @overload
    async def async_get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: type[T]) -> T | None: ...

    @overload
    async def async_get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: None = None) -> Any: ...

    async def async_get_cache(
        self, key: AuthObject, *, local_only: bool = False, model_type: type[BaseModel] | None = None
    ) -> object:
        store: Final = self._store
        try:
            cached: Final = store.get_local(key) if local_only or not store.has_redis else await store.get(key)
        except NativeRedisStoreError as e:
            verbose_proxy_logger.error("UserApiKeyCache.async_get_cache failed for key=%r: %s", key.key, e)
            return None
        return self._decoded(_CACHED_VALUE.validate_python(cached), key, model_type, "async_get_cache")

    def set_cache(
        self,
        key: AuthObject,
        value: CacheValue,
        *,
        ttl: float | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
    ) -> None:
        payload: Final = CacheCodec.serialize(value, model_type=model_type)
        try:
            if local_only:
                self._store.set_local(key, payload, ttl)
            else:
                self._store.set_sync(key, payload, ttl)
        except Exception as e:  # noqa: BLE001  # DualCache.set_cache swallows a failed write
            verbose_proxy_logger.error("UserApiKeyCache.set_cache failed for key=%r: %s", key.key, e)

    async def async_set_cache(
        self,
        key: AuthObject,
        value: CacheValue,
        *,
        ttl: float | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
    ) -> None:
        """Inside a request the Redis SET rides the request's pipeline (memory is written at once); anywhere else it
        goes to Redis directly."""
        payload: Final = CacheCodec.serialize(value, model_type=model_type)
        try:
            if local_only:
                self._store.set_local(key, payload, ttl)
            elif self._store.declare_set(key, payload, ttl) is None:
                await self._store.set(key, payload, ttl)
        except Exception as e:  # noqa: BLE001  # DualCache.async_set_cache swallows a failed write
            verbose_proxy_logger.error("UserApiKeyCache.async_set_cache failed for key=%r: %s", key.key, e)

    async def async_set_cache_pipeline(
        self, cache_list: Sequence[tuple[AuthObject, CacheValue]], *, ttl: float | None = None
    ) -> None:
        """Batch writes with the same codec boundary as ``async_set_cache`` without ``model_type``."""
        entries: Final = tuple((key, CacheCodec.serialize(value, model_type=None)) for key, value in cache_list)
        try:
            await self._store.set_many(entries, ttl)
        except Exception as e:  # noqa: BLE001  # DualCache.async_set_cache_pipeline swallows a failed write
            verbose_proxy_logger.error("UserApiKeyCache.async_set_cache_pipeline failed: %s", e)

    def delete_cache(self, key: str) -> None:
        self._void_leases((key,))
        self._store.delete_sync((key,))

    async def async_delete_cache(self, key: str) -> None:
        self._void_leases((key,))
        await self._store.delete((key,))

    def declare_delete(self, key: str) -> Awaitable[None] | None:
        """The DEL on the request's pre-call batch; None outside a request, so the caller deletes."""
        self._void_leases((key,))
        return self._store.declare_delete(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        """Both partitions drop every key, then Redis; raises when Redis failed, with memory already dropped."""
        self._void_leases(keys)
        if keys:
            await self._store.delete(keys)

    def flush_cache(self) -> None:
        self._leases.clear()
        self._decoded_models.clear()
        self._store.flush_local()

    async def increment_if_present(self, key: AuthObject, amount: float) -> float | None:
        """``INCRBYFLOAT`` of an entry Redis still holds; None without Redis or when Redis lacks the entry. Raises
        when Redis failed."""
        return await self._store.increment_if_present(key, amount)


def _validated(cached: object, key: AuthObject, model_type: type[BaseModel], method: str) -> BaseModel | None:
    decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
    if decoded is None:
        verbose_proxy_logger.error(
            "UserApiKeyCache.%s failed to deserialize cached value for key=%r model_type=%s",
            method,
            key.key,
            getattr(model_type, "__name__", str(model_type)),
        )
    return decoded


#: Cached under ``AuthObject.user_object_permission`` when the user links no permission row, so a human without an
#: entitlement costs no DB read per request.
USER_NO_MCP_PERMISSION_SENTINEL: Final = "__user_no_mcp_permission__"

#: Cached under ``AuthObject.tag_registry`` when the table exceeds ``TAG_REGISTRY_MAX_SIZE``: registry unusable, fall
#: back to the per-tag lookup.
TAG_REGISTRY_OVERFLOW_SENTINEL: Final = "__tag_registry_overflow__"

#: Cached under ``AuthObject.model_access_group_registry`` when the table exceeds
#: ``MODEL_ACCESS_GROUP_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-group lookup.
MODEL_ACCESS_GROUP_REGISTRY_OVERFLOW_SENTINEL: Final = "__model_access_group_registry_overflow__"

#: Cached under ``AuthObject.end_user_restricted_registry`` when the restricted set exceeds
#: ``END_USER_RESTRICTED_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-id fetch.
END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL: Final = "__end_user_restricted_registry_overflow__"

#: Cached under ``AuthObject.team_membership_reservation`` when a member has no ``LiteLLM_TeamMembership`` row, so a
#: session-token member without a per-member budget costs no DB read per request. The budget readers treat any
#: non-model value as "no row".
NO_TEAM_MEMBERSHIP_SENTINEL: Final = "__no_team_membership__"


def get_management_object_ttl(cache: UserApiKeyCache) -> float:
    """
    In-memory TTL for management-object cache writes (keys, teams, users, budgets, ...).

    Honors ``general_settings.user_api_key_cache_ttl``, which ``proxy_server`` propagates onto
    ``default_in_memory_ttl`` at startup, and falls back to ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL``.
    """
    if cache.default_in_memory_ttl is not None:
        return cache.default_in_memory_ttl
    return DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

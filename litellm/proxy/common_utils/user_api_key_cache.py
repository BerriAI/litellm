from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, TypeAlias, TypeVar, overload

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import DEFAULT_MAX_SIZE_IN_MEMORY
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec, CacheValue
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject
from litellm.redis.stores.connection import connect
from litellm.rust_bridge._native import NativeRedisStoreError

T = TypeVar("T", bound=BaseModel)

from litellm.constants import AUTH_OBJECTS_TARGET as AUTH_OBJECTS_TARGET  # noqa: E402, PLC0414  # public re-export

_CACHED_VALUE: Final = TypeAdapter(object)

AuthCacheFactory: TypeAlias = Callable[[AuthCacheSettings], AuthCacheStore]


def _native_auth_cache(settings: AuthCacheSettings) -> AuthCacheStore:
    return AuthCacheStore(connect(None).auth_cache(settings))


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

    def set_local(self, key: AuthObject, value: object, ttl: float | None) -> None:
        """A memory write alone, with ``default_in_memory_ttl`` when ``ttl`` is None."""
        self._store.set_local(key, value, ttl)

    def delete_local(self, key: str) -> None:
        """Drops this worker's memory copy alone, as a cross-worker eviction broadcast does."""
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
        return _decoded(_CACHED_VALUE.validate_python(cached), key, model_type, "get_cache")

    @overload
    async def async_get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: type[T]) -> T | None: ...

    @overload
    async def async_get_cache(self, key: AuthObject, *, local_only: bool = False, model_type: None = None) -> Any: ...

    async def async_get_cache(
        self, key: AuthObject, *, local_only: bool = False, model_type: type[BaseModel] | None = None
    ) -> object:
        try:
            cached: Final = self._store.get_local(key) if local_only else await self._store.get(key)
        except NativeRedisStoreError as e:
            verbose_proxy_logger.error("UserApiKeyCache.async_get_cache failed for key=%r: %s", key.key, e)
            return None
        return _decoded(_CACHED_VALUE.validate_python(cached), key, model_type, "async_get_cache")

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
        self._store.delete_sync((key,))

    async def async_delete_cache(self, key: str) -> None:
        await self._store.delete((key,))

    def declare_delete(self, key: str) -> Awaitable[None] | None:
        """The DEL on the request's pre-call batch; None outside a request, so the caller deletes."""
        return self._store.declare_delete(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        """Both partitions drop every key, then Redis; raises when Redis failed, with memory already dropped."""
        if keys:
            await self._store.delete(keys)

    def flush_cache(self) -> None:
        self._store.flush_local()

    async def increment_if_present(self, key: AuthObject, amount: float) -> float | None:
        """``INCRBYFLOAT`` of an entry Redis still holds; None without Redis or when Redis lacks the entry. Raises
        when Redis failed."""
        return await self._store.increment_if_present(key, amount)


def _decoded(cached: object, key: AuthObject, model_type: type[BaseModel] | None, method: str) -> object:
    if model_type is None or cached is None:
        return cached
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

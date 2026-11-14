from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypeVar, cast, overload
from weakref import WeakValueDictionary

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec, CacheValue
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore
from litellm.redis.stores.connection import connect

if TYPE_CHECKING:
    from opentelemetry.trace import Span

T = TypeVar("T", bound=BaseModel)

from litellm.constants import AUTH_OBJECTS_TARGET as AUTH_OBJECTS_TARGET  # noqa: E402, PLC0414  # public re-export

_HASHED_TOKEN_CACHE_KEY: Final = re.compile(r"[0-9a-f]{64}")
_CACHED_VALUE: Final = TypeAdapter(object)
_BUDGET_WINDOWS_ABSENT_PREFIX: Final = "budget_windows_absent:"

AuthCacheFactory: TypeAlias = Callable[[RedisCache, AuthCacheSettings], AuthCacheStore]


def _connected_auth_cache(redis_cache: RedisCache, settings: AuthCacheSettings) -> AuthCacheStore:
    return AuthCacheStore(connect(redis_cache).auth_cache(settings))


def is_user_key_cache_key(key: str) -> bool:
    """Only user-key objects are cached under a bare ``hash_token`` digest; every other object uses a prefixed key."""
    return _HASHED_TOKEN_CACHE_KEY.fullmatch(key) is not None


_PIPELINED_SET_OPTIONS: Final = frozenset(("ttl",))
_TTL: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


def _ttl(kwargs: Mapping[str, object]) -> float | None:
    return _TTL.validate_python(kwargs.get("ttl"))


class CacheLease:
    __slots__ = ("__weakref__", "key", "shared")

    def __init__(self, key: str, shared: str | None) -> None:
        self.key: Final = key
        self.shared: Final = shared


class UserApiKeyCache(DualCache):
    """
    DualCache for UserAPIKeyAuth-like payloads.

    Stores a Redis-safe JSON payload in memory and, with ``enable_redis_auth_cache``, in Redis, to avoid
    "memory returns BaseModel, Redis returns dict" format drift.

    When ``model_type`` is provided:
    - writes are serialized via ``CacheCodec.serialize(..., model_type=...)``
    - reads are deserialized via ``CacheCodec.deserialize(..., model_type)``
      and return ``Optional[T]``: the model on success, ``None`` on cache miss
      **or** if the cached payload fails validation (schema drift). On
      validation failure after a cache hit, an error line is emitted via
      ``verbose_proxy_logger``.

    When ``model_type`` is omitted, the interface behaves like ``DualCache``:
    raw cached payload is returned (dict/str/etc.).

    User-key objects (see ``is_user_key_cache_key``) live in their own in-memory partition,
    ``key_object_cache``, so churn in the other management objects cannot evict them.

    Without Redis both partitions are this class's Python memory. ``attach_redis_cache`` moves the whole
    keyspace, both memory partitions included, onto the native ``AuthCacheStore`` (``shared``); redis-py never
    reads or writes it, and ``redis_cache`` stays None.

    ``get_cache`` / ``async_get_cache`` overloads and implementations must be contiguous
    (no other methods in between) so mypy resolves ``@overload`` + implementation correctly.
    """

    def __init__(
        self,
        in_memory_cache: InMemoryCache | None = None,
        redis_cache: RedisCache | None = None,
        default_in_memory_ttl: float | None = None,
        default_redis_ttl: float | None = None,
        key_object_in_memory_cache: InMemoryCache | None = None,
        budget_windows_absent_in_memory_cache: InMemoryCache | None = None,
        auth_cache: AuthCacheFactory = _connected_auth_cache,
    ) -> None:
        super().__init__(
            in_memory_cache=in_memory_cache,
            default_in_memory_ttl=default_in_memory_ttl,
            default_redis_ttl=default_redis_ttl,
        )
        self.key_object_cache: Final = DualCache(
            in_memory_cache=key_object_in_memory_cache or InMemoryCache(),
            default_in_memory_ttl=default_in_memory_ttl,
            default_redis_ttl=default_redis_ttl,
        )
        self.budget_windows_absent_cache: Final = DualCache(
            in_memory_cache=budget_windows_absent_in_memory_cache
            or InMemoryCache(default_ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL),
            default_in_memory_ttl=default_in_memory_ttl,
        )
        self._auth_cache: Final = auth_cache
        self._redis_connection: RedisCache | None = None
        self._shared: AuthCacheStore | None = None
        self._leases: Final[WeakValueDictionary[str, CacheLease]] = WeakValueDictionary()
        self.attach_redis_cache(redis_cache, default_redis_ttl=default_redis_ttl)

    @property
    def shared(self) -> AuthCacheStore | None:
        """The shared store holding the keyspace once ``enable_redis_auth_cache`` attached Redis."""
        return self._shared

    @property
    def has_redis(self) -> bool:
        return self._shared is not None

    @property
    def redis_connection(self) -> RedisCache | None:
        """The Redis the auth cache was attached to, for features that share it under keys of their own."""
        return self._redis_connection

    def _settings(self) -> AuthCacheSettings:
        return AuthCacheSettings(
            memory_entries=self.in_memory_cache.max_size_in_memory, memory_ttl=self.default_in_memory_ttl
        )

    def _python_only(self, method: str) -> None:
        if self._shared is not None:
            raise NotImplementedError(f"UserApiKeyCache.{method} is not part of the shared auth cache keyspace")

    def update_cache_ttl(self, default_in_memory_ttl: float | None, default_redis_ttl: float | None) -> None:
        super().update_cache_ttl(default_in_memory_ttl=default_in_memory_ttl, default_redis_ttl=default_redis_ttl)
        self.key_object_cache.update_cache_ttl(
            default_in_memory_ttl=default_in_memory_ttl, default_redis_ttl=default_redis_ttl
        )
        self.budget_windows_absent_cache.update_cache_ttl(
            default_in_memory_ttl=default_in_memory_ttl, default_redis_ttl=None
        )
        if self._shared is not None:
            self._shared.reconfigure(self._settings())

    def update_in_memory_max_size(self, max_size: int | None) -> None:
        super().update_in_memory_max_size(max_size)
        self.key_object_cache.update_in_memory_max_size(max_size)
        self.budget_windows_absent_cache.update_in_memory_max_size(max_size)
        if self._shared is not None:
            self._shared.reconfigure(self._settings())

    def attach_redis_cache(
        self, redis_cache: RedisCache | None = None, *, default_redis_ttl: float | None = None
    ) -> None:
        if redis_cache is None or self._shared is not None:
            return
        if default_redis_ttl is not None:
            self.update_cache_ttl(default_in_memory_ttl=None, default_redis_ttl=default_redis_ttl)
        self._redis_connection = redis_cache
        self._shared = self._auth_cache(redis_cache, self._settings())

    def get_local(self, key: str) -> object:
        """This worker's memory copy alone."""
        if self._shared is not None:
            return self._shared.get_local(key)
        return self._python_memory(key).get_cache(key=key)

    def set_local(self, key: str, value: object, ttl: float | None) -> None:
        """A memory write alone, with ``default_in_memory_ttl`` when ``ttl`` is None."""
        if self._shared is not None:
            self._shared.set_local(key, value, ttl)
            return
        effective_ttl: Final = self.default_in_memory_ttl if ttl is None else ttl
        if effective_ttl is None:
            self._python_memory(key).set_cache(key=key, value=value)
            return
        self._python_memory(key).set_cache(key=key, value=value, ttl=effective_ttl)

    async def lease(self, key: str) -> CacheLease:
        """Taken on a cache miss, before the database read that will fill ``key``, and handed to
        ``async_set_leased``. Evicting the key voids it: the rows read under it may predate the write behind
        that eviction. With ``enable_redis_auth_cache`` an eviction on any worker voids it. Loads of one key
        on this worker share a lease until an eviction, and it is dropped once none holds it."""
        held: Final = self._leases.get(key)
        if held is not None:
            return held
        shared: Final = None if self._shared is None else await self._shared.lease(key)
        return self._leases.setdefault(key, CacheLease(key, shared))

    async def async_set_leased(self, lease: CacheLease, value: CacheValue, ttl: float | None) -> bool:
        """Caches ``value`` unless the lease's key was evicted since it was taken; False when refused."""
        if self._leases.get(lease.key) is not lease:
            return False
        shared: Final = self._shared
        if shared is None:
            await self.async_set_cache(key=lease.key, value=value, ttl=ttl)
            return True
        payload: Final[object] = CacheCodec.serialize(value, model_type=None)
        written: Final = await shared.set_leased(lease.key, lease.shared, payload, ttl)
        if self._leases.get(lease.key) is not lease:
            return False
        if not written:
            self._void_leases((lease.key,))
            return False
        shared.set_local(lease.key, payload, ttl)
        return True

    def _void_leases(self, keys: Sequence[str]) -> None:
        for key in keys:
            _ = self._leases.pop(key, None)

    def delete_local(self, key: str) -> None:
        """Drops this worker's memory copy alone, as a cross-worker eviction broadcast does."""
        self._void_leases((key,))
        if key.startswith(_BUDGET_WINDOWS_ABSENT_PREFIX):
            self.budget_windows_absent_cache.delete_cache(key)
        if self._shared is not None:
            self._shared.delete_local(key)
            return
        self._python_memory(key).delete_cache(key)

    def _python_memory(self, key: str) -> InMemoryCache:
        return self.key_object_cache.in_memory_cache if is_user_key_cache_key(key) else self.in_memory_cache

    @overload
    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: object,
    ) -> T | None: ...

    @overload
    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: None = None,
        **kwargs: object,
    ) -> Any: ...

    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        shared: Final = self._shared
        cached: Final = _CACHED_VALUE.validate_python(
            self._get_shared_sync(shared, key, local_only)
            if shared is not None
            else self.key_object_cache.get_cache(
                key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
            )
            if is_user_key_cache_key(key)
            else super().get_cache(key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)
        )
        return _decoded(cached, key, model_type, "get_cache")

    @overload
    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: object,
    ) -> T | None: ...

    @overload
    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: None = None,
        **kwargs: object,
    ) -> Any: ...

    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        shared: Final = self._shared
        cached: Final = _CACHED_VALUE.validate_python(
            await self._get_shared(shared, key, local_only)
            if shared is not None
            else await self.key_object_cache.async_get_cache(
                key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
            )
            if is_user_key_cache_key(key)
            else await super().async_get_cache(
                key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
            )
        )
        return _decoded(cached, key, model_type, "async_get_cache")

    @staticmethod
    def _get_shared_sync(shared: AuthCacheStore, key: str, local_only: bool) -> object:
        try:
            return shared.get_local(key) if local_only else shared.get_sync(key)
        except Exception as e:  # noqa: BLE001  # DualCache.get_cache reads a failure as a miss
            verbose_proxy_logger.error("UserApiKeyCache.get_cache failed for key=%r: %s", key, e)
            return None

    @staticmethod
    async def _get_shared(shared: AuthCacheStore, key: str, local_only: bool) -> object:
        try:
            return shared.get_local(key) if local_only else await shared.get(key)
        except Exception as e:  # noqa: BLE001  # DualCache.async_get_cache reads a failure as a miss
            verbose_proxy_logger.error("UserApiKeyCache.async_get_cache failed for key=%r: %s", key, e)
            return None

    def set_cache(self, key: str | None, value: CacheValue, local_only: bool = False, **kwargs: object):
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final[object] = CacheCodec.serialize(value, model_type=model_type)
        shared: Final = self._shared
        if shared is not None:
            if key is None:
                return None
            try:
                if local_only:
                    shared.set_local(key, payload, _ttl(kwargs))
                else:
                    shared.set_sync(key, payload, _ttl(kwargs))
            except Exception as e:  # noqa: BLE001  # DualCache.set_cache swallows a failed write
                verbose_proxy_logger.error("UserApiKeyCache.set_cache failed for key=%r: %s", key, e)
            return None
        if key is not None and is_user_key_cache_key(key):
            return self.key_object_cache.set_cache(key=key, value=payload, local_only=local_only, **kwargs)
        return super().set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    async def async_set_cache(self, key: str | None, value: CacheValue, local_only: bool = False, **kwargs: object):
        """Inside a request the Redis SET rides the request's pipeline (memory is written at once); anywhere
        else, or with options the pipeline does not carry, it goes to Redis directly as before."""
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final[object] = CacheCodec.serialize(value, model_type=model_type)
        ttl: Final = kwargs.get("ttl")
        pipelined: Final = (
            key is not None
            and not local_only
            and kwargs.keys() <= _PIPELINED_SET_OPTIONS
            and (ttl is None or isinstance(ttl, (int, float)))
        )
        shared: Final = self._shared
        if shared is not None:
            if key is not None:
                await self._set_shared(shared, key, payload, local_only=local_only, pipelined=pipelined, kwargs=kwargs)
            return None
        if key is not None and is_user_key_cache_key(key):
            return await self.key_object_cache.async_set_cache(key=key, value=payload, local_only=local_only, **kwargs)
        return await super().async_set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    @staticmethod
    async def _set_shared(
        shared: AuthCacheStore,
        key: str,
        payload: object,
        *,
        local_only: bool,
        pipelined: bool,
        kwargs: Mapping[str, object],
    ) -> None:
        try:
            ttl: Final = _ttl(kwargs)
            if local_only:
                shared.set_local(key, payload, ttl)
            elif not pipelined or shared.declare_set(key, payload, ttl) is None:
                await shared.set(key, payload, ttl)
        except Exception as e:  # noqa: BLE001  # DualCache.async_set_cache swallows a failed write
            verbose_proxy_logger.error("UserApiKeyCache.async_set_cache failed for key=%r: %s", key, e)

    def delete_cache(self, key: str) -> None:
        self._void_leases((key,))
        if self._shared is not None:
            self._shared.delete_sync((key,))
            return
        if is_user_key_cache_key(key):
            self.key_object_cache.delete_cache(key)
            return
        super().delete_cache(key)

    async def async_delete_cache(self, key: str) -> None:
        self._void_leases((key,))
        if self._shared is not None:
            await self._shared.delete((key,))
            return
        if is_user_key_cache_key(key):
            await self.key_object_cache.async_delete_cache(key)
            return
        await super().async_delete_cache(key)

    def declare_delete(self, key: str) -> Awaitable[None] | None:
        """The DEL on the request's pre-call batch; None without Redis or outside a request, so the caller deletes."""
        self._void_leases((key,))
        return None if self._shared is None else self._shared.declare_delete(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        """Batch twin of ``async_delete_cache``, partitioned like
        ``async_set_cache_pipeline``.

        Both partitions are cleared even when one raises, because a caller
        batching these has already committed the rows they cache.
        """
        self._void_leases(keys)
        if self._shared is not None:
            if keys:
                await self._shared.delete(keys)
            return
        key_object_keys: Final = tuple(key for key in keys if is_user_key_cache_key(key))
        other_keys: Final = tuple(key for key in keys if not is_user_key_cache_key(key))
        outcomes: Final = await asyncio.gather(
            self.key_object_cache.async_delete_cache_keys(key_object_keys),
            super().async_delete_cache_keys(other_keys),
            return_exceptions=True,
        )
        failed: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, BaseException))
        if failed:
            raise failed[0]

    def flush_cache(self) -> None:
        self._leases.clear()
        self.budget_windows_absent_cache.flush_cache()
        if self._shared is not None:
            self._shared.flush_local()
            return
        super().flush_cache()
        self.key_object_cache.in_memory_cache.flush_cache()

    async def async_set_cache_pipeline(
        self, cache_list: Sequence[tuple[str, object]], local_only: bool = False, **kwargs: object
    ) -> None:
        """
        Batch writes with the same Codec boundary as ``async_set_cache`` without
        ``model_type``: ``BaseModel`` values become JSON-safe dicts; dicts/scalars unchanged.
        """
        normalized: Final = tuple((key, CacheCodec.serialize(value, model_type=None)) for key, value in cache_list)
        shared: Final = self._shared
        if shared is not None:
            await self._set_many_shared(shared, normalized, local_only=local_only, ttl=_ttl(kwargs))
            return
        key_object_entries: Final = tuple(entry for entry in normalized if is_user_key_cache_key(entry[0]))
        other_entries: Final = tuple(entry for entry in normalized if not is_user_key_cache_key(entry[0]))
        if key_object_entries:
            await self.key_object_cache.async_set_cache_pipeline(
                cache_list=key_object_entries, local_only=local_only, **kwargs
            )
        if other_entries:
            await super().async_set_cache_pipeline(cache_list=other_entries, local_only=local_only, **kwargs)

    @staticmethod
    async def _set_many_shared(
        shared: AuthCacheStore, entries: Sequence[tuple[str, object]], *, local_only: bool, ttl: float | None
    ) -> None:
        try:
            if not local_only:
                await shared.set_many(entries, ttl)
                return
            for key, value in entries:
                shared.set_local(key, value, ttl)
        except Exception as e:  # noqa: BLE001  # DualCache.async_set_cache_pipeline swallows a failed write
            verbose_proxy_logger.error("UserApiKeyCache.async_set_cache_pipeline failed: %s", e)

    async def increment_if_present(self, key: str, amount: float) -> float | None:
        """``INCRBYFLOAT`` of an entry Redis still holds; None without Redis or when Redis lacks the entry. Raises
        when Redis failed."""
        if self._shared is None:
            return None
        return await self._shared.increment_if_present(key, amount)

    async def async_batch_get_cache(
        self,
        keys: list[str],
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        throttle_redis: bool = True,
        **kwargs: object,  # kwargs-ok: forward the base cache options in Python-only mode
    ) -> list[object | None] | None:
        self._python_only("async_batch_get_cache")
        return await super().async_batch_get_cache(
            keys, parent_otel_span=parent_otel_span, local_only=local_only, throttle_redis=throttle_redis, **kwargs
        )

    async def async_increment_cache(
        self,
        key: str,
        value: float,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        refresh_ttl: bool = False,
        **kwargs: object,  # kwargs-ok: forward the base cache options in Python-only mode
    ) -> float | None:
        self._python_only("async_increment_cache")
        return await super().async_increment_cache(
            key, value, parent_otel_span=parent_otel_span, local_only=local_only, refresh_ttl=refresh_ttl, **kwargs
        )


def _decoded(cached: object, key: str, model_type: type[BaseModel] | None, method: str) -> object:
    if model_type is None or cached is None:
        return cached
    decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
    if decoded is None:
        verbose_proxy_logger.error(
            "UserApiKeyCache.%s failed to deserialize cached value for key=%r model_type=%s",
            method,
            key,
            getattr(model_type, "__name__", str(model_type)),
        )
    return decoded


#: Value cached under ``user_object_permission_id_cache_key`` when the user links no permission row,
#: so a human without an entitlement costs no DB read per request. Lives beside the key builder
#: because it is part of the same cache protocol: a reader that knows the key must know this value.
USER_NO_MCP_PERMISSION_SENTINEL: Final = "__user_no_mcp_permission__"


def user_object_permission_id_cache_key(user_id: str) -> str:
    """Cache key for the ``user_id -> object_permission_id`` link.

    Lives here rather than next to either user because two modules own the two halves: the MCP auth
    resolver writes it on read, and ``/user/update`` deletes it after changing the link. A key format
    duplicated across those two drifts silently, and the failure is an entitlement change that never
    takes effect.
    """
    return f"user_object_permission_id:{user_id}"


def object_permission_cache_key(object_permission_id: str) -> str:
    """Cache key ``get_object_permission`` stores a permission row under."""
    return f"object_permission_id:{object_permission_id}"


#: Cached under ``tag_registry_cache_key`` when the table exceeds ``TAG_REGISTRY_MAX_SIZE``:
#: registry unusable, fall back to the per-tag lookup.
TAG_REGISTRY_OVERFLOW_SENTINEL: Final = "__tag_registry_overflow__"


def tag_cache_key(tag_name: str) -> str:
    """Cache key one tag row is stored under; shared so its five reader/writer modules cannot drift."""
    return f"tag:{tag_name}"


def tag_registry_cache_key() -> str:
    """Cache key for the set of tag names that exist in ``LiteLLM_TagTable``."""
    return "tag_registry"


#: Cached under ``model_access_group_registry_cache_key`` when the table exceeds
#: ``MODEL_ACCESS_GROUP_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-group lookup.
MODEL_ACCESS_GROUP_REGISTRY_OVERFLOW_SENTINEL: Final = "__model_access_group_registry_overflow__"


def model_access_group_cache_key(access_group_name: str) -> str:
    """Cache key one model access group budget row is stored under; shared so auth, spend tracking and the management endpoints cannot drift."""
    return f"model_access_group:{access_group_name}"


def model_access_group_registry_cache_key() -> str:
    """Cache key for the set of model access group names that have a budget row."""
    return "model_access_group_registry"


def project_cache_key(project_id: str) -> str:
    return f"project_id:{project_id}"


#: Cached under ``end_user_restricted_registry_cache_key`` when the restricted set exceeds
#: ``END_USER_RESTRICTED_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-id fetch.
END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL: Final = "__end_user_restricted_registry_overflow__"


def end_user_cache_key(end_user_id: str) -> str:
    """Cache key one end-user row is stored under; shared so auth and spend tracking cannot drift."""
    return f"end_user_id:{end_user_id}"


def end_user_restricted_registry_cache_key() -> str:
    """Cache key for the set of end-user ids whose row carries a restriction auth enforces."""
    return "end_user_restricted_registry"


def team_membership_auth_cache_key(team_id: str, user_id: str) -> str:
    """Cache key one team member's ``LiteLLM_TeamMembership`` row is stored under for the admission check."""
    return f"{team_id}_{user_id}"


def team_membership_reservation_cache_key(user_id: str, team_id: str) -> str:
    """Cache key the pre-call budget reservation stores the same ``LiteLLM_TeamMembership`` row under.

    Deliberately not unified with ``team_membership_auth_cache_key``: the two readers wrote independent
    keys before this file existed, so a fix that invalidates one must invalidate both explicitly rather
    than assume a single write is visible to both.
    """
    return f"team_membership:{user_id}:{team_id}"


#: Cached under ``team_membership_reservation_cache_key`` when a member has no ``LiteLLM_TeamMembership``
#: row, so a session-token member without a per-member budget costs no DB read per request. Lives beside
#: the key builder because it is part of the same cache protocol: every reader of the key must know that
#: a plain string here means "no row", distinct from a serialized membership. The two budget readers
#: already treat a non-model value as "no row", so they need no change to stay correct.
NO_TEAM_MEMBERSHIP_SENTINEL: Final = "__no_team_membership__"


def budget_windows_absent_cache_key(entity_type: str, entity_id: str) -> str:
    """Cache key marking a key or team that holds no budget windows, so the spend fallback skips its row read."""
    return f"{_BUDGET_WINDOWS_ABSENT_PREFIX}{entity_type}:{entity_id}"


def get_management_object_ttl(cache: DualCache) -> float:
    """
    In-memory TTL for management-object cache writes (keys, teams, users, budgets, ...).

    Honors ``general_settings.user_api_key_cache_ttl``, which ``proxy_server``
    propagates onto ``default_in_memory_ttl`` at startup, and falls back to
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` when no default is configured.
    """
    configured: Final[float | None] = getattr(cache, "default_in_memory_ttl", None)
    if configured is not None:
        return configured
    return DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

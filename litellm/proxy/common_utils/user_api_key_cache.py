from __future__ import annotations

from typing import Any, Final, TypeVar, cast, overload

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec

T = TypeVar("T", bound=BaseModel)


class UserApiKeyCache(DualCache):
    """
    DualCache wrapper for UserAPIKeyAuth-like payloads.

    Stores a Redis-safe JSON payload in BOTH in-memory and Redis to avoid
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

    ``async_set_cache_pipeline`` applies the same untyped Codec pass as omitting
    ``model_type`` on ``async_set_cache`` (so ``BaseModel`` rows are dumped before Redis).

    ``get_cache`` / ``async_get_cache`` overloads and implementations must be contiguous
    (no other methods in between) so mypy resolves ``@overload`` + implementation correctly.
    """

    @overload
    def get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: Any,
    ) -> T | None: ...

    @overload
    def get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    def get_cache(
        self,
        key,
        parent_otel_span=None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs,
    ) -> Any | BaseModel | None:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        cached: Final = super().get_cache(key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.get_cache failed to deserialize cached value for key=%r model_type=%s",
                key,
                getattr(model_type, "__name__", str(model_type)),
            )
            return None
        return decoded

    @overload
    async def async_get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: Any,
    ) -> T | None: ...

    @overload
    async def async_get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    async def async_get_cache(
        self,
        key,
        parent_otel_span=None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs,
    ) -> Any | BaseModel | None:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        cached: Final = await super().async_get_cache(
            key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
        )
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.async_get_cache failed to deserialize cached value for key=%r model_type=%s",
                key,
                getattr(model_type, "__name__", str(model_type)),
            )
            return None
        return decoded

    def set_cache(self, key, value, local_only: bool = False, **kwargs):
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final = CacheCodec.serialize(value, model_type=model_type)
        return super().set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    async def async_set_cache(self, key, value, local_only: bool = False, **kwargs):
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final = CacheCodec.serialize(value, model_type=model_type)
        return await super().async_set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    async def async_set_cache_pipeline(self, cache_list: list, local_only: bool = False, **kwargs) -> None:
        """
        Batch writes with the same Codec boundary as ``async_set_cache`` without
        ``model_type``: ``BaseModel`` values become JSON-safe dicts; dicts/scalars unchanged.
        """
        normalized: Final = [(key, CacheCodec.serialize(value, model_type=None)) for key, value in cache_list]
        return await super().async_set_cache_pipeline(cache_list=normalized, local_only=local_only, **kwargs)


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

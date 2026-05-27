from __future__ import annotations

from typing import Any, Optional, Type, TypeVar, Union, cast, overload

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
        model_type: Type[T],
        **kwargs: Any,
    ) -> Optional[T]: ...

    @overload
    def get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    def get_cache(  # type: ignore[override]
        self,
        key,
        parent_otel_span=None,
        local_only: bool = False,
        model_type: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[Any, Optional[BaseModel]]:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(Optional[Type[BaseModel]], kwargs.pop("model_type", None))
        cached = super().get_cache(
            key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
        )
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.get_cache failed to deserialize cached value for "
                "key=%r model_type=%s",
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
        model_type: Type[T],
        **kwargs: Any,
    ) -> Optional[T]: ...

    @overload
    async def async_get_cache(
        self,
        key: Any,
        parent_otel_span: Any = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    async def async_get_cache(  # type: ignore[override]
        self,
        key,
        parent_otel_span=None,
        local_only: bool = False,
        model_type: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[Any, Optional[BaseModel]]:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(Optional[Type[BaseModel]], kwargs.pop("model_type", None))
        cached = await super().async_get_cache(
            key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
        )
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.async_get_cache failed to deserialize cached value for "
                "key=%r model_type=%s",
                key,
                getattr(model_type, "__name__", str(model_type)),
            )
            return None
        return decoded

    def _resolve_management_object_ttl(self, kwargs: dict) -> None:
        """
        Honor ``general_settings.user_api_key_cache_ttl`` for management-object writes.

        Auth-related cache writers (``litellm/proxy/auth/auth_checks.py``,
        ``litellm/proxy/auth/handle_jwt.py``,
        ``litellm/proxy/_experimental/mcp_server/mcp_server_manager.py``) explicitly
        pass ``ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` (60s) on every
        write. When the operator sets ``general_settings.user_api_key_cache_ttl``
        in the proxy config, ``proxy_server.py`` updates this cache instance via
        ``update_cache_ttl(default_in_memory_ttl=<configured>)`` -- but DualCache
        only falls back to ``default_in_memory_ttl`` when ``ttl`` is absent from
        kwargs, so the explicit 60s wins and the configured TTL is silently lost.

        This helper treats the ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL``
        constant as a *sentinel*: when the caller passes exactly that value AND
        the cache instance has been configured with a different
        ``default_in_memory_ttl``, the configured value is promoted. Any other
        explicit ttl is left untouched.
        """
        if "ttl" not in kwargs:
            return
        ttl = kwargs["ttl"]
        if ttl is None:
            return
        try:
            ttl_value = float(ttl)
        except (TypeError, ValueError):
            return
        if ttl_value != float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL):
            return
        configured = self.default_in_memory_ttl
        if configured is None:
            return
        try:
            configured_value = float(configured)
        except (TypeError, ValueError):
            return
        if configured_value == float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL):
            return
        kwargs["ttl"] = configured_value

    def set_cache(self, key, value, local_only: bool = False, **kwargs):  # type: ignore[override]
        model_type = cast(Optional[Type[BaseModel]], kwargs.pop("model_type", None))
        payload = CacheCodec.serialize(value, model_type=model_type)
        self._resolve_management_object_ttl(kwargs)
        return super().set_cache(
            key=key, value=payload, local_only=local_only, **kwargs
        )

    async def async_set_cache(self, key, value, local_only: bool = False, **kwargs):  # type: ignore[override]
        model_type = cast(Optional[Type[BaseModel]], kwargs.pop("model_type", None))
        payload = CacheCodec.serialize(value, model_type=model_type)
        self._resolve_management_object_ttl(kwargs)
        return await super().async_set_cache(
            key=key, value=payload, local_only=local_only, **kwargs
        )

    async def async_set_cache_pipeline(  # type: ignore[override]
        self, cache_list: list, local_only: bool = False, **kwargs
    ) -> None:
        """
        Batch writes with the same Codec boundary as ``async_set_cache`` without
        ``model_type``: ``BaseModel`` values become JSON-safe dicts; dicts/scalars unchanged.
        """
        normalized = [
            (key, CacheCodec.serialize(value, model_type=None))
            for key, value in cache_list
        ]
        return await super().async_set_cache_pipeline(
            cache_list=normalized, local_only=local_only, **kwargs
        )

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.state_cache import StateCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class HookCache(StateCache):
    """
    The ``cache`` a pre-call hook is handed. Hooks used to be handed the auth cache itself, and a hook written against
    that reads keys, users and teams out of it by cache key and deletes them from it. So a read answers with this
    worker's copy of an auth object when it holds one and with the hook's own state otherwise, and a delete drops the
    key from both. A write is the hook's own state: this worker keeps the object the hook stored, and the Redis the
    workers share gets it as JSON, a pydantic model dumped as it was when hooks wrote into the auth cache.

    The two share one Redis keyspace, which the auth cache owns: an auth object this worker does not hold is read from
    Redis as any other key of this cache, and a delete reaches Redis through the auth cache alone.
    """

    def __init__(self, auth_cache: Callable[[], UserApiKeyCache], default_in_memory_ttl: float | None = None) -> None:
        super().__init__(default_in_memory_ttl=default_in_memory_ttl)
        self._auth_cache: Final = auth_cache

    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> object:
        held: Final[object] = self._auth_cache().get_local_by_key(key)
        if held is not None:
            return held
        return super().get_cache(key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)

    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> object:
        held: Final[object] = self._auth_cache().get_local_by_key(key)
        if held is not None:
            return held
        return await super().async_get_cache(key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)

    def _keep(self, entries: Sequence[tuple[str, object]], kwargs: Mapping[str, object]) -> None:
        for key, value in entries:
            super().set_cache(key, value, local_only=True, **kwargs)

    def set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        super().set_cache(key, CacheCodec.serialize(value), local_only=local_only, **kwargs)
        self._keep(((key, value),), kwargs)

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        await super().async_set_cache(key, CacheCodec.serialize(value), local_only=local_only, **kwargs)
        self._keep(((key, value),), kwargs)

    async def async_set_cache_pipeline(
        self,
        cache_list: Sequence[tuple[str, object]],
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        shared: Final = tuple((key, CacheCodec.serialize(value)) for key, value in cache_list)
        await super().async_set_cache_pipeline(shared, local_only=local_only, **kwargs)
        self._keep(cache_list, kwargs)

    def delete_cache(self, key: str) -> None:
        self.in_memory_cache.delete_cache(key)
        self._auth_cache().delete_cache(key)

    async def async_delete_cache(self, key: str) -> None:
        self.in_memory_cache.delete_cache(key)
        await self._auth_cache().async_delete_cache(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.in_memory_cache.delete_cache(key)
        await self._auth_cache().async_delete_cache_keys(keys)

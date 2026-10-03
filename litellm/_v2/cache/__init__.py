from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache as CacheFacade
from litellm.types.caching import LiteLLMCacheType

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeCacheHandle

_DURATION: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


class NativeBackend(BaseCache):
    def __init__(self, handle: NativeCacheHandle) -> None:
        self.native_handle = handle

    def get_cache(self, key: str, **kwargs: object) -> object:
        return self.native_handle.get(key)

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        return await self.native_handle.async_get(key)

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.native_handle.set(key, value, ttl=_DURATION.validate_python(kwargs.get("ttl")))

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        await self.native_handle.async_set(key, value, ttl=_DURATION.validate_python(kwargs.get("ttl")))

    async def async_set_cache_pipeline(self, cache_list: Sequence[tuple[str, object]], **kwargs: object) -> None:
        await self.native_handle.async_set_many(cache_list, ttl=_DURATION.validate_python(kwargs.get("ttl")))

    async def batch_cache_write(self, key: str, value: object, **kwargs: object) -> None:
        await self.async_set_cache(key, value, **kwargs)

    def flush_cache(self) -> None:
        self.native_handle.flush()

    async def async_flush_cache(self) -> None:
        await self.native_handle.async_flush()

    async def ping(self) -> bool:
        return await self.native_handle.ping()

    async def disconnect(self) -> None:
        await self.native_handle.disconnect()

    async def delete_cache_keys(self, keys: Sequence[str]) -> None:
        await self.native_handle.delete(keys)

    async def test_connection(self) -> dict[str, str]:
        return {"status": "success" if await self.ping() else "failed"}


class Cache:
    @staticmethod
    def memory(*, ttl: float = 600, capacity: int = 200, max_entry_bytes: int = 4194304) -> CacheFacade:
        from litellm.rust_bridge._native import NativeCacheHandle

        handle: Final = NativeCacheHandle.memory(ttl=ttl, capacity=capacity, max_entry_bytes=max_entry_bytes)
        return CacheFacade(type=LiteLLMCacheType.LOCAL, ttl=ttl, _backend=NativeBackend(handle))

    @staticmethod
    def redis(url: str, *, namespace: str, ttl: float = 600, max_entry_bytes: int = 4194304) -> CacheFacade:
        from litellm.rust_bridge._native import NativeCacheHandle

        handle: Final = NativeCacheHandle.redis(url, namespace=namespace, ttl=ttl, max_entry_bytes=max_entry_bytes)
        return CacheFacade(type=LiteLLMCacheType.REDIS, namespace=namespace, ttl=ttl, _backend=NativeBackend(handle))

import asyncio
from typing import Final

import pytest

import litellm

from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache
from litellm.caching.in_memory_cache import InMemoryCache


class CallerCache(BaseCache):
    def __init__(self, lookup_entered: asyncio.Event | None = None, release: asyncio.Event | None = None) -> None:
        super().__init__()
        self.storage: Final = InMemoryCache()
        self.caller: Final = asyncio.current_task()
        self.lookup_entered: Final = lookup_entered
        self.release: Final = release

    async def async_set_cache_pipeline(self, cache_list: list[tuple[str, object]], ttl: float | None = None) -> None:
        await self.storage.async_set_cache_pipeline(cache_list, ttl=ttl)

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        if self.lookup_entered is not None and self.release is not None:
            self.lookup_entered.set()
            await self.release.wait()
        assert asyncio.current_task() is self.caller
        return self.storage.get_cache(key, **kwargs)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        assert asyncio.current_task() is self.caller
        await asyncio.sleep(0)
        self.storage.set_cache(key, value, **kwargs)


def use_cache(monkeypatch: pytest.MonkeyPatch, cache: Cache | None = None) -> Cache:
    configured: Final = cache or Cache()
    monkeypatch.setattr(litellm, "cache", configured)
    return configured

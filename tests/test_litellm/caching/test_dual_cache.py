import pytest

from litellm.caching.dual_cache import DualCache


class _InMemoryCacheStub:
    def __init__(self) -> None:
        self._store = {}

    def set_cache(self, key, value, **kwargs):
        self._store[key] = value

    def batch_get_cache(self, keys, **kwargs):
        return [self._store.get(key) for key in keys]

    async def async_batch_get_cache(self, keys, **kwargs):
        raise AssertionError(
            "DualCache.batch_get_cache should not call async cache APIs"
        )


class _RedisCacheStub:
    def batch_get_cache(self, key_list, parent_otel_span=None):
        return {key: f"redis:{key}" for key in key_list}

    async def async_batch_get_cache(self, key_list, parent_otel_span=None):
        raise AssertionError(
            "DualCache.batch_get_cache should not call async cache APIs"
        )


@pytest.mark.asyncio
async def test_batch_get_cache_stays_sync_inside_event_loop():
    in_memory_cache = _InMemoryCacheStub()
    in_memory_cache.set_cache("key-1", "memory:key-1")
    redis_cache = _RedisCacheStub()
    dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)

    result = dual_cache.batch_get_cache(keys=["key-1", "key-2"])

    assert result == ["memory:key-1", "redis:key-2"]
    assert in_memory_cache._store["key-2"] == "redis:key-2"

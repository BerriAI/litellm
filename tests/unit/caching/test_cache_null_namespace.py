from typing import Final

from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType


def test_explicit_null_cache_control_uses_cache_namespace():
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, namespace="instance")
    request: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}

    default_key: Final = cache.get_cache_key(**request, metadata=None)
    null_cache_key: Final = cache.get_cache_key(**request, metadata=None, cache=None)
    request_namespace_key: Final = cache.get_cache_key(
        **request,
        metadata=None,
        cache={"namespace": "request"},
    )

    assert null_cache_key == default_key
    assert request_namespace_key.startswith("request:")
    assert request_namespace_key != default_key


def test_explicit_null_cache_control_allows_sync_cache_lookup():
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}

    assert cache.get_cache(**request, cache=None) is None


async def test_explicit_null_cache_control_allows_async_cache_lookup():
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}

    assert await cache.async_get_cache(**request, cache=None) is None

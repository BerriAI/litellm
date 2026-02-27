"""e2e tests: httpx clients obtained via get_async_httpx_client must remain
usable after LLMClientCache evicts their cache entry."""

import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client


@pytest.fixture(autouse=True)
def _tiny_client_cache(monkeypatch):
    """Replace the global client cache with a size-1 cache so eviction
    triggers on the second insert."""
    cache = LLMClientCache(max_size_in_memory=1, default_ttl=600)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", cache)
    yield cache


@pytest.mark.asyncio
async def test_evicted_client_is_not_closed():
    """Get a client via get_async_httpx_client, evict it by caching a second
    one, then verify the first client's transport is still open."""
    client_a = get_async_httpx_client(llm_provider="provider_a")
    # This evicts client_a from cache (capacity=1)
    client_b = get_async_httpx_client(llm_provider="provider_b")

    assert not client_a.client.is_closed
    await client_a.client.aclose()
    await client_b.client.aclose()


@pytest.mark.asyncio
async def test_expired_client_is_not_closed():
    """Get a client, expire it via TTL, then verify the client is still open."""
    cache = litellm.in_memory_llm_clients_cache
    client = get_async_httpx_client(llm_provider="provider_ttl")

    # Force the entry to expire and trigger eviction
    for key in list(cache.ttl_dict.keys()):
        cache.ttl_dict[key] = 0
        # Also fix the heap entry so evict_cache finds it
        cache.expiration_heap = [(0, key) for _, key in cache.expiration_heap]
    cache.evict_cache()

    assert not client.client.is_closed
    await client.client.aclose()

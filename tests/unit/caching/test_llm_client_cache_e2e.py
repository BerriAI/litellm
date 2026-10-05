"""e2e tests: httpx clients obtained via get_async_httpx_client must remain
usable after LLMClientCache evicts their cache entry.

These tests exist to prevent a recurring production crash:
  RuntimeError: Cannot send a request, as the client has been closed.

The bug occurs when LLMClientCache._remove_key() eagerly closes evicted
clients that are still referenced by in-flight requests.  Every test here
sleeps after eviction to let the event loop drain any background close
tasks — a plain ``assert not client.is_closed`` without sleeping is NOT
sufficient to catch the regression (the close task runs asynchronously).

See: https://github.com/BerriAI/litellm/pull/22247
"""

import asyncio

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

    # Sleep to let any background close tasks execute — without this sleep,
    # a regression that schedules close via create_task() would go undetected.
    await asyncio.sleep(0.15)

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

    await asyncio.sleep(0.15)

    assert not client.client.is_closed
    await client.client.aclose()


@pytest.mark.asyncio
async def test_evicted_openai_sdk_client_stays_usable():
    """OpenAI/Azure SDK clients cached in LLMClientCache must remain usable
    after eviction.  This is the exact production scenario: the proxy caches
    an AsyncOpenAI client, the TTL expires, a new request evicts the old
    entry, but a concurrent streaming request is still reading from it.

    Regression guard: if _remove_key ever calls client.close(), the
    underlying httpx client is closed and this test fails.
    """
    from openai import AsyncOpenAI

    cache = litellm.in_memory_llm_clients_cache

    client = AsyncOpenAI(api_key="sk-test", base_url="https://api.openai.com/v1")
    cache.set_cache("openai-client", client, ttl=600)

    # Evict by inserting a second entry (max_size=1)
    cache.set_cache("filler", "x", ttl=600)

    # Let the event loop drain any background close tasks
    await asyncio.sleep(0.15)

    # The SDK client's internal httpx client must still be open
    assert not client._client.is_closed, (
        "AsyncOpenAI client was closed on cache eviction — this causes "
        "'Cannot send a request, as the client has been closed' in production"
    )
    await client.close()


@pytest.mark.asyncio
async def test_ttl_expired_openai_sdk_client_stays_usable():
    """Same as above but triggered via TTL expiry + get_cache (the other
    eviction path)."""
    from openai import AsyncOpenAI

    cache = litellm.in_memory_llm_clients_cache

    client = AsyncOpenAI(api_key="sk-test", base_url="https://api.openai.com/v1")
    cache.set_cache("openai-client", client, ttl=600)

    # Force TTL expiry
    for key in list(cache.ttl_dict.keys()):
        cache.ttl_dict[key] = 0
        cache.expiration_heap = [(0, k) for _, k in cache.expiration_heap]

    # get_cache calls evict_element_if_expired → _remove_key
    result = cache.get_cache("openai-client")
    assert result is None  # expired, so returns None

    await asyncio.sleep(0.15)

    assert not client._client.is_closed, (
        "AsyncOpenAI client was closed on TTL expiry — this causes "
        "'Cannot send a request, as the client has been closed' in production"
    )
    await client.close()

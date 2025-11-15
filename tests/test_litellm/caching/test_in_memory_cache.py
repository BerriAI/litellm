import asyncio
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.in_memory_cache import InMemoryCache


def test_in_memory_openai_obj_cache():
    from openai import OpenAI

    openai_obj = OpenAI(api_key="my-fake-key")

    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value=openai_obj)

    cached_obj = in_memory_cache.get_cache(key="my-fake-key")

    assert cached_obj is not None

    assert cached_obj == openai_obj


def test_in_memory_cache_max_size_per_item():
    """
    Test that the cache will not store items larger than the max size per item
    """
    in_memory_cache = InMemoryCache(max_size_per_item=100)

    result = in_memory_cache.check_value_size("a" * 100000000)

    assert result is False


def test_in_memory_cache_ttl():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value", ttl=10)
    initial_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert initial_ttl_time is not None

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value-2", ttl=10)
    new_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert new_ttl_time == initial_ttl_time  # ttl should not be updated

    ## On object expiration, the ttl should be removed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    time.sleep(1)
    cached_obj = in_memory_cache.get_cache(key="new-fake-key")
    new_ttl_time = in_memory_cache.ttl_dict.get("new-fake-key")
    assert new_ttl_time is None


def test_in_memory_cache_ttl_allow_override():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()
    ## On object expiration, but no get_cache, the override should be allowed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    initial_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert initial_ttl_time is not None
    time.sleep(1)

    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value-2", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    assert new_ttl_time != initial_ttl_time


def test_in_memory_cache_max_size_with_ttl():
    """
    Test that max_size_in_memory is respected even when all items have long TTLs.
    This tests the fix for the unbounded growth issue.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=3)
    long_ttl = 86400  # 1 day
    
    # Fill the cache to max capacity
    for i in range(3):
        in_memory_cache.set_cache(key=f"key_{i}", value=f"value_{i}", ttl=long_ttl)
        time.sleep(0.01)  # Small delay to ensure different timestamps
    
    assert len(in_memory_cache.cache_dict) == 3
    assert len(in_memory_cache.ttl_dict) == 3
    
    # Add another item - should evict the earliest item
    in_memory_cache.set_cache(key="key_3", value="value_3", ttl=long_ttl)
    
    # Cache should still be at max size, not larger
    assert len(in_memory_cache.cache_dict) == 3
    assert len(in_memory_cache.ttl_dict) == 3
    
    # key_0 should have been evicted (it was added first)
    assert "key_0" not in in_memory_cache.cache_dict
    assert "key_0" not in in_memory_cache.ttl_dict
    
    # Other keys should still be present
    assert "key_1" in in_memory_cache.cache_dict
    assert "key_2" in in_memory_cache.cache_dict
    assert "key_3" in in_memory_cache.cache_dict


def test_in_memory_cache_expired_items_evicted_first():
    """
    Test that expired items are evicted before non-expired items when cache is full.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=3)
    
    # Add items with short TTL that will expire
    in_memory_cache.set_cache(key="expired_1", value="value_1", ttl=1)
    in_memory_cache.set_cache(key="expired_2", value="value_2", ttl=1)
    
    # Add item with long TTL
    in_memory_cache.set_cache(key="long_lived", value="value_long", ttl=86400)
    
    assert len(in_memory_cache.cache_dict) == 3
    
    # Wait for short TTL items to expire
    time.sleep(2)
    
    # Add new item - should evict expired items first, not the long-lived one
    in_memory_cache.set_cache(key="new_item", value="new_value", ttl=86400)
    
    # Long-lived item should still be present
    assert "long_lived" in in_memory_cache.cache_dict
    assert "new_item" in in_memory_cache.cache_dict
    
    # Expired items should be gone
    assert "expired_1" not in in_memory_cache.cache_dict
    assert "expired_2" not in in_memory_cache.cache_dict
    assert "expired_1" not in in_memory_cache.ttl_dict
    assert "expired_2" not in in_memory_cache.ttl_dict


def test_in_memory_cache_eviction_order():
    """
    Test that when non-expired items need to be evicted, those with earliest expiration times are evicted first.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=2)
    
    # Add items with different TTLs
    now = time.time()
    in_memory_cache.set_cache(key="early_expire", value="value_1", ttl=100)  # expires in 100 seconds
    time.sleep(0.01)
    in_memory_cache.set_cache(key="late_expire", value="value_2", ttl=200)   # expires in 200 seconds
    
    # Verify TTL order
    early_ttl = in_memory_cache.ttl_dict["early_expire"]
    late_ttl = in_memory_cache.ttl_dict["late_expire"]
    assert early_ttl < late_ttl, "early_expire should have earlier expiration time"
    
    assert len(in_memory_cache.cache_dict) == 2
    
    # Add third item - should evict the one with earliest expiration time
    in_memory_cache.set_cache(key="new_item", value="value_3", ttl=300)
    
    assert len(in_memory_cache.cache_dict) == 2
    
    # Item with earliest expiration should be evicted
    assert "early_expire" not in in_memory_cache.cache_dict
    assert "early_expire" not in in_memory_cache.ttl_dict
    
    # Items with later expiration should remain
    assert "late_expire" in in_memory_cache.cache_dict
    assert "new_item" in in_memory_cache.cache_dict


def test_in_memory_cache_heap_size_staus_bounded():
    """
    Test that the expiration_heap does not grow unbounded when the same key is updated repeaatedly.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=10)

    for i in range(1_000):
        in_memory_cache.set_cache(key="hot_key", value=f"value_{i}", ttl=60)

    # Expiration heap should only have 1 entry
    assert len(in_memory_cache.expiration_heap) == 1


def test_in_memory_cache_cleanup_sync_close():
    """
    Test that objects with sync close() method are properly cleaned up when removed from cache.
    """
    class MockSyncClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    in_memory_cache = InMemoryCache()
    mock_client = MockSyncClient()

    # Add client to cache with short TTL
    in_memory_cache.set_cache(key="sync_client", value=mock_client, ttl=1)

    assert "sync_client" in in_memory_cache.cache_dict
    assert mock_client.closed is False

    # Wait for TTL to expire
    time.sleep(1.1)

    # Access cache to trigger eviction
    result = in_memory_cache.get_cache(key="sync_client")

    # Client should be removed and closed
    assert result is None
    assert "sync_client" not in in_memory_cache.cache_dict
    assert mock_client.closed is True


@pytest.mark.asyncio
async def test_in_memory_cache_cleanup_async_close():
    """
    Test that objects with async close() method are properly cleaned up when removed from cache.
    """
    class MockAsyncClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True
            await asyncio.sleep(0.01)  # Simulate async cleanup

    in_memory_cache = InMemoryCache()
    mock_client = MockAsyncClient()

    # Add client to cache with short TTL
    in_memory_cache.set_cache(key="async_client", value=mock_client, ttl=1)

    assert "async_client" in in_memory_cache.cache_dict
    assert mock_client.closed is False

    # Wait for TTL to expire
    await asyncio.sleep(1.1)

    # Access cache to trigger eviction
    result = in_memory_cache.get_cache(key="async_client")

    # Client should be removed
    assert result is None
    assert "async_client" not in in_memory_cache.cache_dict

    # Give async cleanup task time to complete
    await asyncio.sleep(0.1)

    # Client should be closed
    assert mock_client.closed is True


def test_in_memory_cache_cleanup_no_close_method():
    """
    Test that objects without close() method are removed normally without errors.
    """
    in_memory_cache = InMemoryCache()

    # Add regular object without close() method
    in_memory_cache.set_cache(key="regular_obj", value="test_value", ttl=1)

    assert "regular_obj" in in_memory_cache.cache_dict

    # Wait for TTL to expire
    time.sleep(1.1)

    # Access cache to trigger eviction - should not raise any errors
    result = in_memory_cache.get_cache(key="regular_obj")

    assert result is None
    assert "regular_obj" not in in_memory_cache.cache_dict


def test_in_memory_cache_cleanup_close_raises_error():
    """
    Test that errors during close() are silently handled and don't break cache operations.
    """
    class MockFailingClient:
        def close(self):
            raise RuntimeError("Close failed!")

    in_memory_cache = InMemoryCache()
    mock_client = MockFailingClient()

    # Add client to cache with short TTL
    in_memory_cache.set_cache(key="failing_client", value=mock_client, ttl=1)

    assert "failing_client" in in_memory_cache.cache_dict

    # Wait for TTL to expire
    time.sleep(1.1)

    # Access cache to trigger eviction - should not raise error even if close() fails
    result = in_memory_cache.get_cache(key="failing_client")

    # Client should still be removed from cache
    assert result is None
    assert "failing_client" not in in_memory_cache.cache_dict


def test_in_memory_cache_cleanup_on_eviction():
    """
    Test that objects are cleaned up when evicted due to max_size constraint.
    """
    class MockClient:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True

    in_memory_cache = InMemoryCache(max_size_in_memory=2)

    # Add clients to cache
    client1 = MockClient("client1")
    client2 = MockClient("client2")
    client3 = MockClient("client3")

    in_memory_cache.set_cache(key="client1", value=client1, ttl=100)
    time.sleep(0.01)
    in_memory_cache.set_cache(key="client2", value=client2, ttl=100)

    assert len(in_memory_cache.cache_dict) == 2
    assert client1.closed is False
    assert client2.closed is False

    # Add third client - should evict first client
    in_memory_cache.set_cache(key="client3", value=client3, ttl=100)

    # First client should be evicted and closed
    assert len(in_memory_cache.cache_dict) == 2
    assert "client1" not in in_memory_cache.cache_dict
    assert client1.closed is True

    # Other clients should remain
    assert "client2" in in_memory_cache.cache_dict
    assert "client3" in in_memory_cache.cache_dict
    assert client2.closed is False
    assert client3.closed is False


@pytest.mark.asyncio
async def test_in_memory_cache_cleanup_httpx_client():
    """
    Test that real httpx AsyncClient is properly cleaned up (integration test).
    """
    in_memory_cache = InMemoryCache()

    # Create a real httpx client
    client = httpx.AsyncClient()

    assert client.is_closed is False

    # Add to cache with short TTL
    in_memory_cache.set_cache(key="httpx_client", value=client, ttl=1)

    # Wait for TTL to expire
    await asyncio.sleep(1.1)

    # Trigger eviction
    result = in_memory_cache.get_cache(key="httpx_client")

    assert result is None
    assert "httpx_client" not in in_memory_cache.cache_dict

    # Give async cleanup time to complete
    await asyncio.sleep(0.1)

    # Client should be closed
    assert client.is_closed is True


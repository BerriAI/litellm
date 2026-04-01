"""
Tests for Redis batch caching optimizations (commit 3f52e8c)

Verifies:

1. Batch cache size increased from 100 → 1000 (minimum 1k)
2. Repeated Redis queries for cache misses are throttled
"""

import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("../.."))

import uuid
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE


@pytest.fixture
def cache_setup():
    """Create cache instances for testing"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(
        host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT")
    )
    dual_cache = DualCache(
        in_memory_cache=in_memory,
        redis_cache=redis_cache,
        default_max_redis_batch_cache_size=DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE,
    )
    return dual_cache, in_memory, redis_cache


@pytest.mark.asyncio
async def test_batch_cache_size_is_1000_minimum(cache_setup):
    """Verify batch cache size is set to 1000 (never below 1k)"""
    dual_cache, _, _ = cache_setup
    
    # Critical: batch cache size must be at least DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE
    assert dual_cache.last_redis_batch_access_time.max_size >= DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE


@pytest.mark.asyncio
async def test_throttling_prevents_duplicate_redis_calls(cache_setup):
    """Test throttling prevents repeated Redis queries for cache misses"""
    dual_cache, _, redis_cache = cache_setup
    
    test_keys = [f"miss_{str(uuid.uuid4())}" for _ in range(3)]
    
    # Set short expiry for testing
    dual_cache.redis_batch_cache_expiry = 0.1  # 100ms
    
    with patch.object(
        redis_cache, "async_batch_get_cache", new_callable=AsyncMock
    ) as mock_redis:
        mock_redis.return_value = {key: None for key in test_keys}
        
        # First call hits Redis (no throttle data exists)
        await dual_cache.async_batch_get_cache(test_keys)
        assert mock_redis.call_count == 1
        
        # Second call immediately - throttled (within expiry window)
        await dual_cache.async_batch_get_cache(test_keys)
        assert mock_redis.call_count == 1
        
        # Verify all keys tracked in throttle cache
        for key in test_keys:
            assert key in dual_cache.last_redis_batch_access_time
        
        # Wait for expiry time to pass
        time.sleep(0.15)
        
        # Third call after expiry - call_count increases to 2
        await dual_cache.async_batch_get_cache(test_keys)
        assert mock_redis.call_count == 2


@pytest.mark.asyncio
async def test_basic_functionality_not_broken(cache_setup):
    """Ensure basic cache functionality still works after optimizations"""
    dual_cache, _, _ = cache_setup
    
    # Test basic set/get works
    test_key = f"functional_test_{str(uuid.uuid4())}"
    test_value = {"test": "data"}
    
    await dual_cache.async_set_cache(test_key, test_value)
    result = await dual_cache.async_get_cache(test_key)
    
    assert result == test_value


@pytest.mark.asyncio
async def test_batch_get_with_no_in_memory_cache():
    """Test that batch get works when in_memory_cache is None"""
    redis_cache = RedisCache(
        host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT")
    )
    
    # Create DualCache with no in-memory cache
    dual_cache = DualCache(
        in_memory_cache=None,  # This is the edge case we're testing
        redis_cache=redis_cache,
    )
    
    # Set some test data directly in Redis
    test_key = f"no_memory_test_{str(uuid.uuid4())}"
    test_value = {"test": "data_without_memory_cache"}
    
    await redis_cache.async_set_cache(test_key, test_value)
    
    # Should not crash when fetching from Redis without in-memory cache
    result = await dual_cache.async_batch_get_cache([test_key])
    
    assert result is not None
    assert len(result) == 1
    assert result[0] == test_value


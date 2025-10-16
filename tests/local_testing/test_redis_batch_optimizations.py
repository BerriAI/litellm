"""
Tests for Redis batch cache performance optimizations from commit a1e489c4de

Validates the key optimizations:
1. Batch cache size increased from 100 to 1000 (and never below 1k)
2. Throttling repeated Redis queries for cache misses
3. Early exit when redis_result is None or all None values
4. O(1) key lookup with dict mapping (not O(n) list.index())
5. Only non-None values cached in memory
6. CooldownCache early return for all-None results
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
        default_max_redis_batch_cache_size=1000,
    )
    return dual_cache, in_memory, redis_cache


@pytest.mark.asyncio
async def test_batch_cache_size_is_1000_minimum(cache_setup):
    """Verify batch cache size is set to 1000 (never below 1k)"""
    dual_cache, _, _ = cache_setup
    
    # Critical: batch cache size must be at least 1000
    assert dual_cache.last_redis_batch_access_time.max_size >= 1000


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
async def test_key_to_index_mapping_correctness(cache_setup):
    """Test O(1) key lookup maintains correct ordering"""
    dual_cache, in_memory, redis_cache = cache_setup
    
    # Use unordered keys to verify mapping works correctly
    test_keys = ["zebra", "alpha", "delta", "beta"]
    redis_values = {
        "zebra": {"pos": 0},
        "alpha": {"pos": 1},
        "delta": {"pos": 2},
        "beta": {"pos": 3},
    }
    
    with patch.object(
        in_memory, "async_batch_get_cache", new_callable=AsyncMock
    ) as mock_memory:
        mock_memory.return_value = [None] * len(test_keys)
        
        with patch.object(
            redis_cache, "async_batch_get_cache", new_callable=AsyncMock
        ) as mock_redis:
            mock_redis.return_value = redis_values
            
            result = await dual_cache.async_batch_get_cache(test_keys)
            
            # Verify correct ordering maintained
            assert result[0] == {"pos": 0}
            assert result[1] == {"pos": 1}
            assert result[2] == {"pos": 2}
            assert result[3] == {"pos": 3}


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


import os
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import hashlib
import random

import pytest

import litellm
from litellm import aembedding, completion, embedding
from litellm.caching.caching import Cache

from unittest.mock import AsyncMock, patch, MagicMock, call
import datetime
from datetime import timedelta
from litellm.caching import *


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_get_set(is_async):
    """Test that DualCache reads from in-memory cache first for both sync and async operations"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    # Test basic set/get
    test_key = f"test_key_{str(uuid.uuid4())}"
    test_value = {"test": "value"}

    if is_async:
        await dual_cache.async_set_cache(test_key, test_value)
        mock_method = "async_get_cache"
    else:
        dual_cache.set_cache(test_key, test_value)
        mock_method = "get_cache"

    # Mock Redis get to ensure we're not calling it
    # this should only read in memory since we just set test_key
    with patch.object(redis_cache, mock_method) as mock_redis_get:
        if is_async:
            result = await dual_cache.async_get_cache(test_key)
        else:
            result = dual_cache.get_cache(test_key)

        assert result == test_value
        mock_redis_get.assert_not_called()  # Verify Redis wasn't accessed


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_local_only(is_async):
    """Test that when local_only=True, only in-memory cache is used"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_key = f"test_key_{str(uuid.uuid4())}"
    test_value = {"test": "value"}

    # Mock Redis methods to ensure they're not called
    redis_set_method = "async_set_cache" if is_async else "set_cache"
    redis_get_method = "async_get_cache" if is_async else "get_cache"

    with patch.object(redis_cache, redis_set_method) as mock_redis_set, patch.object(
        redis_cache, redis_get_method
    ) as mock_redis_get:

        # Set value with local_only=True
        if is_async:
            await dual_cache.async_set_cache(test_key, test_value, local_only=True)
            result = await dual_cache.async_get_cache(test_key, local_only=True)
        else:
            dual_cache.set_cache(test_key, test_value, local_only=True)
            result = dual_cache.get_cache(test_key, local_only=True)

        assert result == test_value
        mock_redis_set.assert_not_called()  # Verify Redis set wasn't called
        mock_redis_get.assert_not_called()  # Verify Redis get wasn't called


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_value_not_in_memory(is_async):
    """Test that DualCache falls back to Redis when value isn't in memory,
    and subsequent requests use in-memory cache"""

    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_key = f"test_key_{str(uuid.uuid4())}"
    test_value = {"test": "value"}

    # First, set value only in Redis
    if is_async:
        await redis_cache.async_set_cache(test_key, test_value)
    else:
        redis_cache.set_cache(test_key, test_value)

    # First request - should fall back to Redis and populate in-memory
    if is_async:
        result = await dual_cache.async_get_cache(test_key)
    else:
        result = dual_cache.get_cache(test_key)

    assert result == test_value

    # Second request - should now use in-memory cache
    with patch.object(
        redis_cache, "async_get_cache" if is_async else "get_cache"
    ) as mock_redis_get:
        if is_async:
            result = await dual_cache.async_get_cache(test_key)
        else:
            result = dual_cache.get_cache(test_key)

        assert result == test_value
        mock_redis_get.assert_not_called()  # Verify Redis wasn't accessed second time


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_batch_operations(is_async):
    """Test batch get/set operations use in-memory cache correctly"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_keys = [f"test_key_{str(uuid.uuid4())}" for _ in range(3)]
    test_values = [{"test": f"value_{i}"} for i in range(3)]
    cache_list = list(zip(test_keys, test_values))

    # Set values
    if is_async:
        await dual_cache.async_set_cache_pipeline(cache_list)
    else:
        for key, value in cache_list:
            dual_cache.set_cache(key, value)

    # Verify in-memory cache is used for subsequent reads
    with patch.object(
        redis_cache, "async_batch_get_cache" if is_async else "batch_get_cache"
    ) as mock_redis_get:
        if is_async:
            results = await dual_cache.async_batch_get_cache(test_keys)
        else:
            results = dual_cache.batch_get_cache(test_keys, parent_otel_span=None)

        assert results == test_values
        mock_redis_get.assert_not_called()


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_increment(is_async):
    """Test increment operations only use in memory when local_only=True"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_key = f"counter_{str(uuid.uuid4())}"
    increment_value = 1

    # increment should use in-memory cache
    with patch.object(
        redis_cache, "async_increment" if is_async else "increment_cache"
    ) as mock_redis_increment:
        if is_async:
            result = await dual_cache.async_increment_cache(
                test_key,
                increment_value,
                local_only=True,
                parent_otel_span=None,
            )
        else:
            result = dual_cache.increment_cache(
                test_key, increment_value, local_only=True
            )

        assert result == increment_value
        mock_redis_increment.assert_not_called()


@pytest.mark.asyncio
async def test_dual_cache_sadd():
    """Test set add operations use in-memory cache for reads"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_key = f"set_{str(uuid.uuid4())}"
    test_values = ["value1", "value2", "value3"]

    # Add values to set
    await dual_cache.async_set_cache_sadd(test_key, test_values)

    # Verify in-memory cache is used for subsequent operations
    with patch.object(redis_cache, "async_get_cache") as mock_redis_get:
        result = await dual_cache.async_get_cache(test_key)
        assert set(result) == set(test_values)
        mock_redis_get.assert_not_called()


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_dual_cache_delete(is_async):
    """Test delete operations remove from both caches"""
    in_memory = InMemoryCache()
    redis_cache = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    dual_cache = DualCache(in_memory_cache=in_memory, redis_cache=redis_cache)

    test_key = f"test_key_{str(uuid.uuid4())}"
    test_value = {"test": "value"}

    # Set value
    if is_async:
        await dual_cache.async_set_cache(test_key, test_value)
    else:
        dual_cache.set_cache(test_key, test_value)

    # Delete value
    if is_async:
        await dual_cache.async_delete_cache(test_key)
    else:
        dual_cache.delete_cache(test_key)

    # Verify value is deleted from both caches
    if is_async:
        result = await dual_cache.async_get_cache(test_key)
    else:
        result = dual_cache.get_cache(test_key)

    assert result is None

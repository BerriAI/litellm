import os
import sys
import time
import traceback
import uuid

from dotenv import load_dotenv
from test_rerank import assert_response_shape

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

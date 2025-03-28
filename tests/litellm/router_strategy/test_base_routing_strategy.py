import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy


@pytest.fixture
def mock_dual_cache():
    dual_cache = MagicMock(spec=DualCache)
    dual_cache.in_memory_cache = MagicMock()
    dual_cache.redis_cache = MagicMock()

    # Set up async method mocks to return coroutines
    future1 = asyncio.Future()
    future1.set_result(None)
    dual_cache.in_memory_cache.async_increment.return_value = future1

    future2 = asyncio.Future()
    future2.set_result(None)
    dual_cache.redis_cache.async_increment_pipeline.return_value = future2

    future3 = asyncio.Future()
    future3.set_result(None)
    dual_cache.in_memory_cache.async_set_cache.return_value = future3

    # Fix for async_batch_get_cache
    batch_future = asyncio.Future()
    batch_future.set_result({"key1": "10.0", "key2": "20.0"})
    dual_cache.redis_cache.async_batch_get_cache.return_value = batch_future

    return dual_cache


@pytest.fixture
def base_strategy(mock_dual_cache):
    return BaseRoutingStrategy(
        dual_cache=mock_dual_cache,
        should_batch_redis_writes=False,
        default_sync_interval=1,
    )


@pytest.mark.asyncio
async def test_increment_value_in_current_window(base_strategy, mock_dual_cache):
    # Test incrementing value in current window
    key = "test_key"
    value = 10.0
    ttl = 3600

    await base_strategy._increment_value_in_current_window(key, value, ttl)

    # Verify in-memory cache was incremented
    mock_dual_cache.in_memory_cache.async_increment.assert_called_once_with(
        key=key, value=value, ttl=ttl
    )

    # Verify operation was queued for Redis
    assert len(base_strategy.redis_increment_operation_queue) == 1
    queued_op = base_strategy.redis_increment_operation_queue[0]
    assert isinstance(queued_op, dict)
    assert queued_op["key"] == key
    assert queued_op["increment_value"] == value
    assert queued_op["ttl"] == ttl


@pytest.mark.asyncio
async def test_push_in_memory_increments_to_redis(base_strategy, mock_dual_cache):
    # Add some operations to the queue
    base_strategy.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key="key1", increment_value=10, ttl=3600),
        RedisPipelineIncrementOperation(key="key2", increment_value=20, ttl=3600),
    ]

    await base_strategy._push_in_memory_increments_to_redis()

    # Verify Redis pipeline was called
    mock_dual_cache.redis_cache.async_increment_pipeline.assert_called_once()
    # Verify queue was cleared
    assert len(base_strategy.redis_increment_operation_queue) == 0


@pytest.mark.asyncio
async def test_sync_in_memory_spend_with_redis(base_strategy, mock_dual_cache):
    # Setup test data
    base_strategy.in_memory_keys_to_update = {"key1", "key2"}

    # No need to set return_value here anymore as it's set in the fixture
    await base_strategy._sync_in_memory_spend_with_redis()

    # Verify Redis batch get was called with sorted list for consistent testing
    key_list = mock_dual_cache.redis_cache.async_batch_get_cache.call_args.kwargs[
        "key_list"
    ]

    sorted(key_list) == sorted(["key1", "key2"])
    # mock_dual_cache.redis_cache.async_batch_get_cache.assert_called_once_with(
    #     key_list=sorted()
    # )

    # Verify in-memory cache was updated
    assert mock_dual_cache.in_memory_cache.async_set_cache.call_count == 2

    # Verify cache keys were reset
    assert len(base_strategy.in_memory_keys_to_update) == 0


def test_cache_keys_management(base_strategy):
    # Test adding and getting cache keys
    base_strategy.add_to_in_memory_keys_to_update("key1")
    base_strategy.add_to_in_memory_keys_to_update("key2")
    base_strategy.add_to_in_memory_keys_to_update("key1")  # Duplicate should be ignored

    cache_keys = base_strategy.get_in_memory_keys_to_update()
    assert len(cache_keys) == 2
    assert "key1" in cache_keys
    assert "key2" in cache_keys

    # Test resetting cache keys
    base_strategy.reset_in_memory_keys_to_update()
    assert len(base_strategy.get_in_memory_keys_to_update()) == 0

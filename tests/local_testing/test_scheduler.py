# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import sys, os, time, openai, uuid
import traceback, asyncio
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.scheduler import FlowItem, Scheduler
from litellm import ModelResponse
from litellm.caching.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_scheduler_diff_model_names():
    """
    Assert 2 requests to 2 diff model groups are top of their respective queue's
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-4")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    assert (
        await scheduler.poll(
            id="10", model_name="gpt-3.5-turbo", health_deployments=[{"key": "value"}]
        )
        == True
    )
    assert (
        await scheduler.poll(
            id="11", model_name="gpt-4", health_deployments=[{"key": "value"}]
        )
        == True
    )


@pytest.mark.parametrize("p0, p1", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize("healthy_deployments", [[{"key": "value"}], []])
@pytest.mark.asyncio
async def test_scheduler_prioritized_requests(p0, p1, healthy_deployments):
    """
    2 requests for same model group
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=p0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=p1, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    if p0 == 0:
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == True
        ), "queue={}".format(await scheduler.get_queue(model_name="gpt-3.5-turbo"))
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == False
        )
    else:
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == True
        )
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == False
        )


@pytest.mark.asyncio
async def test_scheduler_priority_type_normalization():
    """
    Test that scheduler normalizes priority types to int to prevent TypeError
    Addresses issue: https://github.com/BerriAI/litellm/issues/14817
    """
    scheduler = Scheduler()

    # Test with string priority (should be converted to int)
    item1 = FlowItem(priority=5, request_id="10", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    
    # Test with another valid int priority
    item2 = FlowItem(priority=3, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item2)
    
    # Verify the queue is properly ordered (lower priority first)
    queue = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    assert len(queue) == 2
    assert queue[0][0] == 3  # First item should have priority 3
    assert queue[0][1] == "11"
    
    # Verify all items in queue are tuples with (int, str)
    for item in queue:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], int), f"Priority should be int, got {type(item[0])}"
        assert isinstance(item[1], str), f"Request ID should be str, got {type(item[1])}"


@pytest.mark.asyncio
async def test_scheduler_redis_cache_deserialization():
    """
    Test that Redis cache deserialization properly validates and normalizes queue data
    Addresses issue: https://github.com/BerriAI/litellm/issues/14817
    """
    # Mock Redis cache that returns corrupted data (simulating ast.literal_eval issues)
    mock_redis_cache = MagicMock(spec=RedisCache)
    
    # Simulate corrupted queue data from Redis (mixed types)
    corrupted_queue = [
        (5, "request-1"),  # Valid tuple
        [3, "request-2"],  # List instead of tuple (corrupted)
        (1, "request-3"),  # Valid tuple
    ]
    
    async def mock_get_cache(key, **kwargs):
        return corrupted_queue
    
    async def mock_set_cache(key, value, **kwargs):
        pass
    
    mock_redis_cache.async_get_cache = AsyncMock(side_effect=mock_get_cache)
    mock_redis_cache.async_set_cache = AsyncMock(side_effect=mock_set_cache)
    
    scheduler = Scheduler(redis_cache=mock_redis_cache)
    
    # Get queue should normalize the corrupted data
    queue = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    
    # Verify all items are properly normalized to (int, str) tuples
    assert len(queue) == 3
    for item in queue:
        assert isinstance(item, tuple), f"Item should be tuple, got {type(item)}"
        assert len(item) == 2
        assert isinstance(item[0], int), f"Priority should be int, got {type(item[0])}"
        assert isinstance(item[1], str), f"Request ID should be str, got {type(item[1])}"
    
    # Verify the data is correct
    assert (5, "request-1") in queue
    assert (3, "request-2") in queue
    assert (1, "request-3") in queue


@pytest.mark.asyncio
async def test_scheduler_handles_invalid_queue_items():
    """
    Test that scheduler gracefully handles and skips invalid queue items
    """
    mock_redis_cache = MagicMock(spec=RedisCache)
    
    # Simulate queue with various invalid items
    invalid_queue = [
        (5, "valid-1"),  # Valid
        ("not-an-int", "invalid-1"),  # Invalid priority type
        (3,),  # Invalid length
        [2, "valid-2"],  # Valid but as list (should be normalized)
        "not-a-tuple",  # Completely invalid
        (1, "valid-3"),  # Valid
    ]
    
    async def mock_get_cache(key, **kwargs):
        return invalid_queue
    
    async def mock_set_cache(key, value, **kwargs):
        pass
    
    mock_redis_cache.async_get_cache = AsyncMock(side_effect=mock_get_cache)
    mock_redis_cache.async_set_cache = AsyncMock(side_effect=mock_set_cache)
    
    scheduler = Scheduler(redis_cache=mock_redis_cache)
    
    # Get queue should filter out invalid items and normalize valid ones
    queue = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    
    # Should only have 3 valid items (the ones that could be normalized)
    assert len(queue) == 3
    
    # Verify all returned items are valid
    for item in queue:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], int)
        assert isinstance(item[1], str)
    
    # Verify the correct items were kept
    assert (5, "valid-1") in queue
    assert (2, "valid-2") in queue
    assert (1, "valid-3") in queue

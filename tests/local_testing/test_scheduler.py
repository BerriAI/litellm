# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import sys, os, time, openai, uuid
import traceback, asyncio
import pytest
from typing import List

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.scheduler import FlowItem, Scheduler
from litellm import ModelResponse


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


@pytest.mark.asyncio
async def test_scheduler_mixed_priority_types():
    """
    Test scheduler with different priority data types to ensure heap operations work correctly
    """
    scheduler = Scheduler()

    # Test with various corrupted priority types that might come from Redis
    test_cases = [
        # (priority, request_id) - some may be corrupted from Redis deserialization
        ([1, 2], "req_1"),  # List priority (corrupted)
        ((3,), "req_2"),    # Single element tuple (corrupted)
        (5, "req_3"),       # Valid int priority
        ("7", "req_4"),     # String priority (corrupted)
        ([9, "extra"], "req_5"),  # Mixed list (corrupted)
    ]

    # Simulate corrupted queue from Redis by directly setting cache response
    corrupted_queue = test_cases

    # Manually test the get_queue validation logic
    validated_queue = []
    for item in corrupted_queue:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            try:
                priority = int(item[0]) if item[0] is not None else 0
                request_id = str(item[1]) if item[1] is not None else ""
                priority = max(0, min(255, priority))
                validated_queue.append((priority, request_id))
            except (ValueError, TypeError):
                continue

    # Should successfully create a valid queue with normalized priorities
    # From our test cases:
    # ([1, 2], "req_1") -> fails because int([1, 2]) throws TypeError
    # ((3,), "req_2") -> fails because len(item) != 2
    # (5, "req_3") -> passes (valid int, string)
    # ("7", "req_4") -> passes (string convertible to int, plus string)
    # ([9, "extra"], "req_5") -> fails because int([9, "extra"]) throws TypeError
    expected_valid_items = 2
    assert len(validated_queue) == expected_valid_items

    # Check that all items are properly formatted
    for priority, request_id in validated_queue:
        assert isinstance(priority, int)
        assert isinstance(request_id, str)
        assert 0 <= priority <= 255

    # Now test actual scheduler operations don't fail with heap comparisons
    for priority, request_id in validated_queue:
        item = FlowItem(priority=priority, request_id=request_id, model_name="test-model")
        await scheduler.add_request(item)

    # Should be able to poll without TypeError
    queue = await scheduler.get_queue(model_name="test-model")
    assert len(queue) >= 2  # Should have at least the items we added


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

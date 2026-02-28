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
from litellm.scheduler import FlowItem, Scheduler, SchedulerCacheKeys
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
async def test_scheduler_poll_persists_queue_to_cache():
    class StubRedisCache:
        def __init__(self):
            self.store = {}

        async def async_get_cache(self, key, **kwargs):
            return self.store.get(key)

        async def async_set_cache(self, key, value, **kwargs):
            self.store[key] = value

    redis_cache = StubRedisCache()
    scheduler = Scheduler(redis_cache=redis_cache)

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    await scheduler.poll(
        id="10", model_name="gpt-3.5-turbo", health_deployments=[]
    )

    queue_key = f"{SchedulerCacheKeys.queue.value}:{item1.model_name}"
    updated_queue = redis_cache.store[queue_key]
    assert updated_queue[0][1] == "11"


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
async def test_scheduler_queue_cleanup_on_timeout():
    """
    Test that a timed-out request is properly removed from the queue.
    This prevents memory leaks from accumulating timed-out requests.
    """
    scheduler = Scheduler()

    # Add multiple requests with different priorities
    item1 = FlowItem(priority=0, request_id="req-0", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=1, request_id="req-1", model_name="gpt-3.5-turbo")
    item3 = FlowItem(priority=2, request_id="req-2", model_name="gpt-3.5-turbo")

    await scheduler.add_request(item1)
    await scheduler.add_request(item2)
    await scheduler.add_request(item3)

    # Verify initial queue size
    queue_before = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    assert len(queue_before) == 3, f"Expected 3 items in queue, got {len(queue_before)}"

    # Simulate timeout cleanup - remove a non-front request (item2)
    await scheduler.remove_request(request_id="req-1", model_name="gpt-3.5-turbo")

    # Verify queue was cleaned up
    queue_after = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    assert len(queue_after) == 2, f"Expected 2 items after cleanup, got {len(queue_after)}"

    # Verify the correct request was removed
    remaining_ids = [item[1] for item in queue_after]
    assert "req-1" not in remaining_ids, "Expected req-1 to be removed"
    assert "req-0" in remaining_ids, "Expected req-0 to remain"
    assert "req-2" in remaining_ids, "Expected req-2 to remain"

    # Verify remaining items are in correct priority order (0 should be first)
    assert queue_after[0][1] == "req-0", "Expected req-0 (priority 0) to be at front"


@pytest.mark.asyncio
async def test_scheduler_concurrent_poll_allows_single_request_per_interval():
    scheduler = Scheduler(polling_interval=1)
    model_name = "gpt-3.5-turbo"

    for i in range(5):
        await scheduler.add_request(
            FlowItem(
                priority=i,
                request_id=f"req-{i}",
                model_name=model_name,
            )
        )

    start_event = asyncio.Event()

    async def _poll(req_id: str) -> bool:
        await start_event.wait()
        return await scheduler.poll(
            id=req_id,
            model_name=model_name,
            health_deployments=[],
        )

    tasks = [asyncio.create_task(_poll(f"req-{i}")) for i in range(5)]
    start_event.set()
    poll_results = await asyncio.gather(*tasks)

    allowed_count = sum(1 for result in poll_results if result)
    assert allowed_count == 1, f"Expected 1 admitted request, got {allowed_count}"

    queue_after = await scheduler.get_queue(model_name=model_name)
    assert len(queue_after) == 4, f"Expected 4 items remaining, got {len(queue_after)}"


@pytest.mark.asyncio
async def test_scheduler_poll_allows_next_request_after_polling_interval(monkeypatch):
    fake_time = {"now": 100.0}

    def _fake_monotonic():
        return fake_time["now"]

    monkeypatch.setattr("litellm.scheduler.time.monotonic", _fake_monotonic)

    scheduler = Scheduler(polling_interval=1)
    model_name = "gpt-3.5-turbo"

    await scheduler.add_request(
        FlowItem(priority=0, request_id="req-0", model_name=model_name)
    )
    await scheduler.add_request(
        FlowItem(priority=1, request_id="req-1", model_name=model_name)
    )

    first_poll = await scheduler.poll(
        id="req-0",
        model_name=model_name,
        health_deployments=[],
    )
    assert first_poll is True

    second_poll_immediate = await scheduler.poll(
        id="req-1",
        model_name=model_name,
        health_deployments=[],
    )
    assert second_poll_immediate is False

    fake_time["now"] = fake_time["now"] + 2

    second_poll_after_wait = await scheduler.poll(
        id="req-1",
        model_name=model_name,
        health_deployments=[],
    )
    assert second_poll_after_wait is True


@pytest.mark.asyncio
async def test_scheduler_get_queue_returns_copy():
    scheduler = Scheduler()
    model_name = "gpt-3.5-turbo"

    await scheduler.add_request(
        FlowItem(priority=0, request_id="req-0", model_name=model_name)
    )

    queue_a = await scheduler.get_queue(model_name=model_name)
    queue_b = await scheduler.get_queue(model_name=model_name)

    assert queue_a == queue_b
    assert queue_a is not queue_b

    queue_a.append((999, "mutated"))
    queue_after_mutation = await scheduler.get_queue(model_name=model_name)
    assert (999, "mutated") not in queue_after_mutation


@pytest.mark.asyncio
async def test_scheduler_normalizes_cached_queue_items_to_tuples():
    class StubRedisCache:
        def __init__(self):
            self.store = {}

        async def async_get_cache(self, key, **kwargs):
            return self.store.get(key)

        async def async_set_cache(self, key, value, **kwargs):
            self.store[key] = value

    redis_cache = StubRedisCache()
    scheduler = Scheduler(redis_cache=redis_cache)
    model_name = "gpt-3.5-turbo"
    queue_key = f"{SchedulerCacheKeys.queue.value}:{model_name}"

    # Simulate Redis JSON round-trip shape: list-of-lists.
    redis_cache.store[queue_key] = [[0, "req-0"]]

    await scheduler.add_request(
        FlowItem(priority=1, request_id="req-1", model_name=model_name)
    )

    queue = await scheduler.get_queue(model_name=model_name)
    assert queue == [(0, "req-0"), (1, "req-1")]

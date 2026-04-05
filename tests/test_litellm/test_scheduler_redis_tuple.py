"""
Test that Scheduler.get_queue normalizes JSON-deserialized lists back to tuples.

When Redis is used as the cache backend, JSON serialization converts tuples
to lists. heapq requires consistent types for comparison, so get_queue must
normalize items back to tuples.

Regression test for https://github.com/BerriAI/litellm/issues/25157
"""

import pytest

from litellm.scheduler import FlowItem, Scheduler


class FakeRedisCache:
    """Minimal stub that simulates Redis returning JSON-deserialized data."""

    def __init__(self):
        self.store: dict = {}

    async def async_get_cache(self, key, **kwargs):
        return self.store.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        # Simulate JSON round-trip: tuples become lists
        import json

        self.store[key] = json.loads(json.dumps(value))


@pytest.mark.asyncio
async def test_scheduler_add_request_after_redis_roundtrip():
    """
    After a Redis round-trip, queue items are lists (not tuples).
    Adding a new request should not raise TypeError from heapq comparison.
    """
    fake_redis = FakeRedisCache()
    scheduler = Scheduler(redis_cache=fake_redis)

    # First request — goes into an empty queue, no comparison needed
    item1 = FlowItem(priority=0, request_id="req-1", model_name="test-model")
    await scheduler.add_request(item1)

    # Second request — queue from cache has list items, heappush must compare
    # Without the fix, this raises:
    #   TypeError: '<' not supported between instances of 'tuple' and 'list'
    item2 = FlowItem(priority=1, request_id="req-2", model_name="test-model")
    await scheduler.add_request(item2)

    # Verify both items are in the queue
    queue = await scheduler.get_queue(model_name="test-model")
    request_ids = {item[1] for item in queue}
    assert "req-1" in request_ids
    assert "req-2" in request_ids

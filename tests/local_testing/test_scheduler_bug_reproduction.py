# What is this?
## Test to reproduce and verify fix for scheduler priority type mixing bug
## Issue: TypeError: '<' not supported between instances of 'tuple' and 'list' during heap operations

import sys, os
import pytest
import asyncio
import heapq
from unittest.mock import patch, AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router
from litellm.scheduler import FlowItem, Scheduler
from litellm.caching.redis_cache import RedisCache


class TestSchedulerBugReproduction:
    """Test reproduction and fix validation for the scheduler priority type mixing bug"""

    @pytest.mark.asyncio
    async def test_reproduce_original_bug_without_fix(self):
        """
        Reproduce the original TypeError that occurred with mixed priority types
        This test demonstrates what would happen without our fixes
        """
        # Create corrupted queue data that would come from Redis ast.literal_eval fallback
        corrupted_queue_data = [
            (5, "req_1"),           # Valid int priority
            ([1, 2], "req_2"),      # List priority (corrupted from Redis)
            ((3, 4), "req_3"),      # Tuple priority (corrupted from Redis)
            ("7", "req_4"),         # String priority
        ]

        # This would cause TypeError in heap operations without our fix
        heap_queue = []

        # Try to add items directly to heap without validation (original behavior)
        with pytest.raises((TypeError, ValueError)):
            for priority, request_id in corrupted_queue_data:
                # This is what the original code did - directly push to heap
                # It would fail with TypeError: '<' not supported between instances
                heapq.heappush(heap_queue, (priority, request_id))

    @pytest.mark.asyncio
    async def test_scheduler_handles_corrupted_redis_data(self):
        """
        Test that our scheduler now handles corrupted Redis data gracefully
        """
        scheduler = Scheduler()

        # Simulate corrupted data that might come from Redis deserialization
        corrupted_data = [
            ([1, 2], "req_1"),      # List priority (would cause TypeError)
            ((3, 4), "req_2"),      # Tuple priority (would cause TypeError)
            (5, "req_3"),           # Valid int priority
            ("7", "req_4"),         # String priority
            ([9, "extra", "data"], "req_5"),  # Complex list (would cause TypeError)
            (None, "req_6"),        # None priority
            ({"priority": 10}, "req_7"),  # Dict priority (would cause TypeError)
        ]

        # Mock the cache to return corrupted data
        mock_cache = AsyncMock()
        mock_cache.async_get_cache.return_value = corrupted_data
        scheduler.cache = mock_cache

        # Our fixed get_queue should handle this gracefully
        queue = await scheduler.get_queue("test-model")

        # Should successfully return a validated queue with only valid items
        assert isinstance(queue, list)

        # All items should be properly formatted (int, str) tuples
        for item in queue:
            assert isinstance(item, tuple)
            assert len(item) == 2
            priority, request_id = item
            assert isinstance(priority, int)
            assert isinstance(request_id, str)
            assert 0 <= priority <= 255

        # Should be able to add items to scheduler without TypeError
        for priority, request_id in queue:
            item = FlowItem(priority=priority, request_id=request_id, model_name="test-model")
            await scheduler.add_request(item)  # Should not raise TypeError

    @pytest.mark.asyncio
    async def test_redis_cache_deserialization_fix(self):
        """
        Test that Redis cache deserialization now handles corrupted data properly
        """
        # Create a minimal RedisCache instance for testing deserialization logic
        redis_cache = RedisCache.__new__(RedisCache)  # Create without calling __init__

        # Test various corrupted data scenarios that could come from ast.literal_eval
        test_cases = [
            # Valid queue data
            b'[(1, "req_1"), (2, "req_2")]',

            # Corrupted queue data that would cause issues
            b'[([1, 2], "req_1"), (5, "req_2")]',  # Mixed types
            b'[((1, 2, 3), "req_1"), (5, "req_2")]',  # Invalid tuple length
            b'[(1, "req_1"), ("invalid", "req_2")]',  # String priority
        ]

        for test_data in test_cases:
            # Should not raise exception and should return valid data or None
            result = redis_cache._get_cache_logic(test_data)

            if result is not None:
                assert isinstance(result, list)
                # If it's a queue-like structure, all items should be valid
                if result and isinstance(result[0], tuple) and len(result[0]) == 2:
                    for item in result:
                        if isinstance(item, tuple) and len(item) == 2:
                            priority, request_id = item
                            assert isinstance(priority, int)
                            assert isinstance(request_id, str)

    @pytest.mark.asyncio
    async def test_router_priority_normalization_prevents_bug(self):
        """
        Test that router priority normalization prevents malformed data from reaching scheduler
        """
        router = Router(
            model_list=[{
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake_key"}
            }],
            default_priority=100
        )

        # Test various problematic priority values that clients might send
        problematic_priorities = [
            [1, 2, 3],           # List priority
            (5, 6),              # Tuple priority
            {"priority": 10},     # Dict priority
            "not_a_number",      # Invalid string
            None,                # None priority
            float('inf'),        # Invalid float
        ]

        for priority in problematic_priorities:
            # Should normalize to valid integer or default
            normalized = router._normalize_priority(priority, router.default_priority)

            if normalized is not None:
                assert isinstance(normalized, int)
                assert 0 <= normalized <= 255

            # If normalized is None and we have a default, that's also valid
            # The router will handle None appropriately

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_mixed_priorities(self):
        """
        Test the original scenario: concurrent requests with mixed priority types
        This reproduces the exact conditions described in the issue
        """
        scheduler = Scheduler()

        # Simulate concurrent requests with different priority field types
        request_scenarios = [
            # Request 1: Valid priority
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "priority": 5},

            # Request 2: Invalid priority (list) - this would cause the original bug
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "priority": [1, 2, 3]},

            # Request 3: Invalid priority (tuple) - this would cause the original bug
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "priority": (1, 2)},

            # Request 4: No priority
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},

            # Request 5: String priority
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "priority": "high"},
        ]

        # Simulate processing these requests concurrently
        tasks = []
        for i, request_data in enumerate(request_scenarios):
            # Extract and normalize priority (this is what our fix does)
            raw_priority = request_data.get("priority")

            # Use our normalization logic
            if raw_priority is None:
                normalized_priority = None
            else:
                try:
                    if isinstance(raw_priority, int):
                        normalized_priority = max(0, min(255, raw_priority))
                    elif isinstance(raw_priority, (float)):
                        normalized_priority = max(0, min(255, int(raw_priority)))
                    elif isinstance(raw_priority, str):
                        try:
                            parsed = float(raw_priority.strip())
                            normalized_priority = max(0, min(255, int(parsed)))
                        except:
                            normalized_priority = 100  # default
                    elif isinstance(raw_priority, (list, tuple)):
                        # Try first numeric element
                        for item in raw_priority:
                            try:
                                if isinstance(item, (int, float)):
                                    normalized_priority = max(0, min(255, int(item)))
                                    break
                            except:
                                continue
                        else:
                            normalized_priority = 100  # default
                    else:
                        normalized_priority = 100  # default
                except:
                    normalized_priority = 100  # default

            if normalized_priority is not None:
                # Create FlowItem with normalized priority
                item = FlowItem(
                    priority=normalized_priority,
                    request_id=f"req_{i}",
                    model_name="gpt-4"
                )
                # Should not raise TypeError during heap operations
                task = scheduler.add_request(item)
                tasks.append(task)

        # All requests should be added successfully without TypeError
        await asyncio.gather(*tasks)

        # Verify queue operations work correctly
        queue = await scheduler.get_queue("gpt-4")
        assert isinstance(queue, list)
        assert len(queue) > 0

        # All queue items should have correct format
        for item in queue:
            assert isinstance(item, tuple)
            assert len(item) == 2
            priority, request_id = item
            assert isinstance(priority, int)
            assert isinstance(request_id, str)

    def test_heap_operations_with_mixed_types_original_error(self):
        """
        Demonstrate the original error that would occur in heap operations
        """
        # This is what would happen in the original code with corrupted Redis data
        mixed_queue = [
            (5, "req_1"),           # int priority
            ([1, 2], "req_2"),      # list priority (corrupted)
            ((3, 4), "req_3"),      # tuple priority (corrupted)
        ]

        heap_queue = []

        # The first item works fine
        heapq.heappush(heap_queue, mixed_queue[0])  # (5, "req_1")

        # The second item would cause TypeError in original code
        with pytest.raises(TypeError, match="'<' not supported between instances"):
            heapq.heappush(heap_queue, mixed_queue[1])  # ([1, 2], "req_2")

    @pytest.mark.asyncio
    async def test_end_to_end_bug_fix_validation(self):
        """
        End-to-end test that validates the complete fix works as intended
        """
        # Create router with scheduler
        router = Router(
            model_list=[{
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake_key"}
            }],
            default_priority=50
        )

        # Test requests that would have caused the original bug
        test_requests = [
            {"priority": 5},                    # Valid
            {"priority": [1, 2, 3]},           # Would cause TypeError
            {"priority": (3, 4)},              # Would cause TypeError
            {"priority": "invalid"},           # Would cause issues
            {"priority": None},                # Edge case
            {},                                # No priority
        ]

        # Process all requests - should complete without errors
        for i, request_kwargs in enumerate(test_requests):
            # This simulates what happens in router.acompletion
            raw_priority = request_kwargs.get("priority")
            normalized_priority = router._normalize_priority(raw_priority, router.default_priority)

            # Should always get a valid priority or None
            if normalized_priority is not None:
                assert isinstance(normalized_priority, int)
                assert 0 <= normalized_priority <= 255

                # Create scheduler item (this is what caused the original error)
                item = FlowItem(
                    priority=normalized_priority,
                    request_id=f"test_req_{i}",
                    model_name="gpt-3.5-turbo"
                )

                # Add to scheduler - should not raise TypeError
                await router.scheduler.add_request(item)

        # Verify scheduler queue is healthy
        queue = await router.scheduler.get_queue("gpt-3.5-turbo")
        assert isinstance(queue, list)

        # All items should be properly formatted for heap operations
        for item in queue:
            priority, request_id = item
            assert isinstance(priority, int)
            assert isinstance(request_id, str)
            assert 0 <= priority <= 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
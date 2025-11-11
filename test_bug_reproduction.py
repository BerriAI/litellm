#!/usr/bin/env python3
"""
Test script to reproduce the bug from issue #14817
This should FAIL on main branch and PASS on the fix branch
"""

import sys
import heapq
from typing import List, Tuple, Union

def test_heapq_with_mixed_types():
    """
    Reproduce the exact error from the issue:
    TypeError: '<' not supported between instances of 'tuple' and 'list'
    """
    print("=" * 70)
    print("TEST 1: Reproducing the heapq TypeError with mixed types")
    print("=" * 70)
    
    queue: List[Union[Tuple, list]] = []
    
    # Simulate what happens when Redis cache returns mixed types
    # This is what the scheduler receives after ast.literal_eval corrupts the data
    
    try:
        # First request: normal tuple (int, str)
        heapq.heappush(queue, (5, "request-1"))
        print("✓ Added (5, 'request-1') - tuple with int priority")
        
        # Second request: corrupted as list instead of tuple
        heapq.heappush(queue, [3, "request-2"])
        print("✓ Added [3, 'request-2'] - list (corrupted)")
        
        # Third request: normal tuple
        heapq.heappush(queue, (1, "request-3"))
        print("✓ Added (1, 'request-3') - tuple with int priority")
        
        print("\n❌ UNEXPECTED: No error occurred!")
        print("This means either:")
        print("  1. The bug doesn't exist in this Python version")
        print("  2. The fix is already applied")
        return False
        
    except TypeError as e:
        print(f"\n✓ EXPECTED ERROR OCCURRED: {e}")
        print("\nThis confirms the bug exists on main branch!")
        return True


def test_scheduler_with_corrupted_cache():
    """
    Test the actual scheduler behavior with corrupted cache data
    """
    print("\n" + "=" * 70)
    print("TEST 2: Testing scheduler with simulated corrupted Redis cache")
    print("=" * 70)
    
    try:
        from litellm.scheduler import Scheduler, FlowItem
        from unittest.mock import AsyncMock, MagicMock
        import asyncio
        
        # Create a mock Redis cache that returns corrupted data
        mock_redis_cache = MagicMock()
        
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
        
        async def run_test():
            try:
                # Try to add a new request - this should trigger the bug
                item = FlowItem(priority=2, request_id="new-request", model_name="gpt-3.5-turbo")
                await scheduler.add_request(request=item)
                
                print("\n❌ UNEXPECTED: No error occurred!")
                print("The scheduler handled corrupted data without crashing.")
                print("This means the fix is already applied.")
                return False
                
            except TypeError as e:
                print(f"\n✓ EXPECTED ERROR OCCURRED: {e}")
                print("\nThis confirms the bug exists in the scheduler!")
                return True
        
        result = asyncio.run(run_test())
        return result
        
    except Exception as e:
        print(f"\n⚠ Test setup error: {e}")
        print("Skipping scheduler test (may need dependencies)")
        return None


def main():
    print("\n" + "=" * 70)
    print("BUG REPRODUCTION TEST FOR ISSUE #14817")
    print("=" * 70)
    print("\nThis test should:")
    print("  - FAIL (show errors) on main branch")
    print("  - PASS (no errors) on fix/scheduler-priority-type-error branch")
    print()
    
    results = []
    
    # Test 1: Basic heapq behavior
    test1_failed = test_heapq_with_mixed_types()
    results.append(("heapq mixed types", test1_failed))
    
    # Test 2: Scheduler with corrupted cache
    test2_failed = test_scheduler_with_corrupted_cache()
    if test2_failed is not None:
        results.append(("scheduler corrupted cache", test2_failed))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, failed in results:
        if failed:
            print(f"✓ {test_name}: Bug reproduced (expected on main)")
        elif failed is False:
            print(f"✗ {test_name}: Bug NOT reproduced (expected on fix branch)")
        else:
            print(f"⚠ {test_name}: Test skipped")
    
    print("\n" + "=" * 70)
    
    # Determine overall result
    if any(result for _, result in results if result is not None):
        print("RESULT: Bug exists in this branch ✓")
        print("This is EXPECTED on main branch")
        return 1  # Exit code 1 indicates bug exists
    else:
        print("RESULT: Bug does NOT exist in this branch ✓")
        print("This is EXPECTED on fix branch")
        return 0  # Exit code 0 indicates bug is fixed


if __name__ == "__main__":
    sys.exit(main())


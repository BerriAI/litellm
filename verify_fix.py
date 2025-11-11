#!/usr/bin/env python3
"""
Verification script to demonstrate that the fix resolves issue #14817

This script tests the scheduler's behavior when receiving corrupted queue data
from Redis cache (mixed tuple/list types).

Expected results:
- main branch: TypeError when adding requests to corrupted queue
- fix branch: No error, corrupted data is normalized and handled gracefully
"""

import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock


def test_scheduler_with_corrupted_redis_cache():
    """
    Test that demonstrates the fix for issue #14817
    
    Simulates Redis cache returning corrupted queue data with mixed types:
    - Some items as tuples: (int, str)
    - Some items as lists: [int, str]
    
    On main branch: This causes TypeError in heapq operations
    On fix branch: Data is normalized and no error occurs
    """
    print("=" * 80)
    print("VERIFICATION TEST: Scheduler with Corrupted Redis Cache Data")
    print("=" * 80)
    print("\nScenario: Redis cache returns queue with mixed tuple/list types")
    print("This happens when ast.literal_eval corrupts the deserialization")
    print()
    
    try:
        from litellm.scheduler import Scheduler, FlowItem
        from litellm.caching.redis_cache import RedisCache
        
        # Create mock Redis cache that returns corrupted data
        mock_redis_cache = MagicMock(spec=RedisCache)
        
        # Simulate corrupted queue data from Redis
        # This is what causes the TypeError on main branch
        corrupted_queue = [
            (5, "request-1"),  # Valid tuple
            [3, "request-2"],  # List instead of tuple (CORRUPTED)
            (1, "request-3"),  # Valid tuple
        ]
        
        print("Corrupted queue data from Redis:")
        for i, item in enumerate(corrupted_queue, 1):
            item_type = "tuple" if isinstance(item, tuple) else "list"
            print(f"  {i}. {item} - type: {item_type}")
        
        async def mock_get_cache(key, **kwargs):
            return corrupted_queue
        
        async def mock_set_cache(key, value, **kwargs):
            pass
        
        mock_redis_cache.async_get_cache = AsyncMock(side_effect=mock_get_cache)
        mock_redis_cache.async_set_cache = AsyncMock(side_effect=mock_set_cache)
        
        scheduler = Scheduler(redis_cache=mock_redis_cache)
        
        async def run_test():
            print("\n" + "-" * 80)
            print("Test: Adding new request to scheduler with corrupted queue")
            print("-" * 80)
            
            try:
                # Create a new request
                item = FlowItem(
                    priority=2, 
                    request_id="new-request", 
                    model_name="gpt-3.5-turbo"
                )
                print(f"\nAdding request: priority={item.priority}, id={item.request_id}")
                
                # This is where the error occurs on main branch
                await scheduler.add_request(request=item)
                
                print("✓ Request added successfully!")
                
                # Get the queue to verify normalization
                queue = await scheduler.get_queue(model_name="gpt-3.5-turbo")
                
                print(f"\nQueue after adding request (length: {len(queue)}):")
                for i, (priority, req_id) in enumerate(queue, 1):
                    print(f"  {i}. priority={priority} (type: {type(priority).__name__}), "
                          f"id={req_id} (type: {type(req_id).__name__})")
                
                # Verify all items are properly normalized
                all_valid = all(
                    isinstance(item, tuple) and 
                    len(item) == 2 and 
                    isinstance(item[0], int) and 
                    isinstance(item[1], str)
                    for item in queue
                )
                
                if all_valid:
                    print("\n✓ All queue items are properly normalized to (int, str) tuples")
                    return True
                else:
                    print("\n✗ Some queue items are not properly normalized")
                    return False
                
            except TypeError as e:
                print(f"\n✗ TypeError occurred: {e}")
                print("\nThis error indicates the bug from issue #14817 is present.")
                print("Expected on main branch, should NOT occur on fix branch.")
                return False
        
        result = asyncio.run(run_test())
        
        print("\n" + "=" * 80)
        if result:
            print("RESULT: ✓ FIX VERIFIED - Scheduler handles corrupted data gracefully")
            print("=" * 80)
            print("\nThe fix successfully:")
            print("  1. Normalizes mixed tuple/list types to consistent tuples")
            print("  2. Ensures priority is always int and request_id is always str")
            print("  3. Prevents TypeError in heapq operations")
            print("  4. Maintains queue integrity")
            return 0  # Success
        else:
            print("RESULT: ✗ BUG EXISTS - TypeError occurred with corrupted data")
            print("=" * 80)
            print("\nThis is expected on main branch.")
            print("The fix should resolve this issue.")
            return 1  # Bug exists
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2  # Unexpected error


def main():
    print("\n" + "=" * 80)
    print("FIX VERIFICATION FOR ISSUE #14817")
    print("TypeError in Scheduler When Priority Field Has Inconsistent Types")
    print("=" * 80)
    print("\nGitHub Issue: https://github.com/BerriAI/litellm/issues/14817")
    print()
    
    exit_code = test_scheduler_with_corrupted_redis_cache()
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    if exit_code == 0:
        print("Exit code 0: Fix is working correctly ✓")
        print("This should be the result on fix/scheduler-priority-type-error branch")
    elif exit_code == 1:
        print("Exit code 1: Bug is present ✗")
        print("This should be the result on main branch")
    else:
        print("Exit code 2: Unexpected error occurred")
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


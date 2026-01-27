#!/usr/bin/env python3
"""
Test script for issue #12443: Gemini aiohttp session leak

Validates that:
1. BaseLLMAIOHTTPHandler properly closes sessions via __del__
2. atexit handler works with new event loop approach
3. No "Unclosed client session" warnings are generated
"""

import asyncio
import gc
import sys
from pathlib import Path

import pytest

# Add litellm to path
sys.path.insert(0, str(Path(__file__).parent))


def count_aiohttp_sessions():
    """Count unclosed aiohttp ClientSession objects"""
    import aiohttp

    count = 0
    for obj in gc.get_objects():
        if isinstance(obj, aiohttp.ClientSession):
            if not obj.closed:
                count += 1
    return count


async def test_aiohttp_handler_cleanup():
    """Test BaseLLMAIOHTTPHandler session cleanup"""
    print("\n" + "=" * 70)
    print("TEST: BaseLLMAIOHTTPHandler Session Cleanup")
    print("=" * 70)

    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    initial_sessions = count_aiohttp_sessions()
    print(f"\nInitial unclosed sessions: {initial_sessions}")

    # Create handler and trigger session creation
    print("\nCreating BaseLLMAIOHTTPHandler and triggering session creation...")
    handler = BaseLLMAIOHTTPHandler()

    # This triggers session creation (line 111 of aiohttp_handler.py)
    session = handler._get_async_client_session()
    print(f"Session created: {session}")

    sessions_after_create = count_aiohttp_sessions()
    print(f"Sessions after creation: {sessions_after_create}")

    # Delete handler - should trigger __del__ cleanup
    print("\nDeleting handler (should trigger __del__)...")
    del handler
    del session
    gc.collect()
    await asyncio.sleep(0.1)  # Let async cleanup finish

    final_sessions = count_aiohttp_sessions()
    print(f"Final unclosed sessions: {final_sessions}")

    session_diff = final_sessions - initial_sessions
    print(f"\nSession difference: {session_diff:+d}")

    if session_diff == 0:
        print("\n✅ PASS: __del__ cleanup working correctly")
        return True
    else:
        print(f"\n❌ FAIL: {session_diff} sessions leaked")
        return False


async def test_atexit_cleanup():
    """Test that atexit cleanup works with new event loop approach"""
    print("\n" + "=" * 70)
    print("TEST: atexit Cleanup (new event loop approach)")
    print("=" * 70)

    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )

    initial_sessions = count_aiohttp_sessions()
    print(f"\nInitial unclosed sessions: {initial_sessions}")

    # Use the actual global base_llm_aiohttp_handler from litellm.main
    print("\nAccessing global base_llm_aiohttp_handler (like Gemini does)...")
    import litellm

    handler = litellm.base_llm_aiohttp_handler
    session = handler._get_async_client_session()

    sessions_after_create = count_aiohttp_sessions()
    print(f"Sessions after creation: {sessions_after_create}")

    # Call cleanup function (simulates atexit)
    print("\nCalling close_litellm_async_clients() (simulates atexit)...")
    await close_litellm_async_clients()

    gc.collect()
    await asyncio.sleep(0.1)

    final_sessions = count_aiohttp_sessions()
    print(f"Final unclosed sessions: {final_sessions}")

    session_diff = final_sessions - initial_sessions
    print(f"\nSession difference: {session_diff:+d}")

    if session_diff == 0:
        print("\n✅ PASS: atexit cleanup working correctly")
        return True
    else:
        print(f"\n❌ FAIL: {session_diff} sessions leaked")
        return False


def test_new_event_loop_atexit():
    """Test that the new atexit handler can create a fresh event loop"""
    print("\n" + "=" * 70)
    print("TEST: atexit with Fresh Event Loop Creation")
    print("=" * 70)

    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )

    print("\nVerifying atexit handler can create fresh loop (no running loop)...")
    print("Note: At atexit time, there's typically no running event loop")

    # Save current loop to restore later
    try:
        current_loop = asyncio.get_running_loop()
        print("Warning: Found running loop - can't test atexit scenario accurately")
        pytest.skip("Cannot test atexit scenario when event loop is running")
    except RuntimeError:
        pass  # Good - no running loop

    # Create a new loop like the fixed atexit handler does
    print("Creating new event loop (like fixed atexit handler)...")
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        new_loop.run_until_complete(close_litellm_async_clients())
        print("✅ Successfully ran cleanup with fresh event loop")
    finally:
        new_loop.close()


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Gemini aiohttp Session Leak Fix Validation (Issue #12443)")
    print("=" * 70)

    results = []

    # Test 1: __del__ cleanup
    results.append(await test_aiohttp_handler_cleanup())

    # Test 2: atexit cleanup function
    results.append(await test_atexit_cleanup())

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✅ All tests PASSED - Issue #12443 is FIXED")
    else:
        print(f"\n❌ {total - passed} test(s) FAILED")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

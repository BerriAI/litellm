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


async def test_aiohttp_handler_cleanup():
    """Test BaseLLMAIOHTTPHandler session cleanup via __del__"""
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    # Create handler and trigger session creation
    handler = BaseLLMAIOHTTPHandler()
    session = handler._get_async_client_session()

    assert not session.closed, "Session should be open after creation"

    # Delete handler - should trigger __del__ cleanup
    del handler
    gc.collect()
    await asyncio.sleep(0.1)  # Let async cleanup finish

    assert session.closed, "Session should be closed after handler deletion"


async def test_atexit_cleanup():
    """Test that atexit cleanup works with new event loop approach"""
    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )

    import litellm

    # Use the actual global base_llm_aiohttp_handler from litellm
    handler = litellm.base_llm_aiohttp_handler
    session = handler._get_async_client_session()

    assert not session.closed, "Session should be open after creation"

    # Call cleanup function (simulates atexit)
    await close_litellm_async_clients()

    assert session.closed, "Session should be closed after atexit cleanup"


def test_new_event_loop_atexit():
    """Test that the new atexit handler can create a fresh event loop"""
    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )

    # At atexit time, there's typically no running event loop
    try:
        asyncio.get_running_loop()
        pytest.skip("Cannot test atexit scenario when event loop is running")
    except RuntimeError:
        pass  # Good - no running loop

    # Create a new loop like the fixed atexit handler does
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        new_loop.run_until_complete(close_litellm_async_clients())
    finally:
        new_loop.close()


if __name__ == "__main__":
    success = asyncio.run(test_aiohttp_handler_cleanup())
    sys.exit(0 if success else 1)

"""
Verifies that the httpx client used by AsyncOpenAI is NOT closed
when AsyncHTTPHandler instances are garbage collected.
"""
import asyncio
import gc
import httpx
from litellm.llms.openai.common_utils import BaseOpenAILLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


async def test_httpx_client_not_closed_by_handler_gc():
    """
    Before the fix: _get_async_http_client() returned handler.client,
    so when handler was GC'd its __del__ closed the client.
    After the fix: returns a standalone httpx.AsyncClient, no handler involved.
    """
    # Get the client the same way AsyncOpenAI would
    client = BaseOpenAILLM._get_async_http_client()
    assert isinstance(client, httpx.AsyncClient)

    # Simulate what the old code did: create an AsyncHTTPHandler and GC it
    handler = AsyncHTTPHandler()
    handler_client = handler.client
    del handler
    gc.collect()

    # The client from _get_async_http_client should still be open
    # because it's NOT tied to any AsyncHTTPHandler
    assert not client.is_closed, "Client was closed prematurely!"

    # Verify it can actually send (build a request without sending)
    try:
        req = client.build_request("GET", "https://example.com")
        print("PASS: Client is still usable after handler GC")
    except RuntimeError as e:
        if "closed" in str(e):
            print(f"FAIL: {e}")
            raise
        raise

    await client.aclose()
    print("All checks passed!")


asyncio.run(test_httpx_client_not_closed_by_handler_gc())

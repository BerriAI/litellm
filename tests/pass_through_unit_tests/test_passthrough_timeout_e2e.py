"""
E2E test: verify pass-through endpoint timeout actually fires against a real slow server.

This test starts a local aiohttp server, so it lives outside tests/test_litellm/
(which only allows mock tests).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Headers, QueryParams
from fastapi import Request

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)
from litellm.proxy.proxy_server import ProxyException


@pytest.mark.asyncio
async def test_pass_through_request_timeout_actually_fires():
    """
    Start a real HTTP server that sleeps for 10s, call pass_through_request
    with timeout=2, and verify a timeout exception is raised.
    This proves the timeout config actually works end-to-end.
    """
    from aiohttp import web

    # 1. Create a slow HTTP server
    async def slow_handler(request):
        await asyncio.sleep(10)  # 10 seconds - longer than timeout
        return web.Response(text='{"result": "ok"}', content_type="application/json")

    app = web.Application()
    app.router.add_post("/slow", slow_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    # 2. Flush client cache to ensure fresh httpx client with our timeout
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()

    # 3. Build a mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.query_params = QueryParams("")
    mock_request.method = "POST"
    mock_request.body = AsyncMock(return_value=b'{"test": "data"}')

    mock_user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
    mock_proxy_logging.post_call_failure_hook = AsyncMock()

    try:
        with patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            mock_proxy_logging,
        ), patch(
            "litellm.proxy.pass_through_endpoints.passthrough_guardrails.PassthroughGuardrailHandler.collect_guardrails",
            return_value=[],
        ):
            with pytest.raises(ProxyException) as exc_info:
                await pass_through_request(
                    request=mock_request,
                    target=f"http://127.0.0.1:{port}/slow",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=mock_user_api_key_dict,
                    timeout=2,  # 2 seconds - should fire before the 10s sleep
                )

            # Verify the exception message indicates a timeout
            assert (
                "timeout" in exc_info.value.message.lower()
            ), f"Expected 'timeout' in message, got: {exc_info.value.message}"
    finally:
        await runner.cleanup()
        if cache is not None:
            cache.flush_cache()

"""
Regression test for https://github.com/BerriAI/litellm/issues/22706
ProxyException raised during MCP auth should preserve status code, not return 500.
"""
import pytest
from unittest.mock import AsyncMock, patch
from starlette.types import Scope, Receive, Send


@pytest.mark.asyncio
async def test_proxy_exception_preserves_status_code_in_mcp():
    """
    When a ProxyException (e.g. 403) is raised during MCP auth,
    the response should return 403 - not 500.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp

    # Minimal ASGI scope for an MCP request
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "query_string": b"",
    }
    receive: Receive = AsyncMock()

    # Capture what status code was sent in the response
    sent_responses = []

    async def mock_send(message):
        sent_responses.append(message)

    # Simulate custom auth raising a 403 ProxyException
    proxy_exc = ProxyException(
        message="Forbidden - token expired",
        type="auth_error",
        param=None,
        code=403,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new=AsyncMock(side_effect=proxy_exc),
    ):
        await handle_streamable_http_mcp(scope, receive, mock_send)

    # Should have sent a response (not re-raised as 500)
    assert len(sent_responses) > 0, "No response was sent"

    # Find the http.response.start message which contains the status code
    start_msg = next(
        (m for m in sent_responses if m.get("type") == "http.response.start"), None
    )
    assert start_msg is not None, "No http.response.start message found"

    # THIS IS THE BUG - before fix this will be 500, after fix it should be 403
    assert start_msg["status"] == 403, (
        f"Expected 403 but got {start_msg['status']} - "
        "ProxyException status code is being lost and converted to 500"
    )

@pytest.mark.asyncio
async def test_proxy_exception_preserves_status_code_in_sse_mcp():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/22706
    ProxyException raised during SSE MCP auth should preserve status code, not return 500.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy._experimental.mcp_server.server import handle_sse_mcp

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "query_string": b"",
    }
    receive: Receive = AsyncMock()
    sent_responses = []

    async def mock_send(message):
        sent_responses.append(message)

    proxy_exc = ProxyException(
        message="Forbidden - token expired",
        type="auth_error",
        param=None,
        code=403,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new=AsyncMock(side_effect=proxy_exc),
    ):
        await handle_sse_mcp(scope, receive, mock_send)

    assert len(sent_responses) > 0, "No response was sent"
    start_msg = next(
        (m for m in sent_responses if m.get("type") == "http.response.start"), None
    )
    assert start_msg is not None, "No http.response.start message found"
    assert start_msg["status"] == 403, (
        f"Expected 403 but got {start_msg['status']}"
    )
    
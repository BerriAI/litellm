"""
Regression tests for MCP handler ProxyException response behavior (Issue #22706).

Ensures custom/auth ProxyExceptions are not rewritten to generic 500 responses.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import ProxyException


def _base_scope(path: str = "/mcp"):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"content-type", b"application/json")],
    }


@pytest.mark.asyncio
async def test_streamable_http_preserves_proxy_exception_status_and_payload():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = _base_scope("/mcp")
    receive = AsyncMock()
    sent_messages = []

    async def send(message):
        sent_messages.append(message)

    proxy_exc = ProxyException(
        message="Forbidden by custom auth",
        type="auth_error",
        param="api_key",
        code=403,
        headers={"www-authenticate": "Bearer realm=litellm"},
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        side_effect=proxy_exc,
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert len(sent_messages) >= 2

    start_msg = sent_messages[0]
    body_msg = sent_messages[1]

    assert start_msg["type"] == "http.response.start"
    assert start_msg["status"] == 403

    response_headers = {
        k.decode().lower(): v.decode() for k, v in start_msg.get("headers", [])
    }
    assert response_headers.get("www-authenticate") == "Bearer realm=litellm"

    assert body_msg["type"] == "http.response.body"
    payload = json.loads(body_msg["body"].decode())
    assert payload["error"]["message"] == "Forbidden by custom auth"
    assert payload["error"]["type"] == "auth_error"
    assert int(payload["error"]["code"]) == 403


@pytest.mark.asyncio
async def test_sse_handler_preserves_proxy_exception_status_and_payload():
    try:
        from litellm.proxy._experimental.mcp_server.server import handle_sse_mcp
    except ImportError:
        pytest.skip("MCP server not available")

    scope = _base_scope("/sse")
    receive = AsyncMock()
    sent_messages = []

    async def send(message):
        sent_messages.append(message)

    proxy_exc = ProxyException(
        message="OAuth token expired",
        type="auth_error",
        param="authorization",
        code=401,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        side_effect=proxy_exc,
    ):
        await handle_sse_mcp(scope, receive, send)

    assert len(sent_messages) >= 2

    start_msg = sent_messages[0]
    body_msg = sent_messages[1]

    assert start_msg["type"] == "http.response.start"
    assert start_msg["status"] == 401

    assert body_msg["type"] == "http.response.body"
    payload = json.loads(body_msg["body"].decode())
    assert payload["error"]["message"] == "OAuth token expired"
    assert payload["error"]["type"] == "auth_error"
    assert int(payload["error"]["code"]) == 401

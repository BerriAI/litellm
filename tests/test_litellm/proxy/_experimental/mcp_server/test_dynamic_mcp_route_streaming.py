"""Tests for dynamic_mcp_route streaming behavior and MCP mount ordering."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.responses import StreamingResponse


@pytest.mark.asyncio
async def test_dynamic_mcp_route_streams_sse_response():
    """dynamic_mcp_route should return a StreamingResponse for SSE content."""
    try:
        from litellm.proxy.proxy_server import dynamic_mcp_route
    except ImportError:
        pytest.skip("proxy_server not available")

    mock_request = MagicMock()
    mock_request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/test_server/mcp",
        "headers": [],
        "query_string": b"",
    }
    mock_request.receive = AsyncMock()

    sse_chunks = [b"event: message\ndata: chunk1\n\n", b"event: message\ndata: chunk2\n\n"]

    async def fake_asgi_handler(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                ],
            }
        )
        for i, chunk in enumerate(sse_chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": i < len(sse_chunks) - 1,
                }
            )

    mock_server = MagicMock()
    mock_server.server_name = "test_server"

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=mock_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.handle_streamable_http_mcp",
            side_effect=fake_asgi_handler,
        ),
    ):
        response = await dynamic_mcp_route("test_server", mock_request)

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"

    # Consume the body generator and verify chunks
    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk
    assert collected == b"".join(sse_chunks)


@pytest.mark.asyncio
async def test_dynamic_mcp_route_handles_non_streaming_response():
    """dynamic_mcp_route should also work for non-SSE (single-chunk) responses."""
    try:
        from litellm.proxy.proxy_server import dynamic_mcp_route
    except ImportError:
        pytest.skip("proxy_server not available")

    mock_request = MagicMock()
    mock_request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/test_server/mcp",
        "headers": [],
        "query_string": b"",
    }
    mock_request.receive = AsyncMock()

    body = b'{"jsonrpc":"2.0","result":{},"id":1}'

    async def fake_asgi_handler(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    mock_server = MagicMock()
    mock_server.server_name = "test_server"

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=mock_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.handle_streamable_http_mcp",
            side_effect=fake_asgi_handler,
        ),
    ):
        response = await dynamic_mcp_route("test_server", mock_request)

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.media_type == "application/json"

    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk
    assert collected == body


@pytest.mark.asyncio
async def test_dynamic_mcp_route_returns_404_for_unknown_server():
    """dynamic_mcp_route should raise 404 for a server that doesn't exist."""
    try:
        from litellm.proxy.proxy_server import dynamic_mcp_route
    except ImportError:
        pytest.skip("proxy_server not available")

    mock_request = MagicMock()
    mock_request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/nonexistent/mcp",
        "headers": [],
        "query_string": b"",
    }

    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.get_mcp_server_by_name",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await dynamic_mcp_route("nonexistent", mock_request)
        assert exc_info.value.status_code == 404


def test_sse_mount_precedes_catch_all():
    """The /sse mount must appear before the / catch-all in the MCP sub-app."""
    try:
        from litellm.proxy._experimental.mcp_server.server import app as mcp_app
    except ImportError:
        pytest.skip("MCP server module not available")

    from starlette.routing import Mount

    mounts = [r for r in mcp_app.routes if isinstance(r, Mount)]
    mount_paths = [m.path for m in mounts]

    # Starlette normalizes "/" to "" for mount paths
    catch_all = "" if "" in mount_paths else "/"

    # /sse must be present and before the catch-all
    assert "/sse" in mount_paths, f"Missing /sse mount. Found: {mount_paths}"
    assert catch_all in mount_paths, f"Missing catch-all mount. Found: {mount_paths}"
    assert mount_paths.index("/sse") < mount_paths.index(
        catch_all
    ), f"/sse must precede catch-all. Order: {mount_paths}"

    # The broken /mcp and /{mcp_server_name}/mcp mounts should not exist
    assert "/mcp" not in mount_paths, f"Unexpected /mcp mount found. Mounts: {mount_paths}"
    assert "/{mcp_server_name}/mcp" not in mount_paths, (
        f"Unexpected /{{mcp_server_name}}/mcp mount found. Mounts: {mount_paths}"
    )

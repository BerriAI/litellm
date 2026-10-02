import asyncio
from typing import Final

import httpx
import pytest
from fastapi import HTTPException
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.proxy_server import _stream_mcp_asgi_response


@pytest.mark.asyncio
async def test_stream_mcp_asgi_response_propagates_pre_header_http_exception():
    async def handle_fn(_scope, _receive, _send):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={
                "WWW-Authenticate": "Bearer authorization_uri=https://example.test/auth"
            },
        )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(
            _stream_mcp_asgi_response(
                handle_fn,
                {"type": "http", "method": "POST", "path": "/mcp", "headers": []},
                receive,
            ),
            timeout=1.0,
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {
        "WWW-Authenticate": "Bearer authorization_uri=https://example.test/auth"
    }


@pytest.mark.asyncio
async def test_streamed_initialize_preserves_session_until_delete() -> None:
    manager: Final = StreamableHTTPSessionManager(app=Server("session-test"), stateless=False)

    async def endpoint(request: Request) -> Response:
        return await _stream_mcp_asgi_response(manager.handle_request, request.scope, request.receive)

    app: Final = Starlette(routes=[Route("/mcp", endpoint, methods=["POST", "DELETE"])])
    async with manager.run(), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Accept": "application/json, text/event-stream"},
    ) as client:
        async with asyncio.timeout(5):
            initialized: Final = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "session-test", "version": "1"},
                    },
                },
            )
            assert initialized.status_code == 200
            headers: Final = {"mcp-session-id": initialized.headers["mcp-session-id"]}
            notification: Final = await client.post(
                "/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            assert notification.status_code == 202
            ping: Final = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
            assert (await client.post("/mcp", headers=headers, json=ping)).status_code == 200
            assert (await client.delete("/mcp", headers=headers)).status_code == 200
            assert (await client.post("/mcp", headers=headers, json=ping)).status_code == 404


@pytest.mark.asyncio
async def test_disconnected_response_cancels_the_active_handler() -> None:
    stopped: Final = asyncio.Event()

    async def handle(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async with asyncio.timeout(1):
        response: Final = await _stream_mcp_asgi_response(handle, {}, receive)
        assert await anext(response.body_iterator) == b"first"
        await response.body_iterator.aclose()
        assert stopped.is_set()

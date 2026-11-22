import asyncio
from typing import Final

import pytest
from fastapi import HTTPException
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.proxy_server import _stream_mcp_asgi_response


@pytest.mark.asyncio
async def test_completed_response_finishes_handler_cleanup():
    body_consumed: Final = asyncio.Event()
    completed: Final = asyncio.Event()

    async def handle_fn(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"initialized", "more_body": False})
        await body_consumed.wait()
        completed.set()

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    response: Final = await _stream_mcp_asgi_response(
        handle_fn, {"type": "http", "method": "POST", "path": "/mcp", "headers": []}, receive
    )
    body: Final = aiter(response.body_iterator)
    assert await anext(body) == b"initialized"
    body_consumed.set()
    assert await anext(body, None) is None
    assert completed.is_set()


@pytest.mark.asyncio
async def test_disconnected_response_cancels_handler():
    body_received: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()
    hold_open: Final = asyncio.Event()

    async def handle_fn(_scope: Scope, _receive: Receive, send: Send) -> None:
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"event", "more_body": True})
            await hold_open.wait()
        finally:
            cancelled.set()

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    response: Final = await _stream_mcp_asgi_response(
        handle_fn, {"type": "http", "method": "POST", "path": "/mcp", "headers": []}, receive
    )

    async def consume() -> None:
        async for _ in response.body_iterator:
            body_received.set()

    consumer: Final = asyncio.create_task(consume())
    await body_received.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert cancelled.is_set()


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

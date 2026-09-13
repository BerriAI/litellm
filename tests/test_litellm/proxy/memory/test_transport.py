import asyncio
from typing import Final

import pytest
from starlette.requests import Request
from starlette.types import Receive, Scope, Send

from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.memory.transport import gateway_round


@pytest.mark.asyncio
async def test_stream_reaches_client_before_model_finishes_and_disconnect_cancels_the_model() -> None:
    continuing: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["client"] == ("192.0.2.3", 12345)
        assert scope["query_string"] == b"api-version=test"
        body: Final = await _read_request_body(Request(scope, receive))
        assert body["model"] == "test"
        assert body["stream"] is True
        assert body["extra_headers"] == {"anthropic-beta": "test-beta"}
        assert body["headers"] == {"x-custom": "preserved"}
        assert not Request(scope).headers.get("idempotency-key")
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b"first delta", "more_body": True})
        try:
            await continuing.wait()
        finally:
            cancelled.set()

    request: Final = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"api-version=test",
            "headers": [(b"idempotency-key", b"outer-request")],
            "client": ("192.0.2.3", 12345),
            "server": ("gateway.example", 443),
            "parsed_body": (("model", "stream"), {"model": "original-body", "stream": False}),
            "state": {"_cached_headers": {"content-type": "application/x-www-form-urlencoded"}},
        }
    )
    async with gateway_round(
        app,
        request,
        {
            "model": "test",
            "stream": True,
            "extra_headers": {"Idempotency-Key": "outer-request", "anthropic-beta": "test-beta"},
            "headers": {"X-Request-ID": "outer-request", "x-custom": "preserved"},
        },
    ) as call:
        stream: Final = call.chunks()
        assert await asyncio.wait_for(anext(stream), timeout=1) == b"first delta"
        assert not continuing.is_set()
    assert cancelled.is_set()
    assert call.task is not None and call.task.cancelled()
    await stream.aclose()

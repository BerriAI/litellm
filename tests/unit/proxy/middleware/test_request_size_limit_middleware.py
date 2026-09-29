import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.types import Message

from litellm.proxy.middleware.request_size_limit_middleware import (
    RequestSizeLimitMiddleware,
)


def test_request_size_limit_middleware_rejects_content_length_before_body_read():
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: 1,
            is_request_size_limit_enabled=lambda: True,
        )
    )

    response = client.post(
        "/chat/completions",
        content=b"x" * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}
    assert response.headers["content-length"] == str(len(response.content))
    assert downstream_called is False


def test_request_size_limit_middleware_zero_limit_disables_guard():
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: 0,
            is_request_size_limit_enabled=lambda: True,
        )
    )

    response = client.post(
        "/chat/completions",
        content=b"x",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert downstream_called is True


@pytest.mark.asyncio
async def test_request_size_limit_middleware_rejects_streamed_body_without_content_length():
    received_body_bytes = 0

    async def app(scope, receive, send):
        nonlocal received_body_bytes
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            received_body_bytes += len(message.get("body", b""))
            if not message.get("more_body", False):
                break

        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = RequestSizeLimitMiddleware(
        app,
        get_max_request_size_mb=lambda: 1,
        is_request_size_limit_enabled=lambda: True,
    )
    sent_messages: list[Message] = []
    receive_messages: list[Message] = [
        {
            "type": "http.request",
            "body": b"x" * (1024 * 1024),
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"y",
            "more_body": False,
        },
    ]

    async def receive():
        return receive_messages.pop(0)

    async def send(message):
        sent_messages.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )

    expected_body = b'{"error":"Request size is too large. Max size is 1 MB"}'
    assert sent_messages[0] == {
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(expected_body)).encode("latin-1")),
        ],
    }
    assert sent_messages[1] == {
        "type": "http.response.body",
        "body": expected_body,
        "more_body": False,
    }
    assert received_body_bytes == 1024 * 1024

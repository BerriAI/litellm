from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.request_size_limit_middleware import (
    RequestSizeLimitMiddleware,
)


@pytest.mark.parametrize(("path", "root_path", "operator_mb", "route_mb"), [
    ("/chat/completions", "", 1, None),
    ("/claude-code/plugins/upload", "", None, 1),
    ("/gateway/claude-code/plugins/upload", "/gateway", None, 1),
    ("/claude-code/plugins/upload", "", 2, 1),
    ("/claude-code/plugins/upload", "", 1, 2),
])
def test_request_size_limit_middleware_rejects_content_length_before_body_read(
    path: str, root_path: str, operator_mb: int | None, route_mb: int | None,
) -> None:
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: operator_mb,
            is_request_size_limit_enabled=lambda: operator_mb is not None,
            route_max_request_size_mb={("POST", "/claude-code/plugins/upload"): route_mb} if route_mb else None,
        ),
        root_path=root_path,
    )

    response = client.post(
        path,
        content=b"x" * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}
    assert response.headers["content-length"] == str(len(response.content))
    assert downstream_called is False


@pytest.mark.parametrize(("enabled", "operator_mb"), [(True, 0), (False, 1)])
@pytest.mark.parametrize(("method", "path"), [
    ("POST", "/chat/completions"),
    ("PUT", "/claude-code/plugins/upload"),
    ("POST", "/claude-code/plugins/upload-other"),
    ("POST", "/claude-code/plugins"),
])
def test_request_size_limit_middleware_zero_or_disabled_global_limit_preserves_other_routes(
    enabled: bool, operator_mb: int, method: str, path: str,
) -> None:
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: operator_mb,
            is_request_size_limit_enabled=lambda: enabled,
            route_max_request_size_mb={("POST", "/claude-code/plugins/upload"): 1},
        )
    )

    response = client.request(
        method, path,
        content=b"x" * (1024 * 1024 + 1),
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


def test_an_oversized_messages_request_gets_the_anthropic_request_too_large_error() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"ok": True})(scope, receive, send)

    client: Final = TestClient(
        RequestSizeLimitMiddleware(app, get_max_request_size_mb=lambda: 1, is_request_size_limit_enabled=lambda: True)
    )

    response: Final = client.post(
        "/v1/messages", content=b"x" * (1024 * 1024 + 1), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413
    assert response.json() == {
        "type": "error",
        "error": {"type": "request_too_large", "message": "Request size is too large. Max size is 1 MB"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/claude-code/plugins/upload", "/chat/completions"))
async def test_chunked_upload_overflow_stops_fastapi_and_auth_multipart_parsing(
    path: str,
) -> None:
    from litellm.proxy.proxy_server import app

    limited: Final = RequestSizeLimitMiddleware(
        app,
        get_max_request_size_mb=lambda: None,
        is_request_size_limit_enabled=lambda: False,
        route_max_request_size_mb={("POST", path): 1 / 1024},
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield b'--skill\r\nContent-Disposition: form-data; name="file"; filename="skill.zip"\r\n\r\n'
        yield b"x" * 1024
        pytest.fail("The server consumed more of the upload after its byte limit")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="http://test") as client:
        response: Final = await client.post(
            path,
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=skill"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request size is too large. Max size is 0.0009765625 MB"}

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


@pytest.mark.parametrize(("method", "path", "root_path", "operator_mb", "route_mb"), [
    ("POST", "/chat/completions", "", 1, None),
    ("POST", "/claude-code/plugins/upload", "", None, 1),
    ("POST", "/skill-hub/skills/upload", "", None, 1),
    ("POST", "/gateway/claude-code/plugins/upload", "/gateway", None, 1),
    ("POST", "/claude-code/plugins/upload", "", 2, 1),
    ("POST", "/claude-code/plugins/upload", "", 1, 2),
    ("PUT", "/skill-hub/skills/example/upload", "", None, 1),
    ("PUT", "/gateway/skill-hub/skills/example/upload", "/gateway", None, 1),
    ("PUT", "/skill-hub/skills/example/upload", "/gateway", None, 1),
    ("PUT", "/skill-hub/skills/example/upload", "", 1, 2),
])
def test_request_size_limit_middleware_rejects_content_length_before_body_read(
    method: str, path: str, root_path: str, operator_mb: int | None, route_mb: int | None,
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
            route_max_request_size_mb={
                ("POST", "/claude-code/plugins/upload"): route_mb,
                ("POST", "/skill-hub/skills/upload"): route_mb,
                ("PUT", "/skill-hub/skills/{name}/upload"): route_mb,
            } if route_mb else None,
        ),
        root_path=root_path,
    )

    response = client.request(
        method, path,
        content=b"x" * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}
    assert response.headers["content-length"] == str(len(response.content))
    assert downstream_called is False


@pytest.mark.parametrize(("enabled", "operator_mb"), [(True, 0), (False, 1)])
@pytest.mark.parametrize(("method", "path", "root_path"), [
    ("POST", "/chat/completions", ""),
    ("PUT", "/claude-code/plugins/upload", ""),
    ("POST", "/claude-code/plugins/upload-other", ""),
    ("POST", "/claude-code/plugins", ""),
    ("POST", "/skill-hub/skills/example/upload", ""),
    ("PUT", "/skill-hub/skills/example/upload-other", ""),
    ("PUT", "/skill-hub/skills/example/nested/upload", ""),
    ("PUT", "/skill-hub/skills//upload", ""),
    ("PUT", "/gateway-other/skill-hub/skills/example/upload", "/gateway"),
])
def test_request_size_limit_middleware_zero_or_disabled_global_limit_preserves_other_routes(
    enabled: bool, operator_mb: int, method: str, path: str, root_path: str,
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
            route_max_request_size_mb={
                ("POST", "/claude-code/plugins/upload"): 1,
                ("PUT", "/skill-hub/skills/{name}/upload"): 1,
            },
        ),
        root_path=root_path,
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
@pytest.mark.parametrize(("method", "path", "root_path", "operator_mb"), [
    ("POST", "/chat/completions", "", 1),
    ("POST", "/skill-hub/skills/upload", "", None),
    ("PUT", "/skill-hub/skills/example/upload", "", None),
    ("PUT", "/gateway/skill-hub/skills/example/upload", "/gateway", None),
])
async def test_request_size_limit_middleware_rejects_streamed_body_without_content_length(
    method: str, path: str, root_path: str, operator_mb: int | None,
) -> None:
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
        get_max_request_size_mb=lambda: operator_mb,
        is_request_size_limit_enabled=lambda: operator_mb is not None,
        route_max_request_size_mb={
            ("POST", "/skill-hub/skills/upload"): 1, ("PUT", "/skill-hub/skills/{name}/upload"): 1,
        },
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
            "method": method,
            "path": path,
            "root_path": root_path,
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
@pytest.mark.parametrize(("method", "path", "route"), [
    ("POST", "/claude-code/plugins/upload", "/claude-code/plugins/upload"),
    ("POST", "/skill-hub/skills/upload", "/skill-hub/skills/upload"),
    ("POST", "/chat/completions", "/chat/completions"),
    ("PUT", "/skill-hub/skills/example/upload", "/skill-hub/skills/{name}/upload"),
])
async def test_chunked_upload_overflow_stops_fastapi_and_auth_multipart_parsing(
    method: str, path: str, route: str,
) -> None:
    from litellm.proxy.proxy_server import app

    limited: Final = RequestSizeLimitMiddleware(
        app,
        get_max_request_size_mb=lambda: None,
        is_request_size_limit_enabled=lambda: False,
        route_max_request_size_mb={(method, route): 1 / 1024},
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield b'--skill\r\nContent-Disposition: form-data; name="file"; filename="skill.zip"\r\n\r\n'
        yield b"x" * 1024
        pytest.fail("The server consumed more of the upload after its byte limit")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="http://test") as client:
        response: Final = await client.request(
            method, path,
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=skill"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request size is too large. Max size is 0.0009765625 MB"}

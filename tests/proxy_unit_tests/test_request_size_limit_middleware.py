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
            get_max_file_size_mb=lambda: None,
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
            get_max_file_size_mb=lambda: None,
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
        get_max_file_size_mb=lambda: None,
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


def _size_limited_client(
    max_request_size_mb: float | None,
    max_file_size_mb: float | None,
    root_path: str = "",
) -> TestClient:
    async def app(scope, receive, send):
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    return TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: max_request_size_mb,
            get_max_file_size_mb=lambda: max_file_size_mb,
            is_request_size_limit_enabled=lambda: True,
        ),
        root_path=root_path,
    )


@pytest.mark.parametrize(
    "path",
    [
        "/files",
        "/v1/files",
        "/openai/v1/files",
        "/azure/v1/files",
    ],
)
def test_file_upload_routes_use_max_file_size_mb(path):
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5)

    response = client.post(path, content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_file_upload_override_does_not_leak_to_other_routes():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5)

    response = client.post("/chat/completions", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}


@pytest.mark.parametrize("root_path", ["/api/genai", "/api/genai/"])
def test_file_upload_override_applies_under_server_root_path(root_path):
    # A multi-segment prefix so an unstripped path cannot match the
    # /{provider}/v1/files pattern by accident.
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5, root_path=root_path)

    response = client.post("/api/genai/v1/files", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 200


def test_root_path_is_only_stripped_on_segment_boundaries():
    # "/v" is a character-wise prefix of "/v1/files" but not a segment of it,
    # so stripping it would mangle the path into "1/files" and lose the match.
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5, root_path="/v")

    response = client.post("/v1/files", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_app_root_path_takes_precedence_over_root_path():
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = RequestSizeLimitMiddleware(
        app,
        get_max_request_size_mb=lambda: 1,
        get_max_file_size_mb=lambda: 5,
        is_request_size_limit_enabled=lambda: True,
    )
    body_size = 2 * 1024 * 1024

    async def receive():
        return {"type": "http.request", "body": b"x" * body_size, "more_body": False}

    async def send(message):
        return None

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/genai/v1/files",
            "app_root_path": "/api/genai",
            "root_path": "/api/genai/v1/files",
            "headers": [(b"content-length", str(body_size).encode("latin-1"))],
        },
        receive,
        send,
    )

    assert downstream_called is True


def test_file_upload_over_its_own_limit_is_rejected():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5)

    response = client.post("/v1/files", content=b"x" * (5 * 1024 * 1024 + 1))

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 5 MB"}


def test_file_upload_falls_back_to_max_request_size_mb_when_unset():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=None)

    response = client.post("/v1/files", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}


def test_zero_max_file_size_mb_uncaps_uploads_only():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=0)

    assert client.post("/v1/files", content=b"x" * (2 * 1024 * 1024)).status_code == 200
    assert client.post("/chat/completions", content=b"x" * (2 * 1024 * 1024)).status_code == 413


def test_vector_store_files_route_is_not_a_file_upload_route():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5)

    response = client.post("/v1/vector_stores/vs_1/files", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 413


def test_non_post_requests_to_files_route_use_the_global_limit():
    client = _size_limited_client(max_request_size_mb=1, max_file_size_mb=5)

    response = client.request("PUT", "/v1/files", content=b"x" * (2 * 1024 * 1024))

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_streamed_file_upload_rejected_at_max_file_size_mb():
    async def app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect" or not message.get("more_body", False):
                break
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = RequestSizeLimitMiddleware(
        app,
        get_max_request_size_mb=lambda: 1,
        get_max_file_size_mb=lambda: 2,
        is_request_size_limit_enabled=lambda: True,
    )
    sent_messages: list[Message] = []
    receive_messages: list[Message] = [
        {"type": "http.request", "body": b"x" * (2 * 1024 * 1024), "more_body": True},
        {"type": "http.request", "body": b"y", "more_body": False},
    ]

    async def receive():
        return receive_messages.pop(0)

    async def send(message):
        sent_messages.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/files",
            "headers": [(b"content-type", b"multipart/form-data; boundary=x")],
        },
        receive,
        send,
    )

    assert sent_messages[0]["status"] == 413
    assert sent_messages[1]["body"] == b'{"error":"Request size is too large. Max size is 2 MB"}'

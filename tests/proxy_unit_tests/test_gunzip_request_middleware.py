"""
Tests for GunzipRequestMiddleware.

Covers all Content-Encoding scenarios:
- gzip-encoded body is decompressed before the route sees it
- Content-Encoding header stripped, Content-Length rewritten
- case-insensitive: GZIP, Gzip, gzip all work
- whitespace-tolerant: "  gzip  " works
- no Content-Encoding header -> passthrough untouched
- non-gzip encodings (deflate, br, identity) -> passthrough untouched
- invalid gzip data -> 400
- empty body with Content-Encoding: gzip -> 400
- chunked ASGI body (more_body=True) reassembled correctly
- non-http scope (lifespan) -> passthrough
- client disconnect during body read -> clean exit
- only decompressed result reaches downstream
"""

import asyncio
import gzip
import json

import pytest

from litellm.proxy.middleware.gunzip_request_middleware import GunzipRequestMiddleware

PAYLOAD = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello " * 100}]}


def _make_scope(headers=None, method="POST", path="/chat/completions"):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }


def _make_receive(body=b"", chunks=None):
    """Create a receive callable. If chunks is given, deliver them in order."""
    if chunks is not None:
        queue = list(chunks)
    else:
        queue = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


def _make_send(captured):
    async def send(message):
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
            captured["headers"] = dict(message.get("headers", []))
        elif message["type"] == "http.response.body":
            captured.setdefault("chunks", []).append(message.get("body", b""))

    return send


def _echo_app(captured):
    """ASGI app that reads the body and echoes it back as JSON."""

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return
        headers = dict(scope.get("headers", []))
        response_body = json.dumps(
            {
                "received": body.decode("utf-8"),
                "content_encoding": headers.get(b"content-encoding", b"").decode(),
                "content_length": headers.get(b"content-length", b"").decode(),
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(response_body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": response_body})

    return app


def _run_middleware(headers, body=b"", chunks=None):
    """Run the middleware with the given headers/body and return (status, json_body, raw_text)."""
    captured = {}
    scope = _make_scope(headers)
    receive = _make_receive(body, chunks)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(_echo_app(captured))
    asyncio.run(middleware(scope, receive, send))
    raw = b"".join(captured.get("chunks", []))
    return captured.get("status"), raw, captured


# ---------------------------------------------------------------------------
# gzip decompression
# ---------------------------------------------------------------------------


def test_should_decompress_gzip_request_body():
    """Gzip body with Content-Encoding: gzip is decompressed for downstream."""
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200
    data = json.loads(raw)
    assert json.loads(data["received"]) == PAYLOAD
    assert data["content_encoding"] == ""
    assert data["content_length"] == str(len(json.dumps(PAYLOAD).encode("utf-8")))


# ---------------------------------------------------------------------------
# no Content-Encoding header
# ---------------------------------------------------------------------------


def test_should_pass_through_when_no_content_encoding():
    """No Content-Encoding header -> body passes through untouched."""
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware({"content-type": "application/json"}, body=raw_json)
    assert status == 200
    data = json.loads(raw)
    assert json.loads(data["received"]) == PAYLOAD
    assert data["content_encoding"] == ""


# ---------------------------------------------------------------------------
# non-gzip Content-Encoding values -> passthrough
# ---------------------------------------------------------------------------


def test_should_pass_through_non_gzip_encoding_deflate():
    """Content-Encoding: deflate -> passthrough."""
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "deflate"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


def test_should_pass_through_non_gzip_encoding_br():
    """Content-Encoding: br -> passthrough."""
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "br"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


def test_should_pass_through_non_gzip_encoding_identity():
    """Content-Encoding: identity -> passthrough."""
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "identity"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


# ---------------------------------------------------------------------------
# case-insensitive and whitespace-tolerant Content-Encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoding_header",
    ["gzip", "GZIP", "Gzip", "  gzip  ", " GZIP ", "\tgzip\t"],
)
def test_should_handle_case_and_whitespace_variants(encoding_header):
    """Content-Encoding is case-insensitive and whitespace-tolerant."""
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": encoding_header},
        body=compressed,
    )
    assert status == 200
    assert json.loads(json.loads(raw)["received"]) == PAYLOAD


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------


def test_should_return_400_on_invalid_gzip_body():
    """Invalid gzip data with Content-Encoding: gzip -> 400."""
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=b"this is definitely not gzip data",
    )
    assert status == 400
    assert b"gzip" in raw.lower()


def test_should_return_400_on_empty_gzip_body():
    """Empty body with Content-Encoding: gzip is malformed -> 400."""
    status, _, _ = _run_middleware(
        {"content-encoding": "gzip"},
        body=b"",
    )
    assert status == 400


# ---------------------------------------------------------------------------
# chunked ASGI body
# ---------------------------------------------------------------------------


def test_should_reassemble_chunked_asgi_body():
    """Body delivered in multiple http.request frames is reassembled correctly."""
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    third = max(1, len(compressed) // 3)
    chunks = [
        {"type": "http.request", "body": compressed[:third], "more_body": True},
        {"type": "http.request", "body": compressed[third : 2 * third], "more_body": True},
        {"type": "http.request", "body": compressed[2 * third :], "more_body": False},
    ]
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        chunks=chunks,
    )
    assert status == 200
    assert json.loads(json.loads(raw)["received"]) == PAYLOAD


# ---------------------------------------------------------------------------
# non-http scope
# ---------------------------------------------------------------------------


def test_should_pass_through_non_http_scope():
    """Lifespan/websocket scopes pass through untouched."""
    called = {}

    async def dummy_app(scope, receive, send):
        called["scope_type"] = scope["type"]

    middleware = GunzipRequestMiddleware(dummy_app)
    asyncio.run(middleware({"type": "lifespan"}, None, None))
    assert called["scope_type"] == "lifespan"


# ---------------------------------------------------------------------------
# client disconnect
# ---------------------------------------------------------------------------


def test_should_handle_disconnect_during_body_read():
    """Client disconnect during body read -> middleware exits cleanly."""
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    scope = _make_scope({"content-encoding": "gzip"})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(app)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_called is False


# ---------------------------------------------------------------------------
# only decompressed result reaches downstream
# ---------------------------------------------------------------------------


def test_should_only_pass_decompressed_result():
    """Verify that ONLY the decompressed result reaches downstream."""
    payload = {"data": "sensitive_content_" * 200}
    raw_json = json.dumps(payload).encode("utf-8")
    compressed = gzip.compress(raw_json)
    assert compressed != raw_json

    downstream_body = None

    async def capturing_app(scope, receive, send):
        nonlocal downstream_body
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return
        downstream_body = body

    scope = _make_scope({"content-type": "application/json", "content-encoding": "gzip"})
    body_queue = [{"type": "http.request", "body": compressed, "more_body": False}]

    async def receive():
        return body_queue.pop(0) if body_queue else {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(capturing_app)
    asyncio.run(middleware(scope, receive, send))

    assert downstream_body == raw_json
    assert json.loads(downstream_body) == payload
    assert downstream_body != compressed

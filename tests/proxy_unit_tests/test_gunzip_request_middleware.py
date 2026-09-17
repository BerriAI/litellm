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
- truncated gzip data (raises EOFError) -> 400
- corrupt DEFLATE data (raises zlib.error) -> 400
- empty body with Content-Encoding: gzip -> 400
- decompressed body exceeding size limit -> 413
- chunked ASGI body (more_body=True) reassembled correctly
- non-http scope (lifespan) -> passthrough
- client disconnect during body read -> clean exit
- only decompressed result reaches downstream
- post-body receive() delegates to original receive (disconnect propagation)
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
    captured = {}
    scope = _make_scope(headers)
    receive = _make_receive(body, chunks)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(_echo_app(captured))
    asyncio.run(middleware(scope, receive, send))
    raw = b"".join(captured.get("chunks", []))
    return captured.get("status"), raw, captured


def test_should_decompress_gzip_request_body():
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


def test_should_pass_through_when_no_content_encoding():
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware({"content-type": "application/json"}, body=raw_json)
    assert status == 200
    data = json.loads(raw)
    assert json.loads(data["received"]) == PAYLOAD
    assert data["content_encoding"] == ""


def test_should_pass_through_non_gzip_encoding_deflate():
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "deflate"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


def test_should_pass_through_non_gzip_encoding_br():
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "br"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


def test_should_pass_through_non_gzip_encoding_identity():
    raw_json = json.dumps(PAYLOAD).encode("utf-8")
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "identity"},
        body=raw_json,
    )
    assert status == 200
    assert json.loads(raw)["received"] == json.dumps(PAYLOAD)


@pytest.mark.parametrize(
    "encoding_header",
    ["gzip", "GZIP", "Gzip", "  gzip  ", " GZIP ", "\tgzip\t"],
)
def test_should_handle_case_and_whitespace_variants(encoding_header):
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": encoding_header},
        body=compressed,
    )
    assert status == 200
    assert json.loads(json.loads(raw)["received"]) == PAYLOAD


def test_should_return_400_on_invalid_gzip_body():
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=b"this is definitely not gzip data",
    )
    assert status == 400
    assert b"gzip" in raw.lower()


def test_should_return_400_on_truncated_gzip_body():
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    truncated = compressed[:-5]
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=truncated,
    )
    assert status == 400


def test_should_return_400_on_corrupt_deflate_data():
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    corrupt = bytearray(compressed)
    corrupt[10] ^= 0xFF
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=bytes(corrupt),
    )
    assert status == 400


def test_should_return_400_on_empty_gzip_body():
    status, _, _ = _run_middleware(
        {"content-encoding": "gzip"},
        body=b"",
    )
    assert status == 400


def test_should_return_413_on_decompressed_body_exceeding_limit():
    from litellm.proxy.middleware.gunzip_request_middleware import (
        MAX_DECOMPRESSED_SIZE,
    )

    decompressed_size = MAX_DECOMPRESSED_SIZE + 1
    huge_payload = b"\x00" * decompressed_size
    compressed = gzip.compress(huge_payload)
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 413
    assert b"size" in raw.lower() or b"exceed" in raw.lower()


def test_should_reassemble_chunked_asgi_body():
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


def test_should_pass_through_non_http_scope():
    called = {}

    async def dummy_app(scope, receive, send):
        called["scope_type"] = scope["type"]

    middleware = GunzipRequestMiddleware(dummy_app)
    asyncio.run(middleware({"type": "lifespan"}, None, None))
    assert called["scope_type"] == "lifespan"


def test_should_handle_disconnect_during_body_read():
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


def test_should_only_pass_decompressed_result():
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


def test_should_delegate_to_original_receive_after_body():
    """After the decompressed body is delivered, subsequent receive() calls
    must delegate to the original receive so disconnects propagate."""
    receive_calls = []

    async def capturing_app(scope, receive, send):
        first = await receive()
        receive_calls.append(first)
        second = await receive()
        receive_calls.append(second)

    payload = {"msg": "test"}
    compressed = gzip.compress(json.dumps(payload).encode("utf-8"))

    scope = _make_scope({"content-encoding": "gzip"})

    original_calls = 0

    async def original_receive():
        nonlocal original_calls
        original_calls += 1
        return {"type": "http.disconnect"}

    body_queue = [{"type": "http.request", "body": compressed, "more_body": False}]

    async def receive():
        if body_queue:
            return body_queue.pop(0)
        return await original_receive()

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(capturing_app)
    asyncio.run(middleware(scope, receive, send))

    assert receive_calls[0]["type"] == "http.request"
    assert receive_calls[0]["body"] == json.dumps(payload).encode("utf-8")
    assert receive_calls[1]["type"] == "http.disconnect"
    assert original_calls == 1

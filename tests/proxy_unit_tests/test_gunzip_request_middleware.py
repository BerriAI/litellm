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
- truncated gzip data -> 400
- corrupt gzip data -> 400
- empty body with Content-Encoding: gzip -> passthrough to app
- decompressed body exceeding size limit -> 413
- chunked ASGI body (more_body=True) reassembled correctly
- non-http scope (lifespan) -> passthrough
- client disconnect during body read -> clean exit
- only decompressed result reaches downstream
- post-body receive() delegates to original receive (disconnect propagation)
- concatenated gzip members -> all members decompressed
- pass-through endpoints are skipped (signed gzip bodies preserved)
- premium-gated size limit via max_request_size_mb
- non-premium fallback to 100MB default
- error responses use JSON format
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
    assert "gzip" in json.loads(raw)["error"].lower()


def test_should_return_400_on_truncated_gzip_body():
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    truncated = compressed[:-5]
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=truncated,
    )
    assert status == 400


def test_should_return_400_on_truncated_gzip_header_only():
    """A gzip header with no deflate data (truncated before any output) should
    return 400, not be treated as a valid empty body."""
    header_only = gzip.compress(b"")[:10]  # gzip header, no deflate data
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=header_only,
    )
    assert status == 400


def test_should_return_400_on_corrupt_gzip_data():
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    corrupt = bytearray(compressed)
    corrupt[10] ^= 0xFF
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=bytes(corrupt),
    )
    assert status == 400


def test_should_decompress_valid_empty_gzip_stream():
    """A complete gzip stream of empty content (gzip.compress(b"")) decompresses
    to b"" and is passed to app as empty body."""
    compressed = gzip.compress(b"")
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200
    assert json.loads(raw)["received"] == ""


def test_should_handle_empty_gzip_body():
    """Empty body with Content-Encoding: gzip is passed to app as empty body
    with Content-Length: 0 and Content-Encoding stripped."""
    downstream_body = None
    downstream_headers = {}

    async def app(scope, receive, send):
        nonlocal downstream_body, downstream_headers
        downstream_headers = dict(scope.get("headers", []))
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body"):
                    break
            elif msg["type"] == "http.disconnect":
                return
        downstream_body = body

    scope = _make_scope({"content-encoding": "gzip"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(app)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_body == b""
    assert b"content-encoding" not in downstream_headers
    assert downstream_headers.get(b"content-length") == b"0"


def test_should_return_413_on_decompressed_body_exceeding_limit(monkeypatch):
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0.0009765625")
    decompressed_size = 2048
    huge_payload = b"\x00" * decompressed_size
    compressed = gzip.compress(huge_payload)
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 413
    assert "size" in json.loads(raw)["error"].lower() or "exceed" in json.loads(raw)["error"].lower()


def test_should_decompress_within_default_limit():
    """Normal payload within 100MB default limit works fine."""
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200
    assert json.loads(json.loads(raw)["received"]) == PAYLOAD


def test_should_use_get_max_size_when_premium_enabled():
    """When premium is enabled and get_max_size returns MB, it is converted to bytes."""

    def custom_get_max_size():
        return 1  # 1 MB

    decompressed_size = 2 * 1024 * 1024  # 2 MB > 1 MB limit
    huge_payload = b"\x00" * decompressed_size
    compressed = gzip.compress(huge_payload)
    captured = {}
    scope = _make_scope({"content-type": "application/octet-stream", "content-encoding": "gzip"})
    receive = _make_receive(compressed)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(_echo_app(captured), get_max_size=custom_get_max_size, is_premium=lambda: True)
    asyncio.run(middleware(scope, receive, send))
    assert captured.get("status") == 413


def test_should_be_unlimited_when_max_request_size_mb_is_zero():
    """Premium user with max_request_size_mb <= 0 gets unlimited decompression."""
    import os as _os

    payload = _os.urandom(200 * 1024)
    compressed = gzip.compress(payload)

    downstream_body = None

    async def binary_app(scope, receive, send):
        nonlocal downstream_body
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body"):
                    break
            elif msg["type"] == "http.disconnect":
                return
        downstream_body = body

    captured = {}
    scope = _make_scope({"content-type": "application/octet-stream", "content-encoding": "gzip"})
    receive = _make_receive(compressed)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(binary_app, get_max_size=lambda: 0, is_premium=lambda: True)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_body == payload


def test_should_allow_unlimited_when_env_var_is_zero(monkeypatch):
    """LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB=0 means unlimited (no size check).

    Sets a small _DEFAULT_MAX_DECOMPRESSED_SIZE so that without env=0 the
    payload would be rejected, proving that 0 truly disables the limit.
    """
    monkeypatch.setattr("litellm.proxy.middleware.gunzip_request_middleware._DEFAULT_MAX_DECOMPRESSED_SIZE", 100)
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0")

    import os as _os

    payload = _os.urandom(200 * 1024)
    compressed = gzip.compress(payload)

    downstream_body = None

    async def binary_app(scope, receive, send):
        nonlocal downstream_body
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body"):
                    break
            elif msg["type"] == "http.disconnect":
                return
        downstream_body = body

    captured = {}
    scope = _make_scope({"content-type": "application/octet-stream", "content-encoding": "gzip"})
    receive = _make_receive(compressed)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(binary_app)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_body == payload


def test_should_fallback_on_invalid_env_var(monkeypatch, caplog):
    """Invalid LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB logs error and uses default."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "abc")
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    with caplog.at_level("ERROR"):
        status, _, _ = _run_middleware(
            {"content-type": "application/json", "content-encoding": "gzip"},
            body=compressed,
        )
    assert status == 200
    assert any("Invalid LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB" in record.message for record in caplog.records)


def test_should_fallback_on_nan_env_var(monkeypatch):
    """nan env var falls back to default instead of 500."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "nan")
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200


def test_should_fallback_on_inf_env_var(monkeypatch):
    """inf env var falls back to default instead of 500."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "inf")
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, _, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200


def test_should_warn_on_zero_env_var(monkeypatch, caplog):
    """LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB=0 warns about disabled bomb protection."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0")
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    with caplog.at_level("WARNING"):
        status, _, _ = _run_middleware(
            {"content-type": "application/json", "content-encoding": "gzip"},
            body=compressed,
        )
    assert status == 200
    assert any("gzip bomb" in record.message for record in caplog.records)


def test_should_fallback_to_default_when_not_premium(monkeypatch):
    """Non-premium users get the env var limit, not the configured max_request_size_mb."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0.0001")
    payload = b"\x00" * 200  # 200 bytes > ~100 byte limit
    compressed = gzip.compress(payload)
    captured = {}
    scope = _make_scope({"content-type": "application/octet-stream", "content-encoding": "gzip"})
    receive = _make_receive(compressed)
    send = _make_send(captured)
    middleware = GunzipRequestMiddleware(_echo_app(captured), get_max_size=lambda: 999, is_premium=lambda: False)
    asyncio.run(middleware(scope, receive, send))
    assert captured.get("status") == 413


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


def test_should_handle_empty_first_member_then_data():
    """Concatenated gzip with empty first member + data in second member.
    The empty first member should not cause early return b"" that drops the
    second member's data."""
    member1 = gzip.compress(b"")
    member2 = gzip.compress(b"hello")
    concatenated = member1 + member2
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 200
    assert json.loads(raw)["received"] == "hello"


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


def test_should_handle_concatenated_gzip_members():
    """Multiple gzip members concatenated in one stream are all decompressed."""
    payload_a = json.dumps({"part": "a", "data": "x" * 100}).encode("utf-8")
    payload_b = json.dumps({"part": "b", "data": "y" * 100}).encode("utf-8")
    combined = payload_a + payload_b
    member_a = gzip.compress(payload_a)
    member_b = gzip.compress(payload_b)
    concatenated = member_a + member_b

    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 200
    data = json.loads(raw)
    assert data["received"].encode("utf-8") == combined


def test_should_handle_concatenated_gzip_members_in_chunks():
    """Concatenated gzip members arriving across multiple ASGI chunks."""
    payload_a = json.dumps({"part": "a"}).encode("utf-8")
    payload_b = json.dumps({"part": "b"}).encode("utf-8")
    combined = payload_a + payload_b
    member_a = gzip.compress(payload_a)
    member_b = gzip.compress(payload_b)
    concatenated = member_a + member_b
    half = len(concatenated) // 2

    chunks = [
        {"type": "http.request", "body": concatenated[:half], "more_body": True},
        {"type": "http.request", "body": concatenated[half:], "more_body": False},
    ]
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        chunks=chunks,
    )
    assert status == 200
    assert json.loads(raw)["received"].encode("utf-8") == combined


def test_should_return_413_on_concatenated_gzip_exceeding_limit(monkeypatch):
    """Size limit applies across all concatenated members, not per-member."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0.00048828125")
    payload_a = b"\x00" * 400
    payload_b = b"\x00" * 400
    concatenated = gzip.compress(payload_a) + gzip.compress(payload_b)

    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 413


def test_should_return_413_on_streaming_decompress_exceeding_limit(monkeypatch):
    """Size limit triggers during streaming decompress, before flush()."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "9.5367431640625e-05")
    payload = b"\x00" * 200
    compressed = gzip.compress(payload)
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 413


def test_should_return_413_during_concatenated_member_chunk_decompress(monkeypatch):
    """Size limit triggers during _decompress_chunk of a concatenated member, before flush."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "9.5367431640625e-05")
    payload_a = b"\x00" * 50
    payload_b = b"\x00" * 200
    concatenated = gzip.compress(payload_a) + gzip.compress(payload_b)
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 413


def test_should_handle_large_chunk_triggering_unconsumed_tail():
    """A single large compressed chunk decompressing beyond _DECOMPRESS_BUFSIZE
    triggers the unconsumed_tail drain loop in _decompress_chunk."""
    import os as _os

    payload = _os.urandom(200 * 1024)
    compressed = gzip.compress(payload)

    downstream_body = None

    async def binary_app(scope, receive, send):
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

    scope = _make_scope({"content-type": "application/octet-stream", "content-encoding": "gzip"})

    async def receive():
        return {"type": "http.request", "body": compressed, "more_body": False}

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(binary_app)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_body == payload


def test_should_skip_non_request_non_disconnect_messages():
    """ASGI messages that are neither http.request nor http.disconnect are skipped."""
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

    scope = _make_scope({"content-encoding": "gzip"})
    compressed = gzip.compress(b'{"ok":true}')
    msg_queue = [
        {"type": "lifespan.startup"},
        {"type": "http.request", "body": compressed, "more_body": False},
    ]

    async def receive():
        return msg_queue.pop(0) if msg_queue else {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    middleware = GunzipRequestMiddleware(capturing_app)
    asyncio.run(middleware(scope, receive, send))
    assert downstream_body == b'{"ok":true}'


def test_should_skip_pass_through_endpoints():
    """Pass-through endpoints (e.g. /bedrock, /anthropic) are not decompressed
    because clients may send gzip bodies that are part of a signature."""
    compressed = gzip.compress(b'{"ok":true}')
    downstream_body = None
    downstream_encoding = None

    async def capturing_app(scope, receive, send):
        nonlocal downstream_body, downstream_encoding
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
        headers = dict(scope.get("headers", []))
        downstream_encoding = headers.get(b"content-encoding", b"").decode()

    for prefix in ["/bedrock", "/anthropic", "/vertex-ai", "/openai"]:
        scope = _make_scope({"content-encoding": "gzip"}, path=prefix + "/v1/test")
        downstream_body = None
        downstream_encoding = None

        async def receive():
            return {"type": "http.request", "body": compressed, "more_body": False}

        async def send(message):
            pass

        middleware = GunzipRequestMiddleware(capturing_app)
        asyncio.run(middleware(scope, receive, send))
        # Body is untouched (still compressed), content-encoding preserved
        assert downstream_body == compressed, f"Failed for {prefix}"
        assert downstream_encoding == "gzip", f"Failed for {prefix}"


def test_should_decompress_non_pass_through_paths():
    """Non-pass-through paths like /chat/completions are still decompressed."""
    compressed = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
    status, raw, _ = _run_middleware(
        {"content-type": "application/json", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 200
    assert json.loads(json.loads(raw)["received"]) == PAYLOAD


def test_should_return_413_on_drain_loop_overflow(monkeypatch):
    """Size limit triggers inside _decompress_chunk drain loop, not just the
    initial decompress call."""
    import os as _os

    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0.0625009536743164")
    payload = _os.urandom(200 * 1024)
    compressed = gzip.compress(payload)
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 413


def test_should_return_400_on_truncated_second_concatenated_member():
    """A truncated second gzip member in a concatenated stream returns 400."""
    payload_a = json.dumps({"part": "a"}).encode("utf-8")
    member_a = gzip.compress(payload_a)
    member_b_truncated = gzip.compress(b"x" * 100)[:-5]
    concatenated = member_a + member_b_truncated

    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 400


def test_should_return_400_on_trailing_junk_after_gzip():
    """Trailing non-gzip junk after a valid gzip member returns 400."""
    member = gzip.compress(b"\x00" * 50)
    junk = b"\x00" * 200
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=member + junk,
    )
    assert status == 400


def test_should_return_400_on_too_many_concatenated_members():
    """Concatenated gzip streams with more than _MAX_GZIP_MEMBERS+1 return 400
    (first member processed in _stream_decompress, _MAX_GZIP_MEMBERS in _drain_remaining_members)."""
    import litellm.proxy.middleware.gunzip_request_middleware as mod

    member = gzip.compress(b"x")
    concatenated = member * (mod._MAX_GZIP_MEMBERS + 2)
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 400


def test_should_allow_max_concatenated_members():
    """Concatenated gzip streams with exactly _MAX_GZIP_MEMBERS+1 succeed."""
    import litellm.proxy.middleware.gunzip_request_middleware as mod

    member = gzip.compress(b"x")
    concatenated = member * (mod._MAX_GZIP_MEMBERS + 1)
    status, raw, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=concatenated,
    )
    assert status == 200
    assert json.loads(raw)["received"] == "x" * (mod._MAX_GZIP_MEMBERS + 1)


def test_should_return_413_on_decompress_exactly_at_limit(monkeypatch):
    """Decompressed output that exactly reaches max_size boundary triggers 413."""
    monkeypatch.setenv("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB", "0.0001")  # ~104 bytes
    payload = b"\x00" * 200
    compressed = gzip.compress(payload)
    status, _, _ = _run_middleware(
        {"content-type": "application/octet-stream", "content-encoding": "gzip"},
        body=compressed,
    )
    assert status == 413

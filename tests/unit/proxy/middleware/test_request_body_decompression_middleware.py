import gzip
import json
import sys
import tracemalloc
import zlib
from collections.abc import Callable
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.request_body_decompression_middleware import (
    RFC_9659_ZSTD_WINDOW_LOG_MAX,
    RequestBodyDecompressionMiddleware,
)

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

MIB: Final = 1024 * 1024
CAP_MB: Final = 1.0
REQUEST_JSON: Final = json.dumps(
    {"model": "jev", "state": "order history " * 200, "questions": {"is_refund_request": {"type": "noul"}}}
).encode()


async def _echo_request(request: Request) -> Response:
    body: Final = await request.body()
    return JSONResponse(
        {
            "body_hex": body.hex(),
            "content_encoding": request.headers.getlist("content-encoding"),
            "content_length": request.headers.get("content-length"),
        }
    )


def _client(max_decompressed_body_size_mb: float = CAP_MB) -> TestClient:
    echo_app: Final = Starlette(routes=[Route("/v1/systemone", _echo_request, methods=["POST"])])
    return TestClient(
        RequestBodyDecompressionMiddleware(echo_app, max_decompressed_body_size_mb=max_decompressed_body_size_mb)
    )


def _post(client: TestClient, body: bytes, content_encodings: tuple[str, ...]) -> Response:
    headers: Final = [("content-type", "application/json"), *(("content-encoding", v) for v in content_encodings)]
    return client.post("/v1/systemone", content=body, headers=headers)


def _zeros_compressed(compress_chunk: Callable[[bytes], bytes], flush: Callable[[], bytes], decoded_mib: int) -> bytes:
    zero_mib: Final = bytes(MIB)
    return b"".join([*(compress_chunk(zero_mib) for _ in range(decoded_mib)), flush()])


def _gzip_bomb(decoded_mib: int) -> bytes:
    compressor: Final = zlib.compressobj(9, zlib.DEFLATED, 31)
    return _zeros_compressed(compressor.compress, compressor.flush, decoded_mib)


def _zstd_bomb(decoded_mib: int) -> bytes:
    compressor: Final = zstd.ZstdCompressor()
    return _zeros_compressed(compressor.compress, compressor.flush, decoded_mib)


@pytest.mark.parametrize(
    ("compressed", "content_encoding"),
    [
        (gzip.compress(REQUEST_JSON), "gzip"),
        (zstd.compress(REQUEST_JSON), "zstd"),
        (gzip.compress(REQUEST_JSON), " GZip "),
        (zstd.compress(REQUEST_JSON), "ZSTD"),
    ],
)
def test_supported_encoding_reaches_the_app_decoded(compressed: bytes, content_encoding: str):
    response: Final = _post(_client(), compressed, (content_encoding,))

    assert response.status_code == 200
    assert response.json() == {
        "body_hex": REQUEST_JSON.hex(),
        "content_encoding": [],
        "content_length": str(len(REQUEST_JSON)),
    }


@pytest.mark.parametrize(
    ("body", "content_encodings"),
    [
        (gzip.compress(REQUEST_JSON), ()),
        (gzip.compress(REQUEST_JSON), ("identity",)),
        (b"\x1b\x00brotli-ish", ("br",)),
        (zlib.compress(REQUEST_JSON), ("deflate",)),
        (gzip.compress(REQUEST_JSON), ("gzip, zstd",)),
        (gzip.compress(REQUEST_JSON), ("gzip", "gzip")),
    ],
)
def test_other_encodings_reach_the_app_untouched(body: bytes, content_encodings: tuple[str, ...]):
    response: Final = _post(_client(), body, content_encodings)

    assert response.status_code == 200
    assert response.json() == {
        "body_hex": body.hex(),
        "content_encoding": list(content_encodings),
        "content_length": str(len(body)),
    }


@pytest.mark.parametrize("content_encoding", ["gzip", "zstd"])
def test_empty_body_with_supported_encoding_reaches_the_app_empty(content_encoding: str):
    response: Final = _post(_client(), b"", (content_encoding,))

    assert response.status_code == 200
    assert response.json() == {"body_hex": "", "content_encoding": [], "content_length": "0"}


@pytest.mark.parametrize(("bomb", "content_encoding"), [(_gzip_bomb, "gzip"), (_zstd_bomb, "zstd")])
def test_decompression_bomb_is_rejected_without_inflating_past_the_cap(
    bomb: Callable[[int], bytes], content_encoding: str
):
    decoded_mib: Final = 200
    compressed: Final = bomb(decoded_mib)
    client: Final = _client()
    tracemalloc.start()
    try:
        response = _post(client, compressed, (content_encoding,))
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(compressed) < CAP_MB * MIB
    assert response.status_code == 413
    assert response.json() == {"error": f"Request size is too large. Max size is {CAP_MB} MB"}
    assert peak_bytes < decoded_mib * MIB / 10


@pytest.mark.parametrize("content_encoding", ["gzip", "zstd"])
def test_compressed_body_over_the_cap_is_rejected_before_decoding(content_encoding: str):
    not_compressed_and_over_cap: Final = b"\x00" * (int(CAP_MB * MIB) + 1)

    response: Final = _post(_client(), not_compressed_and_over_cap, (content_encoding,))

    assert response.status_code == 413
    assert response.json() == {"error": f"Request size is too large. Max size is {CAP_MB} MB"}


def test_body_decoding_to_exactly_the_cap_reaches_the_app():
    exactly_cap: Final = b"x" * int(CAP_MB * MIB)

    response: Final = _post(_client(), zstd.compress(exactly_cap), ("zstd",))

    assert response.status_code == 200
    assert response.json()["content_length"] == str(len(exactly_cap))


def _zstd_frame_needing_a_window_over_the_rfc_9659_limit() -> bytes:
    window_log: Final = RFC_9659_ZSTD_WINDOW_LOG_MAX + 1
    content: Final = bytes(2**RFC_9659_ZSTD_WINDOW_LOG_MAX + MIB)
    return zstd.compress(content, options={zstd.CompressionParameter.window_log: window_log})


@pytest.mark.parametrize(
    ("body", "content_encoding", "message"),
    [
        (b"not gzip at all", "gzip", "Request body is not valid gzip data: "),
        (b"not zstd at all", "zstd", "Request body is not valid zstd data: "),
        (gzip.compress(REQUEST_JSON)[:-8], "gzip", "Request body ends before the end of its gzip stream"),
        (zstd.compress(REQUEST_JSON)[:-8], "zstd", "Request body ends before the end of its zstd stream"),
        (gzip.compress(REQUEST_JSON) + b"{}", "gzip", "Request body has data after the end of its gzip stream"),
        (zstd.compress(REQUEST_JSON) + b"{}", "zstd", "Request body has data after the end of its zstd stream"),
        (_zstd_frame_needing_a_window_over_the_rfc_9659_limit(), "zstd", "Request body is not valid zstd data: "),
    ],
)
def test_undecodable_body_is_rejected_with_an_openai_style_400(body: bytes, content_encoding: str, message: str):
    response: Final = _post(_client(max_decompressed_body_size_mb=50.0), body, (content_encoding,))

    assert response.status_code == 400
    error: Final = response.json()["error"]
    assert error["message"].startswith(message)
    assert {key: error[key] for key in ("type", "param", "code")} == {
        "type": "invalid_request_error",
        "param": None,
        "code": "400",
    }


def _http_scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {"type": "http", "method": "POST", "path": "/v1/systemone", "headers": headers}


def _scripted_receive(messages: list[Message]) -> Receive:
    async def receive() -> Message:
        return messages.pop(0)

    return receive


@pytest.mark.asyncio
async def test_chunked_body_is_decoded_whole_and_later_receives_reach_the_client():
    compressed: Final = zstd.compress(REQUEST_JSON)
    third: Final = len(compressed) // 3
    client_messages: Final[list[Message]] = [
        {"type": "http.request", "body": compressed[:third], "more_body": True},
        {"type": "http.request", "body": compressed[third : 2 * third], "more_body": True},
        {"type": "http.request", "body": compressed[2 * third :], "more_body": False},
        {"type": "http.disconnect"},
    ]
    seen: Final[list[tuple[Scope, list[Message]]]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append((scope, [await receive(), await receive()]))

    async def send(message: Message) -> None:
        raise AssertionError(f"middleware sent {message}")

    headers: Final = [
        (b"content-type", b"application/json"),
        (b"content-encoding", b"zstd"),
        (b"transfer-encoding", b"chunked"),
    ]
    await RequestBodyDecompressionMiddleware(app)(_http_scope(headers), _scripted_receive(client_messages), send)

    [(app_scope, app_messages)] = seen
    assert app_scope["headers"] == [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(REQUEST_JSON)).encode()),
    ]
    assert app_messages == [
        {"type": "http.request", "body": REQUEST_JSON, "more_body": False},
        {"type": "http.disconnect"},
    ]


@pytest.mark.asyncio
async def test_client_disconnect_mid_body_ends_the_request_without_a_response():
    compressed: Final = gzip.compress(REQUEST_JSON)
    client_messages: Final[list[Message]] = [
        {"type": "http.request", "body": compressed[:10], "more_body": True},
        {"type": "http.disconnect"},
    ]
    sent: Final[list[Message]] = []
    app_calls: Final[list[Scope]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        app_calls.append(scope)

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestBodyDecompressionMiddleware(app)(
        _http_scope([(b"content-encoding", b"gzip")]), _scripted_receive(client_messages), send
    )

    assert (app_calls, sent) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {"type": "lifespan"},
        {"type": "websocket", "path": "/v1/realtime", "headers": [(b"content-encoding", b"gzip")]},
    ],
)
async def test_non_http_scopes_reach_the_app_untouched(scope: Scope):
    seen: Final[list[tuple[Scope, Receive]]] = []

    async def receive() -> Message:
        raise AssertionError("middleware read from a non-http scope")

    async def app(app_scope: Scope, app_receive: Receive, send: Send) -> None:
        seen.append((app_scope, app_receive))

    async def send(message: Message) -> None:
        raise AssertionError(f"middleware sent {message}")

    await RequestBodyDecompressionMiddleware(app)(scope, receive, send)

    [(app_scope, app_receive)] = seen
    assert app_scope is scope
    assert app_receive is receive

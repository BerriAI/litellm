import asyncio
import base64
import gzip
import json
import math
import random
import struct
import sys
import tracemalloc
import zlib
from collections.abc import Callable
from types import MappingProxyType
from typing import Final

import anyio
import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.request_body_decompression_middleware import (
    GZIP,
    GZIP_MAGIC,
    RFC_9659_ZSTD_WINDOW_LOG_MAX,
    ZSTD_FRAME_MAGIC,
    RequestBodyDecompressionMiddleware,
    load_zstd_codec,
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


def _client(max_decompressed_body_size_mb: float = CAP_MB, **options: object) -> TestClient:
    echo_app: Final = Starlette(
        routes=[
            Route("/v1/systemone", _echo_request, methods=["POST"]),
            Route("/plugin-proxy/{path:path}", _echo_request, methods=["POST"]),
            Route("/claude_code_gateway/v1/{signal}", _echo_request, methods=["POST"]),
        ]
    )
    return TestClient(
        RequestBodyDecompressionMiddleware(
            echo_app,
            max_decompressed_body_size_mb=max_decompressed_body_size_mb,
            **options,  # pyright: ignore[reportArgumentType]  # each test passes one typed keyword
        )
    )


def _low_entropy_text(length: int) -> bytes:
    return bytes(random.Random(0).choices(b"abcd", k=length))


def _post(client: TestClient, body: bytes, content_encodings: tuple[str, ...], path: str = "/v1/systemone") -> Response:
    headers: Final = [("content-type", "application/json"), *(("content-encoding", v) for v in content_encodings)]
    return client.post(path, content=body, headers=headers)


def _decoded_too_large(max_mb: float) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": f"Request body is larger than {max_mb:g} MB when decoded",
            "type": "invalid_request_error",
            "param": None,
            "code": "413",
        }
    }


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
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), "gzip", id="gzip"),
        pytest.param(zstd.compress(REQUEST_JSON), "zstd", id="zstd"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), " GZip ", id="gzip-mixed-case-padded"),
        pytest.param(zstd.compress(REQUEST_JSON), "ZSTD", id="zstd-upper-case"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), "x-gzip", id="x-gzip"),
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
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), (), id="no-header"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), ("identity",), id="identity"),
        pytest.param(b"\x1b\x00brotli-ish", ("br",), id="br"),
        pytest.param(REQUEST_JSON, ("compress",), id="compress"),
        pytest.param(zlib.compress(REQUEST_JSON), ("deflate",), id="deflate"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), ("gzip, zstd",), id="stacked-in-one-header"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), ("gzip", "gzip"), id="two-headers"),
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


@pytest.mark.parametrize(
    ("body", "content_encoding"),
    [
        pytest.param(REQUEST_JSON, "gzip", id="plain-labeled-gzip"),
        pytest.param(REQUEST_JSON, "zstd", id="plain-labeled-zstd"),
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), "zstd", id="gzip-labeled-zstd"),
        pytest.param(zstd.compress(REQUEST_JSON), "gzip", id="zstd-labeled-gzip"),
        pytest.param(_low_entropy_text(2 * int(CAP_MB * MIB)), "gzip", id="plain-past-the-cap-labeled-gzip"),
        pytest.param(_low_entropy_text(2 * int(CAP_MB * MIB)), "zstd", id="plain-past-the-cap-labeled-zstd"),
        pytest.param(b"{}", "zstd", id="tiny-plain-labeled-zstd"),
    ],
)
def test_body_that_is_not_in_its_labeled_encoding_reaches_the_app_as_sent(body: bytes, content_encoding: str):
    response: Final = _post(_client(), body, (content_encoding,))

    assert response.status_code == 200
    assert response.json() == {"body_hex": body.hex(), "content_encoding": [], "content_length": str(len(body))}


@pytest.mark.parametrize("content_encoding", ["gzip", "zstd"])
def test_empty_body_with_supported_encoding_reaches_the_app_empty(content_encoding: str):
    response: Final = _post(_client(), b"", (content_encoding,))

    assert response.status_code == 200
    assert response.json() == {"body_hex": "", "content_encoding": [], "content_length": "0"}


@pytest.mark.parametrize(("bomb", "content_encoding"), [(_gzip_bomb, "gzip"), (_zstd_bomb, "zstd")])
def test_decompression_bomb_is_rejected_without_inflating_past_the_ratio_limit(
    bomb: Callable[[int], bytes], content_encoding: str
):
    decoded_mib: Final = 200
    compressed: Final = bomb(decoded_mib)
    client: Final = _client(max_decompressed_body_size_mb=decoded_mib * 2)
    tracemalloc.start()
    try:
        response = _post(client, compressed, (content_encoding,))
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert response.status_code == 413
    assert response.json()["error"]["message"] == "Request body expands more than 200 times when decoded"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert peak_bytes < decoded_mib * MIB / 2


def test_ratio_limit_rejects_only_a_body_that_expands_past_it():
    compressed: Final = zstd.compress(REQUEST_JSON)
    ratio: Final = math.ceil(len(REQUEST_JSON) / len(compressed))

    at_limit: Final = _post(_client(max_decompression_ratio=ratio), compressed, ("zstd",))
    over_limit: Final = _post(_client(max_decompression_ratio=ratio - 1), compressed, ("zstd",))

    assert at_limit.status_code == 200
    assert over_limit.status_code == 413


def test_body_decoding_past_the_cap_is_rejected():
    compressed: Final = zstd.compress(_low_entropy_text(2 * int(CAP_MB * MIB)))

    response: Final = _post(_client(), compressed, ("zstd",))

    assert len(compressed) < CAP_MB * MIB
    assert response.status_code == 413
    assert response.json() == _decoded_too_large(CAP_MB)


@pytest.mark.parametrize(
    ("operator_max_mb", "effective_max_mb"),
    [(0.25, 0.25), (CAP_MB * 4, CAP_MB), (None, CAP_MB), (0, CAP_MB)],
)
def test_operator_request_size_limit_caps_the_decoded_body_when_it_is_lower(
    operator_max_mb: float | None, effective_max_mb: float
):
    client: Final = _client(get_max_request_size_mb=lambda: operator_max_mb)
    within: Final = zstd.compress(_low_entropy_text(int(effective_max_mb * MIB)))
    over: Final = zstd.compress(_low_entropy_text(int(effective_max_mb * MIB) + 1))

    assert _post(client, within, ("zstd",)).status_code == 200
    over_response: Final = _post(client, over, ("zstd",))
    assert over_response.status_code == 413
    assert over_response.json() == _decoded_too_large(effective_max_mb)


@pytest.mark.parametrize(("signature", "content_encoding"), [(GZIP_MAGIC, "gzip"), (ZSTD_FRAME_MAGIC, "zstd")])
def test_compressed_body_over_the_cap_is_rejected_before_decoding(signature: bytes, content_encoding: str):
    over_cap_on_the_wire: Final = signature + bytes(int(CAP_MB * MIB))

    response: Final = _post(_client(), over_cap_on_the_wire, (content_encoding,))

    assert response.status_code == 413
    assert response.json() == {"error": f"Request size is too large. Max size is {CAP_MB} MB"}


def test_body_decoding_to_exactly_the_cap_reaches_the_app():
    exactly_cap: Final = _low_entropy_text(int(CAP_MB * MIB))

    response: Final = _post(_client(), zstd.compress(exactly_cap), ("zstd",))

    assert response.status_code == 200
    assert response.json()["content_length"] == str(len(exactly_cap))


def _zstd_frame_needing_a_window_over_the_rfc_9659_limit() -> bytes:
    window_log: Final = RFC_9659_ZSTD_WINDOW_LOG_MAX + 1
    content: Final = bytes(2**RFC_9659_ZSTD_WINDOW_LOG_MAX + MIB)
    return zstd.compress(content, options={zstd.CompressionParameter.window_log: window_log})


def _zstd_skippable_frame(content: bytes) -> bytes:
    return struct.pack("<II", 0x184D2A53, len(content)) + content


def _gzip_declaring_a_smaller_size(content: bytes) -> bytes:
    return gzip.compress(content, mtime=0)[:-4] + (len(content) // 2).to_bytes(4, "little")


@pytest.mark.parametrize(
    ("body", "content_encoding", "message"),
    [
        pytest.param(
            GZIP_MAGIC + b"not gzip at all", "gzip", "Request body is not valid gzip data: ", id="corrupt-gzip"
        ),
        pytest.param(
            ZSTD_FRAME_MAGIC + b"not zstd at all", "zstd", "Request body is not valid zstd data: ", id="corrupt-zstd"
        ),
        pytest.param(
            gzip.compress(REQUEST_JSON, mtime=0)[:-8],
            "gzip",
            "Request body ends before the end of its gzip stream",
            id="truncated-gzip",
        ),
        pytest.param(
            zstd.compress(REQUEST_JSON)[:-8],
            "zstd",
            "Request body ends before the end of its zstd stream",
            id="truncated-zstd",
        ),
        pytest.param(
            gzip.compress(REQUEST_JSON, mtime=0) + b"{}",
            "gzip",
            "Request body has data after the end of its gzip stream",
            id="gzip-trailing-data",
        ),
        pytest.param(
            zstd.compress(REQUEST_JSON) + b"{}",
            "zstd",
            "Request body has data after the end of its zstd stream",
            id="zstd-trailing-data",
        ),
        pytest.param(
            _zstd_frame_needing_a_window_over_the_rfc_9659_limit(),
            "zstd",
            "Request body is not valid zstd data: ",
            id="zstd-window-over-the-rfc-limit",
        ),
        pytest.param(
            _zstd_skippable_frame(b"meta") + zstd.compress(REQUEST_JSON),
            "zstd",
            "Request body has data after the end of its zstd stream",
            id="zstd-skippable-frame-first",
        ),
        pytest.param(
            _gzip_declaring_a_smaller_size(REQUEST_JSON),
            "gzip",
            "Request body decodes past the size its gzip data declares",
            id="gzip-declaring-a-smaller-size",
        ),
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


def test_zstd_body_gets_a_415_naming_gzip_when_this_python_cannot_decode_zstd():
    client: Final = _client(codecs=MappingProxyType({"gzip": GZIP}))

    zstd_response: Final = _post(client, zstd.compress(REQUEST_JSON), ("zstd",))
    gzip_response: Final = _post(client, gzip.compress(REQUEST_JSON, mtime=0), ("gzip",))
    plain_labeled_zstd_response: Final = _post(client, REQUEST_JSON, ("zstd",))

    assert zstd_response.status_code == 415
    assert zstd_response.headers["accept-encoding"] == "gzip"
    assert zstd_response.json()["error"]["message"] == (
        "This proxy cannot decode zstd request bodies. Supported encodings: gzip"
    )
    assert gzip_response.status_code == 200
    assert plain_labeled_zstd_response.json()["body_hex"] == REQUEST_JSON.hex()


def test_zstd_is_unavailable_without_failing_the_import_on_a_python_without_zstd(monkeypatch: pytest.MonkeyPatch):
    for module in ("_zstd", "compression.zstd", "backports.zstd"):
        monkeypatch.setitem(sys.modules, module, None)

    assert load_zstd_codec() is None


@pytest.mark.parametrize(
    "path",
    [
        "/plugin-proxy/notes/v1/run",
        "/claude_code_gateway/v1/metrics",
        "/claude_code_gateway/v1/logs",
        "/claude_code_gateway/v1/traces",
    ],
)
@pytest.mark.parametrize(
    ("body", "content_encoding"),
    [
        pytest.param(gzip.compress(REQUEST_JSON, mtime=0), "gzip", id="gzip"),
        pytest.param(zstd.compress(REQUEST_JSON), "zstd", id="zstd"),
        pytest.param(zstd.compress(_low_entropy_text(int(CAP_MB * MIB) * 2)), "zstd", id="zstd-past-the-cap"),
        pytest.param(_gzip_bomb(2), "gzip", id="gzip-past-the-ratio-limit"),
        pytest.param(
            gzip.compress(REQUEST_JSON, mtime=0) + gzip.compress(REQUEST_JSON, mtime=0), "gzip", id="gzip-two-members"
        ),
    ],
)
def test_raw_body_routes_get_the_body_as_sent(path: str, body: bytes, content_encoding: str):
    response: Final = _post(_client(codecs=MappingProxyType({"gzip": GZIP})), body, (content_encoding,), path=path)

    assert response.status_code == 200
    assert response.json() == {
        "body_hex": body.hex(),
        "content_encoding": [content_encoding],
        "content_length": str(len(body)),
    }


@pytest.mark.asyncio
async def test_compressed_bodies_past_the_in_flight_budget_get_a_503_until_earlier_ones_finish():
    compressed: Final = zstd.compress(REQUEST_JSON)
    app_has_the_first_body: Final = asyncio.Event()
    let_the_first_request_finish: Final = asyncio.Event()

    async def route_holding_the_first_request(request: Request) -> Response:
        body: Final = await request.body()
        if not app_has_the_first_body.is_set():
            app_has_the_first_body.set()
            await let_the_first_request_finish.wait()
        return JSONResponse({"decoded_bytes": len(body)})

    middleware: Final = RequestBodyDecompressionMiddleware(
        Starlette(routes=[Route("/v1/systemone", route_holding_the_first_request, methods=["POST"])]),
        max_decompressed_body_size_mb=CAP_MB,
        max_decoded_bodies_in_flight_mb=1.5 * len(REQUEST_JSON) / MIB,
    )
    headers: Final = {"content-type": "application/json", "content-encoding": "zstd"}
    with anyio.fail_after(30):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://proxy") as client:
            first: Final = asyncio.create_task(client.post("/v1/systemone", content=compressed, headers=headers))
            await app_has_the_first_body.wait()
            while_first_is_in_flight: Final = await client.post("/v1/systemone", content=compressed, headers=headers)
            let_the_first_request_finish.set()
            first_response: Final = await first
            after_first_finished: Final = await client.post("/v1/systemone", content=compressed, headers=headers)

    assert first_response.json() == {"decoded_bytes": len(REQUEST_JSON)}
    assert while_first_is_in_flight.status_code == 503
    assert while_first_is_in_flight.headers["retry-after"] == "1"
    assert while_first_is_in_flight.json()["error"]["type"] == "overloaded_error"
    assert after_first_finished.json() == {"decoded_bytes": len(REQUEST_JSON)}


@pytest.mark.asyncio
@pytest.mark.parametrize("compress", [gzip.compress, zstd.compress])
async def test_concurrent_compressed_bodies_count_their_declared_size_against_the_budget(
    compress: Callable[[bytes], bytes],
):
    hard_to_compress: Final = json.dumps(
        {"model": "jev", "image": base64.b64encode(random.Random(0).randbytes(48 * 1024)).decode()}
    ).encode()
    compressed: Final = compress(hard_to_compress)
    all_bodies_arrived: Final = asyncio.Event()
    bodies_seen: Final[list[int]] = []

    async def route_holding_every_request(request: Request) -> Response:
        bodies_seen.append(len(await request.body()))
        if len(bodies_seen) == 3:
            all_bodies_arrived.set()
        await all_bodies_arrived.wait()
        return JSONResponse({"decoded_bytes": bodies_seen[-1]})

    middleware: Final = RequestBodyDecompressionMiddleware(
        Starlette(routes=[Route("/v1/systemone", route_holding_every_request, methods=["POST"])]),
        max_decompressed_body_size_mb=CAP_MB,
        max_decoded_bodies_in_flight_mb=3.5 * len(hard_to_compress) / MIB,
    )
    headers: Final = {
        "content-type": "application/json",
        "content-encoding": "zstd" if compress is zstd.compress else "gzip",
    }
    with anyio.fail_after(30):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://proxy") as client:
            responses: Final = await asyncio.gather(
                *(client.post("/v1/systemone", content=compressed, headers=headers) for _ in range(3))
            )

    assert len(compressed) * 200 > CAP_MB * MIB
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert bodies_seen == [len(hard_to_compress)] * 3


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
        {"type": "http.request", "body": compressed[:2], "more_body": True},
        {"type": "http.request", "body": compressed[2:third], "more_body": True},
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
async def test_chunked_plain_body_labeled_gzip_reaches_the_app_in_the_chunks_sent():
    client_messages: Final[list[Message]] = [
        {"type": "http.request", "body": REQUEST_JSON[:1], "more_body": True},
        {"type": "http.request", "body": REQUEST_JSON[1:9], "more_body": True},
        {"type": "http.request", "body": REQUEST_JSON[9:], "more_body": False},
        {"type": "http.disconnect"},
    ]
    seen: Final[list[tuple[Scope, list[Message]]]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append((scope, [await receive(), await receive(), await receive()]))

    async def send(message: Message) -> None:
        raise AssertionError(f"middleware sent {message}")

    headers: Final = [(b"content-encoding", b"gzip"), (b"transfer-encoding", b"chunked")]
    await RequestBodyDecompressionMiddleware(app)(_http_scope(headers), _scripted_receive(client_messages), send)

    [(app_scope, app_messages)] = seen
    assert app_scope["headers"] == [(b"transfer-encoding", b"chunked")]
    assert app_messages == [
        {"type": "http.request", "body": REQUEST_JSON[:9], "more_body": True},
        {"type": "http.request", "body": REQUEST_JSON[9:], "more_body": False},
        {"type": "http.disconnect"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_messages",
    [
        pytest.param([{"type": "http.disconnect"}], id="before-the-body"),
        pytest.param(
            [{"type": "http.request", "body": GZIP_MAGIC, "more_body": True}, {"type": "http.disconnect"}],
            id="after-a-short-first-chunk",
        ),
        pytest.param(
            [
                {"type": "http.request", "body": gzip.compress(REQUEST_JSON, mtime=0)[:10], "more_body": True},
                {"type": "http.disconnect"},
            ],
            id="mid-body",
        ),
    ],
)
async def test_client_disconnect_ends_the_request_without_a_response(client_messages: list[Message]):
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
async def test_chunked_compressed_body_past_the_cap_gets_a_413_without_reading_the_rest():
    compressed: Final = gzip.compress(random.Random(0).randbytes(2 * MIB), mtime=0)
    chunk_size: Final = 64 * 1024
    chunks: Final = [compressed[start : start + chunk_size] for start in range(0, len(compressed), chunk_size)]
    client_messages: Final[list[Message]] = [
        *({"type": "http.request", "body": chunk, "more_body": True} for chunk in chunks[:-1]),
        {"type": "http.request", "body": chunks[-1], "more_body": False},
    ]
    sent: Final[list[Message]] = []
    app_calls: Final[list[Scope]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        app_calls.append(scope)

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestBodyDecompressionMiddleware(app, max_decompressed_body_size_mb=CAP_MB)(
        _http_scope([(b"content-encoding", b"gzip"), (b"transfer-encoding", b"chunked")]),
        _scripted_receive(client_messages),
        send,
    )

    assert app_calls == []
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"error": f"Request size is too large. Max size is {CAP_MB} MB"}
    assert client_messages, "the middleware read the whole body past the cap"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_messages", "body_the_app_gets"),
    [
        pytest.param(
            [
                {"type": "http.request", "more_body": True},
                {"type": "http.request", "body": gzip.compress(REQUEST_JSON, mtime=0), "more_body": False},
                {"type": "http.disconnect"},
            ],
            REQUEST_JSON,
            id="first-chunk-without-a-body",
        ),
        pytest.param([{"type": "http.request"}, {"type": "http.disconnect"}], b"", id="empty-request"),
    ],
)
async def test_request_messages_without_a_body_key_count_as_empty_chunks(
    client_messages: list[Message], body_the_app_gets: bytes
):
    seen: Final[list[tuple[Scope, Message]]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append((scope, await receive()))

    async def send(message: Message) -> None:
        raise AssertionError(f"middleware sent {message}")

    await RequestBodyDecompressionMiddleware(app)(
        _http_scope([(b"content-encoding", b"gzip")]), _scripted_receive(client_messages), send
    )

    [(app_scope, first_message)] = seen
    assert (b"content-encoding", b"gzip") not in app_scope["headers"]
    assert first_message == {"type": "http.request", "body": body_the_app_gets, "more_body": False}


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

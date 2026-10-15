import asyncio
import gzip
import json
import threading
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware import gzip_middleware
from litellm.proxy.middleware.gzip_middleware import (
    DASHBOARD_GZIP_PATHS,
    MINIMUM_SIZE_BYTES,
    GZipBufferedResponseMiddleware,
)

LARGE_PAYLOAD = {"rows": [{"date": f"2026-09-{day:02d}", "spend": day * 1.5} for day in range(1, 31)] * 20}
BASE64_EMBEDDINGS_RESPONSE_BYTES: Final = 824_000
HUGE_RESPONSE_BYTES: Final = 2 * 1024 * 1024
STREAM_CHUNKS = tuple(json.dumps({"part": part, "pad": "x" * MINIMUM_SIZE_BYTES}).encode() for part in range(3))


async def _large_json(request: Request) -> Response:
    return JSONResponse(LARGE_PAYLOAD)


async def _small_json(request: Request) -> Response:
    return JSONResponse({"ok": True})


async def _already_encoded(request: Request) -> Response:
    return Response(b"x" * (MINIMUM_SIZE_BYTES * 4), headers={"content-encoding": "br"})


async def _with_etag(request: Request) -> Response:
    return Response(b"y" * (MINIMUM_SIZE_BYTES * 4), headers={"etag": '"v1"'})


async def _partial(request: Request) -> Response:
    return Response(b"p" * (MINIMUM_SIZE_BYTES * 4), status_code=206, headers={"content-range": "bytes 0-1999/9000"})


async def _no_transform(request: Request) -> Response:
    return Response(b"n" * (MINIMUM_SIZE_BYTES * 4), headers={"cache-control": "public, no-transform"})


async def _huge(request: Request) -> Response:
    return Response(b"z" * HUGE_RESPONSE_BYTES, media_type="application/json")


async def _minimum_eligible(request: Request) -> Response:
    return Response(b"u" * MINIMUM_SIZE_BYTES, media_type="application/json")


async def _four_kib(request: Request) -> Response:
    return Response(b"a" * 4096, media_type="application/json")


async def _base64_embeddings_sized(request: Request) -> Response:
    return Response(b"e" * BASE64_EMBEDDINGS_RESPONSE_BYTES, media_type="application/json")


async def _json_stream(request: Request) -> Response:
    async def chunks():
        for chunk in STREAM_CHUNKS:
            yield chunk

    return StreamingResponse(chunks(), media_type="application/json")


APP = Starlette(
    routes=[
        Route("/large", _large_json),
        Route("/small", _small_json),
        Route("/encoded", _already_encoded),
        Route("/stream", _json_stream),
        Route("/etag", _with_etag),
        Route("/huge", _huge),
        Route("/minimum-eligible", _minimum_eligible),
        Route("/four-kib", _four_kib),
        Route("/base64-embeddings-sized", _base64_embeddings_sized),
        Route("/partial", _partial),
        Route("/no-transform", _no_transform),
    ]
)
APP.add_middleware(
    GZipBufferedResponseMiddleware, paths=frozenset(route.path for route in APP.routes if isinstance(route, Route))
)


async def _send_messages(
    path: str, accept_encoding: str | None, app: ASGIApp = APP, method: str = "GET", root_path: str = ""
) -> tuple[Message, ...]:
    headers = [(b"accept-encoding", accept_encoding.encode())] if accept_encoding is not None else []
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": root_path,
        "query_string": b"",
        "headers": headers,
    }
    sent: list[Message] = []  # mutable-ok: ASGI send callback collects messages in order
    requests: Final = iter(({"type": "http.request", "body": b"", "more_body": False},))
    never_disconnects: Final = asyncio.Event()

    async def receive() -> Message:
        request: Final = next(requests, None)
        if request is not None:
            return request
        await never_disconnects.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return tuple(sent)


def _headers(messages: tuple[Message, ...]) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in messages[0]["headers"]}


def _body(messages: tuple[Message, ...]) -> bytes:
    return b"".join(m.get("body", b"") for m in messages[1:])


@pytest.mark.parametrize("accept_encoding", ["gzip, deflate, br", "GZIP", "br;q=1, gzip;q=0.5", "x-gzip", "*"])
@pytest.mark.asyncio
async def test_large_buffered_json_is_gzipped_and_round_trips(accept_encoding):
    messages = await _send_messages("/large", accept_encoding)
    headers = _headers(messages)
    body = _body(messages)

    assert headers["content-encoding"] == "gzip"
    assert headers["vary"] == "Accept-Encoding"
    assert int(headers["content-length"]) == len(body)
    assert json.loads(gzip.decompress(body)) == LARGE_PAYLOAD
    assert len(body) < len(json.dumps(LARGE_PAYLOAD))


@pytest.mark.asyncio
async def test_multi_megabyte_body_round_trips():
    messages = await _send_messages("/huge", "gzip")

    assert _headers(messages)["content-encoding"] == "gzip"
    assert gzip.decompress(_body(messages)) == b"z" * HUGE_RESPONSE_BYTES


@pytest.mark.parametrize(
    "path",
    ["/minimum-eligible", "/four-kib", "/base64-embeddings-sized", "/huge"],
)
@pytest.mark.asyncio
async def test_all_eligible_bodies_compress_on_a_worker_thread(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    loop_thread: Final = threading.get_ident()
    real_compress: Final = gzip.compress

    def compress_naming_its_thread(data: bytes, compresslevel: int = 9) -> bytes:
        return real_compress(b"loop" if threading.get_ident() == loop_thread else b"worker", compresslevel)

    monkeypatch.setattr(gzip_middleware.gzip, "compress", compress_naming_its_thread)

    messages = await _send_messages(path, "gzip")

    assert _headers(messages)["content-encoding"] == "gzip"
    assert gzip.decompress(_body(messages)) == b"worker"


@pytest.mark.parametrize(
    ("path", "accept_encoding", "expected_vary"),
    [
        ("/large", None, "Accept-Encoding"),
        ("/large", "gzip;q=0", "Accept-Encoding"),
        ("/small", "gzip", None),
        ("/etag", "gzip", None),
        ("/stream", "gzip", None),
    ],
)
@pytest.mark.asyncio
async def test_vary_marks_every_negotiable_variant(path, accept_encoding, expected_vary):
    messages = await _send_messages(path, accept_encoding)

    assert _headers(messages).get("vary") == expected_vary


@pytest.mark.parametrize(
    ("path", "accept_encoding", "expected_encoding"),
    [
        ("/large", None, None),
        ("/large", "identity", None),
        ("/large", "gzip;q=0", None),
        ("/large", "br, gzip; q=0.0", None),
        ("/large", "*;q=0", None),
        ("/large", "*, gzip;q=0", None),
        ("/large", "gzip;q=invalid", None),
        ("/small", "gzip", None),
        ("/encoded", "gzip", "br"),
        ("/etag", "gzip", None),
        ("/stream", "gzip", None),
        ("/partial", "gzip", None),
        ("/no-transform", "gzip", None),
    ],
)
@pytest.mark.asyncio
async def test_response_passes_through_unmodified(path, accept_encoding, expected_encoding):
    with_header = await _send_messages(path, accept_encoding)
    without_header = await _send_messages(path, None)

    assert _headers(with_header).get("content-encoding") == expected_encoding
    assert _body(with_header) == _body(without_header)


@pytest.mark.asyncio
async def test_streamed_chunks_are_forwarded_one_by_one():
    messages = await _send_messages("/stream", "gzip")
    chunks = tuple(m["body"] for m in messages[1:] if m.get("body"))

    assert [m["type"] for m in messages].count("http.response.start") == 1
    assert chunks == STREAM_CHUNKS


@pytest.mark.asyncio
async def test_start_message_without_headers_key_is_still_gzipped():
    body: Final = b"h" * (MINIMUM_SIZE_BYTES * 4)

    async def headerless_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": body})

    messages = await _send_messages(
        "/user/daily/activity/aggregated", "gzip", GZipBufferedResponseMiddleware(headerless_app)
    )

    assert _headers(messages)["content-encoding"] == "gzip"
    assert gzip.decompress(_body(messages)) == body


@pytest.mark.asyncio
async def test_start_without_a_body_message_is_still_forwarded():
    async def start_only_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": [(b"x-done", b"1")]})

    messages = await _send_messages(
        "/user/daily/activity/aggregated", "gzip", GZipBufferedResponseMiddleware(start_only_app)
    )

    assert messages == ({"type": "http.response.start", "status": 204, "headers": [(b"x-done", b"1")]},)


@pytest.mark.parametrize("path", sorted(DASHBOARD_GZIP_PATHS))
@pytest.mark.parametrize("root_path", ["", "/gateway"])
@pytest.mark.asyncio
async def test_dashboard_reads_are_compressed_under_root_paths(path: str, root_path: str) -> None:
    async def dashboard(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(LARGE_PAYLOAD)(scope, receive, send)

    messages: Final = await _send_messages(
        root_path + path, "gzip", GZipBufferedResponseMiddleware(dashboard), root_path=root_path
    )

    assert _headers(messages)["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(_body(messages))) == LARGE_PAYLOAD


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/embeddings", "POST"),
        ("/embeddings", "POST"),
        ("/v1/chat/completions", "POST"),
        ("/v1/responses", "POST"),
        ("/v1/messages", "POST"),
        ("/openai/deployments/embedding-model/embeddings", "POST"),
        ("/custom-passthrough", "GET"),
        ("/openapi.json", "GET"),
        ("/health/liveliness", "GET"),
        ("/user/daily/activity/aggregated/unknown", "GET"),
        ("/prefix/user/daily/activity/aggregated", "GET"),
        ("/user/daily/activity/aggregated", "POST"),
        ("/user/daily/activity/aggregated", "HEAD"),
    ],
)
@pytest.mark.asyncio
async def test_non_dashboard_requests_pass_through_without_changing_headers(path: str, method: str) -> None:
    body: Final = b"e" * BASE64_EMBEDDINGS_RESPONSE_BYTES

    async def inference(scope: Scope, receive: Receive, send: Send) -> None:
        await Response(body, media_type="application/json", headers={"vary": "Origin"})(scope, receive, send)

    messages: Final = await _send_messages(path, "gzip", GZipBufferedResponseMiddleware(inference), method=method)

    assert _headers(messages) == {
        "content-type": "application/json",
        "content-length": str(len(body)),
        "vary": "Origin",
    }
    assert _body(messages) == body


def test_proxy_app_does_not_compress_openapi() -> None:
    from litellm.proxy.proxy_server import app

    client: Final = TestClient(app)
    gzip_accepted: Final = client.get("/openapi.json", headers={"accept-encoding": "gzip"})
    identity: Final = client.get("/openapi.json", headers={"accept-encoding": "identity"})

    assert gzip_accepted.status_code == identity.status_code == 200, (
        f"gzip={gzip_accepted.status_code} {gzip_accepted.text[:MINIMUM_SIZE_BYTES]}, "
        f"identity={identity.status_code} {identity.text[:MINIMUM_SIZE_BYTES]}"
    )
    assert "content-encoding" not in gzip_accepted.headers
    assert gzip_accepted.content == identity.content

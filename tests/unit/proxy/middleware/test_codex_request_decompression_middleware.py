from queue import SimpleQueue
from typing import Final

import pytest
import zstandard
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.codex_request_decompression_middleware import (
    CODEX_REQUEST_LIMIT_BYTES,
    CodexRequestDecompressionMiddleware,
)

_BODY: Final = b'{"model":"test-native","stream":true}'


async def _echo(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "body": (await request.body()).hex(),
            "encoding": request.headers.get("content-encoding"),
            "length": request.headers.get("content-length"),
        }
    )


def _client(root_path: str = "") -> TestClient:
    return TestClient(
        Starlette(
            routes=[
                Route("/backend-api/codex/responses", _echo, methods=["POST"]),
                Route("/v1/responses", _echo, methods=["POST"]),
            ],
            middleware=[Middleware(CodexRequestDecompressionMiddleware)],
        ),
        root_path=root_path,
    )


@pytest.mark.asyncio
async def test_empty_decoded_body_is_delivered_once_then_disconnect_is_preserved() -> None:
    incoming: Final = iter(
        (
            {"type": "http.request", "body": zstandard.ZstdCompressor().compress(b""), "more_body": False},
            {"type": "http.disconnect"},
        )
    )
    observed: Final[SimpleQueue[Message]] = SimpleQueue()

    async def receive() -> Message:
        return next(incoming)

    async def send(_: Message) -> None:
        return None

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        observed.put(await receive())
        observed.put(await receive())

    await CodexRequestDecompressionMiddleware(downstream)(
        {
            "type": "http",
            "method": "POST",
            "path": "/backend-api/codex/responses",
            "headers": [(b"content-encoding", b"zstd")],
        },
        receive,
        send,
    )
    assert observed.get_nowait() == {"type": "http.request", "body": b"", "more_body": False}
    assert observed.get_nowait() == {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_disconnect_during_compressed_upload_never_reaches_downstream() -> None:
    incoming: Final = iter(
        (
            {"type": "http.request", "body": b"incomplete-compressed-body", "more_body": True},
            {"type": "http.disconnect"},
        )
    )
    sent: Final[SimpleQueue[Message]] = SimpleQueue()

    async def receive() -> Message:
        return next(incoming)

    async def send(message: Message) -> None:
        sent.put(message)

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"forwarded": True})(scope, receive, send)

    await CodexRequestDecompressionMiddleware(downstream)(
        {
            "type": "http",
            "method": "POST",
            "path": "/backend-api/codex/responses",
            "headers": [(b"content-encoding", b"zstd")],
        },
        receive,
        send,
    )
    assert sent.empty()


@pytest.mark.parametrize("root_path", ["", "/gateway"])
def test_decodes_before_body_consumers_and_clears_stale_headers(root_path: str) -> None:
    encoded: Final = zstandard.ZstdCompressor().compress(_BODY)
    with _client(root_path) as client:
        response: Final = client.post(
            f"{root_path}/backend-api/codex/responses", content=encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 200
    assert response.json() == {"body": _BODY.hex(), "encoding": None, "length": None}


@pytest.mark.parametrize("encoded", [b"invalid", zstandard.ZstdCompressor().compress(_BODY)[:-1]])
def test_invalid_frames_never_reach_body_consumer(encoded: bytes) -> None:
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 400
    assert "error" in response.json()


def test_extra_frames_are_rejected() -> None:
    encoded: Final = zstandard.ZstdCompressor().compress(_BODY)
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=encoded + encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 400


@pytest.mark.parametrize("encoding", ["gzip", "zstd, gzip"])
def test_unsupported_encodings_are_rejected(encoding: str) -> None:
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=_BODY, headers={"content-encoding": encoding}
        )
    assert response.status_code == 415


def test_decoded_body_limit_is_enforced() -> None:
    encoded: Final = zstandard.ZstdCompressor().compress(b"x" * (CODEX_REQUEST_LIMIT_BYTES + 1))
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 413


def test_valid_medium_unknown_content_size_frame_is_not_mistaken_for_an_oversized_window() -> None:
    body: Final = b'{"input":"' + b"x" * (64 * 1024) + b'"}'
    parameters: Final = zstandard.ZstdCompressionParameters.from_level(1, window_log=20, write_content_size=0)
    compressor: Final = zstandard.ZstdCompressor(compression_params=parameters).compressobj()
    encoded: Final = compressor.compress(body) + compressor.flush()
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["body"] == body.hex()


def test_decoder_window_is_bounded_even_when_content_size_is_unknown() -> None:
    parameters: Final = zstandard.ZstdCompressionParameters.from_level(1, window_log=26, write_content_size=0)
    compressor: Final = zstandard.ZstdCompressor(compression_params=parameters).compressobj()
    encoded: Final = compressor.compress(_BODY) + compressor.flush()
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses", content=encoded, headers={"content-encoding": "zstd"}
        )
    assert response.status_code == 413


def test_compressed_input_limit_is_enforced() -> None:
    with _client() as client:
        response: Final = client.post(
            "/backend-api/codex/responses",
            content=b"x" * (CODEX_REQUEST_LIMIT_BYTES + 1),
            headers={"content-encoding": "zstd"},
        )
    assert response.status_code == 413


@pytest.mark.parametrize("path,encoding", [("/v1/responses", "zstd"), ("/backend-api/codex/responses", "identity")])
def test_other_routes_and_identity_requests_are_not_rewritten(path: str, encoding: str) -> None:
    with _client() as client:
        response: Final = client.post(path, content=_BODY, headers={"content-encoding": encoding})
    assert response.status_code == 200
    assert response.json() == {"body": _BODY.hex(), "encoding": encoding, "length": str(len(_BODY))}

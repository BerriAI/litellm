from io import BytesIO
from typing import Final

import zstandard
from starlette._utils import get_route_path
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CODEX_REQUEST_LIMIT_BYTES: Final = 32 * 1024 * 1024


def decode_codex_body(body: bytes) -> bytes | JSONResponse:
    try:
        frame: Final = zstandard.get_frame_parameters(body)
        if frame.window_size > CODEX_REQUEST_LIMIT_BYTES or (
            frame.content_size != zstandard.CONTENTSIZE_UNKNOWN and frame.content_size > CODEX_REQUEST_LIMIT_BYTES
        ):
            return JSONResponse({"error": "Codex request exceeds the decompression limit"}, status_code=413)
        decoded: Final = zstandard.ZstdDecompressor(max_window_size=CODEX_REQUEST_LIMIT_BYTES).decompress(
            body, max_output_size=CODEX_REQUEST_LIMIT_BYTES, allow_extra_data=False
        )
    except zstandard.ZstdError:
        return JSONResponse({"error": "Invalid or oversized zstd request body"}, status_code=400)
    return decoded


async def _read_compressed_body(request: Request) -> bytes | JSONResponse | None:
    buffer: Final = BytesIO()
    try:
        async for chunk in request.stream():
            if buffer.tell() + len(chunk) > CODEX_REQUEST_LIMIT_BYTES:
                return JSONResponse({"error": "Compressed Codex request is too large"}, status_code=413)
            buffer.write(chunk)
    except ClientDisconnect:
        return None
    return buffer.getvalue()


class CodexRequestDecompressionMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or get_route_path(scope) != "/backend-api/codex/responses"
        ):
            await self.app(scope, receive, send)
            return
        headers: Final = Headers(scope=scope)
        encoding: Final = headers.get("content-encoding", "identity").strip().lower()
        if encoding == "identity":
            await self.app(scope, receive, send)
            return
        if encoding != "zstd":
            await JSONResponse({"error": "Unsupported Codex request encoding"}, status_code=415)(scope, receive, send)
            return
        compressed: Final = await _read_compressed_body(Request(scope, receive))
        if compressed is None:
            return
        if isinstance(compressed, JSONResponse):
            await compressed(scope, receive, send)
            return
        decoded: Final = decode_codex_body(compressed)
        if isinstance(decoded, JSONResponse):
            await decoded(scope, receive, send)
            return

        body: Final = iter((decoded,))

        async def decoded_receive() -> Message:
            try:
                chunk: Final = next(body)
            except StopIteration:
                return await receive()
            return {"type": "http.request", "body": chunk, "more_body": False}

        decoded_scope: Final[Scope] = {
            **scope,
            "headers": [
                (name, value)
                for name, value in headers.raw
                if name.lower() not in {b"content-encoding", b"content-length"}
            ],
        }
        await self.app(decoded_scope, decoded_receive, send)

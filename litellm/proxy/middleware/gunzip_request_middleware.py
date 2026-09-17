"""
Gunzip Request Middleware - Pure ASGI implementation

Decompresses gzip-encoded request bodies (Content-Encoding: gzip) before
they reach FastAPI, so JSON body parsing works as usual.

Motivation: clients that send large payloads (e.g. long chat completion
prompts) may compress request bodies with gzip. Standard ASGI servers do
not decompress request bodies, so the proxy would otherwise fail with a
JSON decode error (400).

Enabled by default on the proxy app; requests without
`Content-Encoding: gzip` pass through untouched.
"""

import zlib

from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_DECOMPRESSED_SIZE: int = 512 * 1024 * 1024  # 512 MB


class GunzipRequestMiddleware:
    """
    Middleware to decompress gzip-encoded request bodies.

    On `Content-Encoding: gzip`:
      - buffers the full request body
      - decompresses it incrementally with a hard output-size limit
      - replaces the request body and rewrites `Content-Length`
      - strips `Content-Encoding` so downstream handlers see plain JSON

    Invalid gzip data is rejected with a 400 before reaching any route.
    Decompression that exceeds MAX_DECOMPRESSED_SIZE returns 413.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)
        if headers.get("content-encoding", "").strip().lower() != "gzip":
            await self.app(scope, receive, send)
            return

        compressed = await self._read_body(receive)
        if compressed is None:
            return

        if not compressed:
            await self._send_error(scope, receive, send, "Empty gzip-encoded request body", 400)
            return

        body = self._decompress(compressed)
        if body is None:
            await self._send_error(scope, receive, send, "Invalid gzip-encoded request body", 400)
            return
        if body is False:
            await self._send_error(scope, receive, send, "Decompressed request body exceeds size limit", 413)
            return

        if "content-encoding" in headers:
            del headers["content-encoding"]
        headers["content-length"] = str(len(body))

        body_sent = False

        async def receive_replaced() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, receive_replaced, send)

    @staticmethod
    async def _read_body(receive: Receive) -> bytes | None:
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return None
        return b"".join(chunks)

    @staticmethod
    def _decompress(compressed: bytes) -> bytes | None | bool:
        decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        decompressed_chunks: list[bytes] = []
        total_decompressed = 0
        offset = 0
        chunk_size = 65536
        try:
            while offset < len(compressed):
                end = min(offset + chunk_size, len(compressed))
                decompressed = decompressor.decompress(compressed[offset:end])
                total_decompressed += len(decompressed)
                if total_decompressed > MAX_DECOMPRESSED_SIZE:
                    return False
                decompressed_chunks.append(decompressed)
                offset = end
            tail = decompressor.flush()
            total_decompressed += len(tail)
            if total_decompressed > MAX_DECOMPRESSED_SIZE:
                return False
            decompressed_chunks.append(tail)
            if not decompressor.eof:
                return None
        except (OSError, EOFError, zlib.error):
            return None
        return b"".join(decompressed_chunks)

    @staticmethod
    async def _send_error(scope: Scope, receive: Receive, send: Send, message: str, status: int) -> None:
        response = PlainTextResponse(message, status_code=status)
        await response(scope, receive, send)

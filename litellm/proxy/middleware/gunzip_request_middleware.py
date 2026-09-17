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

import gzip

from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class GunzipRequestMiddleware:
    """
    Middleware to decompress gzip-encoded request bodies.

    On `Content-Encoding: gzip`:
      - buffers the full request body
      - decompresses it with `gzip.decompress`
      - replaces the request body and rewrites `Content-Length`
      - strips `Content-Encoding` so downstream handlers see plain JSON

    Invalid gzip data is rejected with a 400 before reaching any route.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Fast path: only inspect HTTP requests; pass through
        # websocket/lifespan immediately
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)
        if headers.get("content-encoding", "").strip().lower() != "gzip":
            await self.app(scope, receive, send)
            return

        # Buffer the full request body from the ASGI receive channel
        chunks: list = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        compressed = b"".join(chunks)
        if not compressed:
            # An empty body with `Content-Encoding: gzip` is malformed
            # (gzip.decompress(b"") silently returns b""), reject it early
            response = PlainTextResponse("Empty gzip-encoded request body", status_code=400)
            await response(scope, receive, send)
            return

        try:
            body = gzip.decompress(compressed)
        except OSError:
            response = PlainTextResponse("Invalid gzip-encoded request body", status_code=400)
            await response(scope, receive, send)
            return

        # Rewrite headers: strip content-encoding, fix content-length
        if "content-encoding" in headers:
            del headers["content-encoding"]
        headers["content-length"] = str(len(body))

        sent = False

        async def receive_replaced() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, receive_replaced, send)

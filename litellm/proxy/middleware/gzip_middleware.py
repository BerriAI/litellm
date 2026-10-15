import gzip
import os
from collections.abc import Callable
from types import MappingProxyType
from typing import Final, TypeAlias

import anyio
import anyio.lowlevel
import anyio.to_thread
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MINIMUM_SIZE_BYTES: Final = 500
OFF_LOOP_SIZE_BYTES: Final = 32 * 1024
COMPRESS_LEVEL: Final = 6
MAX_CONCURRENT_OFF_LOOP_COMPRESSIONS: Final = (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
)
_OFF_LOOP_COMPRESS_LIMITER: Final = anyio.lowlevel.RunVar[anyio.CapacityLimiter]("litellm_gzip_compress_limiter")
Compressor: TypeAlias = Callable[[bytes, int], bytes]


def _coding_weight(part: str) -> tuple[str, float]:
    coding, _, params = part.partition(";")
    qvalue: Final = next((p.strip()[2:] for p in params.split(";") if p.strip().lower().startswith("q=")), "1")
    try:
        return coding.strip().lower(), float(qvalue)
    except ValueError:
        return coding.strip().lower(), 0.0


def accepts_gzip(accept_encoding: str) -> bool:
    weights: Final = MappingProxyType(dict(_coding_weight(part) for part in accept_encoding.split(",") if part.strip()))
    return weights.get("gzip", weights.get("x-gzip", weights.get("*", 0.0))) > 0


def _off_loop_compress_limiter_for_this_loop() -> anyio.CapacityLimiter:
    existing: Final = _OFF_LOOP_COMPRESS_LIMITER.get(None)
    if existing is not None:
        return existing
    created: Final = anyio.CapacityLimiter(MAX_CONCURRENT_OFF_LOOP_COMPRESSIONS)
    _OFF_LOOP_COMPRESS_LIMITER.set(created)
    return created


async def _compress(body: bytes, compress: Compressor) -> bytes:
    if len(body) < OFF_LOOP_SIZE_BYTES:
        return compress(body, COMPRESS_LEVEL)
    return await anyio.to_thread.run_sync(
        compress, body, COMPRESS_LEVEL, limiter=_off_loop_compress_limiter_for_this_loop()
    )


class _BufferedBodyGzipResponder:
    """Holds the response start until the first body message shows the body is complete, so streams are never delayed."""

    def __init__(self, send: Send, gzip_accepted: bool, compress: Compressor) -> None:
        self.send = send
        self.gzip_accepted = gzip_accepted
        self.compress = compress
        self.held_start: Message | None = None
        self.decided = False

    async def __call__(self, message: Message) -> None:
        if self.decided:
            await self.send(message)
            return
        if message["type"] == "http.response.start":
            self.held_start = message
            return
        self.decided = True
        start: Final = self.held_start
        if start is None:
            await self.send(message)
            return
        body: Final[bytes] = message.get("body", b"")
        start.setdefault("headers", ())
        headers: Final = MutableHeaders(scope=start)
        negotiable: Final = (
            message["type"] == "http.response.body"
            and not message.get("more_body", False)
            and len(body) >= MINIMUM_SIZE_BYTES
            and "content-encoding" not in headers
            and "etag" not in headers
            and start["status"] != 206
            and "no-transform" not in headers.get("cache-control", "").lower()
        )
        if negotiable:
            headers.add_vary_header("Accept-Encoding")
        if not (negotiable and self.gzip_accepted):
            await self.send(start)
            await self.send(message)
            return
        compressed: Final = await _compress(body, self.compress)
        headers["content-encoding"] = "gzip"
        headers["content-length"] = str(len(compressed))
        await self.send(start)
        await self.send({**message, "body": compressed})

    async def release_held_start(self) -> None:
        if not self.decided and self.held_start is not None:
            self.decided = True
            await self.send(self.held_start)


class GZipBufferedResponseMiddleware:
    def __init__(self, app: ASGIApp, compress: Compressor = gzip.compress) -> None:
        self.app = app
        self.compress = compress

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        gzip_accepted: Final = accepts_gzip(Headers(scope=scope).get("accept-encoding", ""))
        responder: Final = _BufferedBodyGzipResponder(send, gzip_accepted, self.compress)
        await self.app(scope, receive, responder)
        await responder.release_held_start()

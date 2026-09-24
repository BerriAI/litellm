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

The decompressed size limit aligns with the proxy's `max_request_size_mb`
setting (premium-gated, same as RequestSizeLimitMiddleware). When not
configured or non-premium, the `LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB`
env var (in MB, default 100) is used.

Memory: decompression processes ASGI body chunks incrementally via
zlib.decompressobj. The decompressed output accumulates in a single
bytearray up to the configured size limit. Peak memory per request is
approximately 2x the decompressed size (bytearray + bytes copy).
N concurrent gzip requests use ~N x max_size memory.
"""

from __future__ import annotations

import asyncio
import enum
import math
import os
import zlib
from collections.abc import Callable

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
)

_DEFAULT_MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024  # 100 MB - covers multimodal (images/audio) + large context
_MAX_GZIP_MEMBERS = 100  # allow up to 100 additional concatenated members after the first


class DecompressResult(enum.Enum):
    """Result of stream decompression."""

    OK = enum.auto()
    DISCONNECT = enum.auto()
    SIZE_EXCEEDED = enum.auto()
    EMPTY_BODY = enum.auto()
    INVALID = enum.auto()


def _resolve_env_max_size() -> int | None:
    """Read and validate LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB env var.

    Invalid values log error and fall back to default.
    """
    env_val = os.environ.get("LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB")
    if not env_val:
        return _DEFAULT_MAX_DECOMPRESSED_SIZE
    try:
        size_mb = float(env_val)
        if not math.isfinite(size_mb):
            raise ValueError(f"not a finite number: {env_val}")
        if size_mb <= 0:
            verbose_proxy_logger.warning(
                "LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB=%s, decompression size limit disabled, "
                "requests may be vulnerable to gzip bomb attacks",
                env_val,
            )
            return None  # <= 0 = unlimited
        return int(size_mb * 1024 * 1024)
    except (ValueError, OverflowError) as e:
        verbose_proxy_logger.error(
            "Invalid LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB=%s, using default %dMB: %s",
            env_val,
            _DEFAULT_MAX_DECOMPRESSED_SIZE // (1024 * 1024),
            e,
        )
        return _DEFAULT_MAX_DECOMPRESSED_SIZE


def _is_pass_through_route(path: str) -> bool:
    """Check if path is a pass-through route (static or dynamically registered)."""
    return InitPassThroughEndpointHelpers.is_registered_pass_through_route(route=path)


class _DecompressOutcome:
    """Result of decompression: status + optional body."""

    __slots__ = ("body", "status")

    def __init__(self, status: DecompressResult, body: bytes = b"") -> None:
        self.status = status
        self.body = body


class GunzipRequestMiddleware:
    """
    Middleware to decompress gzip-encoded request bodies.

    On `Content-Encoding: gzip`:
      - streams compressed chunks from ASGI receive into a decompressor
      - immediately switches to a new decompressor when the current one
        reaches EOF (concatenated gzip members), avoiding unused_data
        accumulation
      - enforces a hard output-size limit during streaming
      - replaces the request body and rewrites `Content-Length`
      - strips `Content-Encoding` so downstream handlers see plain JSON

    Invalid gzip data is rejected with a 400 before reaching any route.
    Decompression that exceeds the size limit returns 413.

    Middleware ordering: registered before CORSMiddleware so error responses
    include CORS headers. Auth (user_api_key_auth) runs as a FastAPI dependency
    in the route layer, after this middleware has decompressed the body.
    """

    def __init__(
        self,
        app: ASGIApp,
        get_max_size: Callable[[], float | int | None] | None = None,
        is_premium: Callable[[], bool] | None = None,
    ) -> None:
        self.app = app
        self.get_max_size = get_max_size
        self.is_premium = is_premium
        self._env_max_size: int | None = None
        # Lazy resolve on first gzip request so YAML environment_variables
        # (loaded by proxy_startup_event after middleware stack construction)
        # are available
        self._env_max_size_resolved = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)
        if headers.get("content-encoding", "").strip().lower() != "gzip":
            await self.app(scope, receive, send)
            return

        if _is_pass_through_route(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        max_size = self._resolve_max_size()
        outcome = await self._stream_decompress(receive, max_size)

        if outcome.status == DecompressResult.OK:
            if "content-encoding" in headers:
                del headers["content-encoding"]
            headers["content-length"] = str(len(outcome.body))
            await self._send_decompressed_body(scope, receive, send, outcome.body)
            return

        if outcome.status == DecompressResult.DISCONNECT:
            return

        if outcome.status == DecompressResult.INVALID:
            verbose_proxy_logger.info("Invalid gzip-encoded request body")
            await self._send_error(scope, receive, send, "Invalid gzip-encoded request body", 400)
            return

        if outcome.status == DecompressResult.SIZE_EXCEEDED:
            verbose_proxy_logger.warning("Decompressed request body exceeds size limit (%s bytes)", max_size)
            await self._send_error(scope, receive, send, "Decompressed request body exceeds size limit", 413)
            return

    async def _send_decompressed_body(self, scope: Scope, receive: Receive, send: Send, body: bytes) -> None:
        body_sent = False

        async def receive_replaced() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}  # mutable-ok: ASGI message dict
            return await receive()

        await self.app(scope, receive_replaced, send)

    def _resolve_max_size(self) -> int | None:
        if self.is_premium is not None and self.is_premium():
            if self.get_max_size is not None:
                size_mb = self.get_max_size()
                if size_mb is not None:
                    if size_mb <= 0:
                        return None  # <= 0 = unlimited, aligned with _mb_to_bytes
                    return int(size_mb * 1024 * 1024)
        if not self._env_max_size_resolved:
            self._env_max_size = _resolve_env_max_size()
            self._env_max_size_resolved = True
        return self._env_max_size

    @staticmethod
    async def _stream_decompress(receive: Receive, max_size: int | None) -> _DecompressOutcome:
        """
        Stream-decompress the request body using a state machine that
        immediately switches decompressors on EOF, avoiding unused_data
        accumulation.

        Each compressed byte is processed exactly once: when a decompressor
        reaches EOF, its unused_data is fed to a new decompressor in the
        same iteration, not deferred to a post-processing pass.
        """
        result = bytearray()  # mutable-ok: single growable buffer for decompressed output
        decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        got_any_data = False
        member_count = 0
        need_final_check = True

        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return _DecompressOutcome(DecompressResult.DISCONNECT)
                if message["type"] != "http.request":
                    continue

                chunk = message.get("body", b"")
                if chunk:
                    got_any_data = True
                    need_final_check = True

                # Feed chunk (or unused_data from previous member) into decompressor.
                # Loop handles concatenated gzip members: when the current
                # decompressor reaches EOF, its unused_data is fed to a
                # new decompressor immediately, preventing accumulation.
                data = chunk
                while data:
                    if max_size is not None:
                        remaining = max_size - len(result)
                        if (
                            remaining <= 0
                        ):  # pragma: no cover - reached in edge cases when decompress output exactly hits limit
                            return _DecompressOutcome(DecompressResult.SIZE_EXCEEDED)
                        out = decompressor.decompress(data, remaining)
                    else:
                        out = decompressor.decompress(data)
                    result.extend(out)

                    # Drain unconsumed_tail (data that decompressor couldn't
                    # output in one call due to max_length)
                    while (
                        decompressor.unconsumed_tail
                    ):  # pragma: no cover - only triggers when decompress output exceeds max_length in one call
                        if max_size is not None:
                            remaining = max_size - len(result)
                            if remaining <= 0:
                                return _DecompressOutcome(DecompressResult.SIZE_EXCEEDED)
                            out = decompressor.decompress(decompressor.unconsumed_tail, remaining)
                        else:
                            out = decompressor.decompress(decompressor.unconsumed_tail)
                        result.extend(out)

                    if decompressor.eof:
                        result.extend(decompressor.flush())
                        need_final_check = False
                        data = decompressor.unused_data
                        if data:
                            member_count += 1
                            if member_count > _MAX_GZIP_MEMBERS:
                                return _DecompressOutcome(DecompressResult.INVALID)
                            decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
                            need_final_check = True
                        else:
                            # No unused_data; next ASGI chunk needs a fresh decompressor
                            decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
                            break
                    else:
                        break

                    await asyncio.sleep(0)

                if not message.get("more_body", False):
                    break

            if not got_any_data:
                return _DecompressOutcome(DecompressResult.OK, b"")

            # Final flush for the last decompressor (only if it was fed data
            # and hasn't reached EOF yet — indicates possible truncation)
            if need_final_check:
                result.extend(decompressor.flush())
                if not decompressor.eof:
                    return _DecompressOutcome(DecompressResult.INVALID)

            if not result:
                return _DecompressOutcome(DecompressResult.OK, b"")

        except (OSError, EOFError, zlib.error):
            return _DecompressOutcome(DecompressResult.INVALID)

        return _DecompressOutcome(DecompressResult.OK, bytes(result))

    @staticmethod
    async def _send_error(scope: Scope, receive: Receive, send: Send, message: str, status: int) -> None:
        response = JSONResponse({"error": message}, status_code=status)  # mutable-ok: error response dict
        await response(scope, receive, send)

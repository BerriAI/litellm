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

Size limits (defense in depth / gzip bomb protection):

1. `max_request_size_mb` (premium-gated, same as RequestSizeLimitMiddleware
   / check_and_validate_max_request_size) - applied to BOTH the compressed
   body (via Content-Length, checked BEFORE any decompression work) and the
   decompressed body (enforced during streaming).
2. `LITELLM_MAX_DECOMPRESSED_REQUEST_SIZE_MB` env var (in MB, default 20) -
   fallback for non-premium deployments or when max_request_size_mb is not
   configured. `<= 0` disables the check (NOT recommended: unbounded
   memory per request, gzip bomb exposure).
3. Hard `_DEFAULT_MAX_DECOMPRESSED_SIZE` of 20 MB if the env var is unset.

The Content-Length pre-check means a request whose wire size already
exceeds the limit is rejected with 413 before the body is read or any
decompression happens. This keeps worst-case CPU/memory work per request
bounded for unauthenticated requests (this middleware runs BEFORE auth,
which is a FastAPI dependency in the route layer).

Memory: decompression processes ASGI body chunks incrementally via
zlib.decompressobj. The decompressed output accumulates in a single
bytearray up to the configured size limit. Peak memory per request is
approximately 2x the decompressed size (bytearray + bytes copy), plus a
transient extra copy when FastAPI parses the body. N concurrent gzip
requests use ~N x max_size memory.

Behavior notes:
- Only exactly `Content-Encoding: gzip` (case/whitespace tolerant) is
  handled; `deflate`, `br`, `identity`, multi-value encodings like
  "gzip, deflate" pass through untouched.
- An empty body with `Content-Encoding: gzip` is passed to the app as an
  empty body (Content-Length: 0, Content-Encoding stripped) instead of
  failing.
- Trailing non-gzip garbage after a complete gzip member returns 400
  (stricter than `gzip -d`, which silently stops at the member end).
- Invalid/truncated gzip -> 400, oversized -> 413, client disconnect
  mid-body -> clean exit without a response.
- Error responses use JSONResponse format (consistent with
  RequestSizeLimitMiddleware).
- Pass-through routes (static + dynamically registered, root_path aware)
  are skipped so signed pass-through bodies are never modified.

Middleware ordering: register this BEFORE CORSMiddleware so error
responses pass back through CORS and include CORS headers.
RequestSizeLimitMiddleware is registered after this one, so it wraps it
and enforces the limit against the compressed wire size (Content-Length,
or chunk counting when it is absent) before any decompression happens,
while this middleware enforces the same limit against the decompressed
output. Both bounds hold independently, so no double headroom is needed
with this stack; with a different ordering the Content-Length pre-check
above keeps either stack safe.
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

# 20 MB - chat completions with huge contexts stay far below this; the env
# var is the supported knob for deployments that need more (multimodal etc.)
_DEFAULT_MAX_DECOMPRESSED_SIZE = 20 * 1024 * 1024  # 20 MB
_MAX_GZIP_MEMBERS = 100  # allow up to 100 additional concatenated members after the first


class DecompressResult(enum.Enum):
    """Result of stream decompression."""

    OK = enum.auto()
    DISCONNECT = enum.auto()
    SIZE_EXCEEDED = enum.auto()
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
        return max(1, int(size_mb * 1024 * 1024))
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
      - rejects the request with 413 before reading the body if the
        compressed Content-Length already exceeds the size limit
      - streams compressed chunks from ASGI receive into a decompressor
      - immediately switches to a new decompressor when the current one
        reaches EOF (concatenated gzip members), avoiding unused_data
        accumulation
      - enforces a hard output-size limit during streaming
      - replaces the request body and rewrites `Content-Length`
      - strips `Content-Encoding` so downstream handlers see plain JSON

    Invalid gzip data is rejected with a 400 before reaching any route.
    Decompression that exceeds the size limit returns 413.
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

        path = scope.get("path", "")
        if _is_pass_through_route(path):
            await self.app(scope, receive, send)
            return

        max_size = self._resolve_max_size()

        # Pre-check the wire size BEFORE any body read/decompression: a gzip
        # stream is never meaningfully smaller than its compressed form, so a
        # request already over the limit here can never become acceptable.
        # This bounds per-request work for unauthenticated requests.
        content_length = headers.get("content-length")
        if max_size is not None and content_length:
            try:
                if int(content_length) > max_size:
                    verbose_proxy_logger.warning(
                        "Rejected gzip request to %s: compressed body size %s bytes exceeds size limit %s bytes "
                        "(checked before decompression)",
                        path,
                        content_length,
                        max_size,
                    )
                    await self._send_error(
                        scope, receive, send, "Decompressed request body exceeds size limit", 413
                    )
                    return
            except ValueError:
                pass

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
            verbose_proxy_logger.debug("Invalid gzip-encoded request body on path %s", path)
            await self._send_error(scope, receive, send, "Invalid gzip-encoded request body", 400)
            return

        if outcome.status == DecompressResult.SIZE_EXCEEDED:
            verbose_proxy_logger.warning(
                "Rejected gzip request to %s: decompressed body exceeds size limit (%s bytes)", path, max_size
            )
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
                    return max(1, int(size_mb * 1024 * 1024))
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
                        if remaining <= 0:
                            return _DecompressOutcome(DecompressResult.SIZE_EXCEEDED)
                        out = decompressor.decompress(data, remaining)
                    else:
                        out = decompressor.decompress(data)
                    result.extend(out)

                    # Drain unconsumed_tail (data that decompressor couldn't
                    # output in one call due to max_length); yield so a single
                    # huge chunk cannot starve other requests on the loop.
                    while decompressor.unconsumed_tail:
                        if max_size is not None:
                            remaining = max_size - len(result)
                            if remaining <= 0:
                                return _DecompressOutcome(DecompressResult.SIZE_EXCEEDED)
                            out = decompressor.decompress(decompressor.unconsumed_tail, remaining)
                        else:
                            out = decompressor.decompress(decompressor.unconsumed_tail)
                        result.extend(out)
                        await asyncio.sleep(0)

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
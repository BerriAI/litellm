import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Final

import anyio

from litellm.constants import STREAM_SSE_KEEPALIVE_PING_CHUNK

ANTHROPIC_PING_SSE_CHUNK: Final = STREAM_SSE_KEEPALIVE_PING_CHUNK
SSE_COMMENT_PING: Final = ": ping\n\n"
SSE_COMMENT_PING_BYTES: Final = SSE_COMMENT_PING.encode()
# The byte form of proxy_server._SSE_FRAME_DELIMITERS, CR-only included: SSE
# terminates a line with CRLF, LF or CR, so a blank line is any of these three.
_SSE_FRAME_DELIMITERS: Final = (b"\r\n\r\n", b"\n\n", b"\r\r")
_SSE_DELIMITER_LOOKBACK: Final = max(len(delimiter) for delimiter in _SSE_FRAME_DELIMITERS)
_STREAM_START_TAIL: Final = b"\n\n"
_SSE_MEDIA_TYPE: Final = "text/event-stream"


def coerce_keepalive_interval(ping_interval_seconds: float | str | None) -> float | None:
    if ping_interval_seconds is None:
        return None
    try:
        interval: Final = float(ping_interval_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(interval) or interval <= 0:
        return None
    return interval


def keepalive_ping_has_fired(elapsed_seconds: float, ping_interval_seconds: float | str | None) -> bool:
    """Whether a keepalive ping has already gone out, which flushes the response headers.

    A caller that discovers a failure after that point cannot raise its way to the client, since
    the status line is already on the wire. With pings disabled nothing flushes early, so a raise
    still carries its real status.
    """
    interval: Final = coerce_keepalive_interval(ping_interval_seconds)
    return interval is not None and elapsed_seconds >= interval


def wrap_sse_stream_with_keepalive_pings(
    stream: AsyncGenerator[str, None],
    ping_interval_seconds: float | str | None,
    ping_chunk: str = ANTHROPIC_PING_SSE_CHUNK,
) -> AsyncGenerator[str, None]:
    """Fill idle gaps in an SSE stream, including the one before its first chunk.

    ``ping_chunk`` is what gets written into those gaps. It defaults to Anthropic's
    own ``ping`` event because that is the protocol the first caller speaks; a
    stream carrying anything else wants ``SSE_COMMENT_PING``, which is a comment
    every conformant SSE client discards rather than a frame it has to understand.
    """
    interval: Final = coerce_keepalive_interval(ping_interval_seconds)
    if interval is None:
        return stream
    return _keepalive_ping_stream(stream=stream, ping_interval_seconds=interval, ping_chunk=ping_chunk)


async def _keepalive_ping_stream(
    stream: AsyncGenerator[str, None],
    ping_interval_seconds: float,
    ping_chunk: str,
) -> AsyncGenerator[str, None]:
    pending = asyncio.ensure_future(
        stream.__anext__()
    )  # rebind-ok: re-armed with the next __anext__ after each delivered chunk
    try:
        while True:
            await asyncio.wait({pending}, timeout=ping_interval_seconds)
            if not pending.done():
                yield ping_chunk
                continue
            try:
                yield pending.result()
            except StopAsyncIteration:
                return
            pending = asyncio.ensure_future(stream.__anext__())
    finally:
        pending.cancel()
        with anyio.CancelScope(shield=True):
            with contextlib.suppress(BaseException):
                await pending
            await stream.aclose()


def is_sse_content_type(content_type: str | None) -> bool:
    return content_type is not None and content_type.split(";", 1)[0].strip().lower() == _SSE_MEDIA_TYPE


def split_complete_sse_frames(pending: bytes) -> tuple[bytes, bytes]:
    """Split buffered SSE bytes into ``(complete_frames, unterminated_tail)``."""
    boundary_end: Final = max(
        (pending.rfind(delimiter) + len(delimiter) for delimiter in _SSE_FRAME_DELIMITERS if delimiter in pending),
        default=0,
    )
    if boundary_end == 0:
        return b"", pending
    return pending[:boundary_end], pending[boundary_end:]


def wrap_passthrough_sse_bytes_with_keepalive_pings(
    stream: AsyncGenerator[bytes, None],
    ping_interval_seconds: float | str | None,
    upstream_headers: Mapping[str, str],
) -> AsyncGenerator[bytes, None]:
    """Fill upstream silence on a byte-relaying passthrough stream with SSE comments.

    Passthrough routes relay upstream bytes verbatim, so a model that thinks for
    longer than an intermediary's idle read timeout has its connection dropped
    before the first token. Only streams the upstream itself declares as
    ``text/event-stream`` are wrapped: a comment spliced into a binary transport
    (AWS event streams on ``/bedrock``, protobuf, NDJSON) would corrupt it.
    """
    interval: Final = coerce_keepalive_interval(ping_interval_seconds)
    if interval is None or not is_sse_content_type(upstream_headers.get("content-type")):
        return stream
    return _keepalive_ping_byte_stream(stream=stream, ping_interval_seconds=interval)


async def _keepalive_ping_byte_stream(
    stream: AsyncGenerator[bytes, None],
    ping_interval_seconds: float,
) -> AsyncGenerator[bytes, None]:
    pending = asyncio.ensure_future(
        stream.__anext__()
    )  # rebind-ok: re-armed with the next __anext__ after each delivered chunk
    # The tail of the bytes relayed so far, long enough to hold any delimiter.
    # Seeded as a delimiter because a stream starts at a frame boundary, and kept
    # across chunks because a delimiter can be split between two transport reads,
    # which testing only the latest chunk would miss for the rest of the stream.
    recent_tail = _STREAM_START_TAIL  # rebind-ok: rolling window over the relayed bytes
    try:
        while True:
            await asyncio.wait((pending,), timeout=ping_interval_seconds)
            if not pending.done():
                # The relayed chunks are raw transport reads, not whole SSE
                # frames, so an upstream that stalls halfway through a frame
                # must not have a comment spliced into it.
                if recent_tail.endswith(_SSE_FRAME_DELIMITERS):
                    yield SSE_COMMENT_PING_BYTES
                continue
            try:
                chunk: bytes = pending.result()
            except StopAsyncIteration:
                return
            if chunk:
                recent_tail = (recent_tail + chunk)[-_SSE_DELIMITER_LOOKBACK:]
            yield chunk
            pending = asyncio.ensure_future(stream.__anext__())
    finally:
        pending.cancel()
        with anyio.CancelScope(shield=True):
            with contextlib.suppress(BaseException):
                await pending
            await stream.aclose()


def resolve_ttft_keepalive_interval(
    deployments: Iterable[Mapping[str, object]],
    global_interval: float | str | None,
) -> float | None:
    """The keepalive interval to use before the upstream has answered at all.

    No deployment has served the request yet, so a per-deployment
    ``keepalive_seconds`` is only trusted when every candidate under the requested
    model carries the same one, which is how the mid-stream engine treats its own
    model_name fallback. Otherwise the operator's global default applies.

    An explicit ``0`` survives as a disable, since coercion rejects it: that keeps
    an operator's documented hard disable working on this path too, rather than
    letting the global switch a deployment back on behind their back.

    A client-supplied value is deliberately not consulted. Opening the response
    early is an operator decision, and a request must not be able to enable it for
    a deployment that never did.
    """
    configured: Final = frozenset(_keepalive_param(deployment) for deployment in deployments)
    agreed: Final = next(iter(configured)) if len(configured) == 1 else None
    return coerce_keepalive_interval(global_interval if agreed is None else agreed)


def _keepalive_param(deployment: Mapping[str, object]) -> float | str | None:
    params: Final = deployment.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    value: Final = params.get("keepalive_seconds")
    return value if isinstance(value, (int, float, str)) else None

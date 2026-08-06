import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator
from typing import Final

import anyio

ANTHROPIC_PING_SSE_CHUNK: Final = 'event: ping\ndata: {"type": "ping"}\n\n'


def _coerce_interval(ping_interval_seconds: float | str | None) -> float | None:
    if ping_interval_seconds is None:
        return None
    try:
        interval: Final = float(ping_interval_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(interval) or interval <= 0:
        return None
    return interval


def wrap_sse_stream_with_keepalive_pings(
    stream: AsyncGenerator[str, None],
    ping_interval_seconds: float | str | None,
) -> AsyncGenerator[str, None]:
    interval: Final = _coerce_interval(ping_interval_seconds)
    if interval is None:
        return stream
    return _keepalive_ping_stream(stream=stream, ping_interval_seconds=interval)


async def _keepalive_ping_stream(
    stream: AsyncGenerator[str, None],
    ping_interval_seconds: float,
) -> AsyncGenerator[str, None]:
    pending = asyncio.ensure_future(
        stream.__anext__()
    )  # rebind-ok: re-armed with the next __anext__ after each delivered chunk
    try:
        while True:
            await asyncio.wait({pending}, timeout=ping_interval_seconds)
            if not pending.done():
                yield ANTHROPIC_PING_SSE_CHUNK
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

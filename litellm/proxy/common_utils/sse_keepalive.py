import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Final

import anyio

ANTHROPIC_PING_SSE_CHUNK: Final = 'event: ping\ndata: {"type": "ping"}\n\n'


def wrap_sse_stream_with_keepalive_pings(
    stream: AsyncGenerator[str, None],
    ping_interval_seconds: float,
) -> AsyncGenerator[str, None]:
    if ping_interval_seconds <= 0:
        return stream
    return _keepalive_ping_stream(stream=stream, ping_interval_seconds=ping_interval_seconds)


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

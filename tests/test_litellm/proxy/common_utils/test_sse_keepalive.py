import asyncio
from collections.abc import AsyncGenerator
from typing import Final, cast

import pytest
from fastapi.responses import StreamingResponse

from litellm.proxy.common_request_processing import create_response
from litellm.proxy.common_utils.sse_keepalive import (
    ANTHROPIC_PING_SSE_CHUNK,
    wrap_sse_stream_with_keepalive_pings,
)

MESSAGE_START_CHUNK: Final = 'data: {"type": "message_start"}\n\n'
TEXT_DELTA_CHUNK: Final = 'data: {"type": "content_block_delta"}\n\n'


@pytest.mark.asyncio
async def test_pings_fill_mid_stream_silence_and_preserve_chunk_order():
    async def gappy_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        await asyncio.sleep(0.3)
        yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=gappy_stream(), ping_interval_seconds=0.05)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == MESSAGE_START_CHUNK
    assert collected[-1] == TEXT_DELTA_CHUNK
    assert ANTHROPIC_PING_SSE_CHUNK in collected[1:-1]
    assert [chunk for chunk in collected if chunk != ANTHROPIC_PING_SSE_CHUNK] == [
        MESSAGE_START_CHUNK,
        TEXT_DELTA_CHUNK,
    ]


@pytest.mark.asyncio
async def test_ping_emitted_while_waiting_for_first_chunk():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds=0.05)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK


@pytest.mark.asyncio
async def test_no_pings_when_chunks_arrive_faster_than_interval():
    async def fast_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        yield TEXT_DELTA_CHUNK
        yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=fast_stream(), ping_interval_seconds=1.0)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected == [MESSAGE_START_CHUNK, TEXT_DELTA_CHUNK, TEXT_DELTA_CHUNK]


@pytest.mark.asyncio
async def test_upstream_exception_propagates():
    async def failing_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        raise ValueError("upstream broke")

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=failing_stream(), ping_interval_seconds=5.0)

    assert await wrapped.__anext__() == MESSAGE_START_CHUNK
    with pytest.raises(ValueError, match="upstream broke"):
        await wrapped.__anext__()


@pytest.mark.asyncio
async def test_aclose_mid_silence_cancels_upstream_and_runs_its_cleanup():
    upstream_cleaned_up: Final = asyncio.Event()

    async def hung_stream() -> AsyncGenerator[str, None]:
        try:
            yield MESSAGE_START_CHUNK
            await asyncio.Event().wait()
            yield TEXT_DELTA_CHUNK
        finally:
            upstream_cleaned_up.set()

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=hung_stream(), ping_interval_seconds=0.05)

    assert await wrapped.__anext__() == MESSAGE_START_CHUNK
    assert await wrapped.__anext__() == ANTHROPIC_PING_SSE_CHUNK
    await wrapped.aclose()

    assert upstream_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_non_positive_interval_returns_stream_unwrapped():
    async def any_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK

    stream: Final = any_stream()
    assert wrap_sse_stream_with_keepalive_pings(stream=stream, ping_interval_seconds=0) is stream
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_interval",
    [
        None,
        "abc",
        "",
        float("inf"),
        float("nan"),
        "-3",
        cast("float | str | None", [15]),
        cast("float | str | None", {"seconds": 15}),
    ],
)
async def test_invalid_config_interval_returns_stream_unwrapped(bad_interval: float | str | None):
    async def any_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK

    stream: Final = any_stream()
    assert wrap_sse_stream_with_keepalive_pings(stream=stream, ping_interval_seconds=bad_interval) is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_numeric_string_interval_from_yaml_config_enables_pings():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds="0.05")
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK


@pytest.mark.asyncio
async def test_create_response_streams_ping_first_for_slow_upstream():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    response: Final = await create_response(
        generator=wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds=0.05),
        media_type="text/event-stream",
        headers={},
    )

    assert isinstance(response, StreamingResponse)
    collected: Final = [chunk async for chunk in response.body_iterator]
    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK

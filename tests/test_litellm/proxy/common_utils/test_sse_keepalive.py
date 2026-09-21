import asyncio
from collections.abc import AsyncGenerator
from typing import Final, cast

import pytest
from fastapi.responses import StreamingResponse

from litellm.proxy.common_request_processing import create_response
from litellm.proxy.common_utils.sse_keepalive import (
    ANTHROPIC_PING_SSE_CHUNK,
    SSE_COMMENT_PING_BYTES,
    resolve_ttft_keepalive_interval,
    split_complete_sse_frames,
    wrap_passthrough_sse_bytes_with_keepalive_pings,
    wrap_sse_stream_with_keepalive_pings,
)

MESSAGE_START_CHUNK: Final = 'data: {"type": "message_start"}\n\n'
TEXT_DELTA_CHUNK: Final = 'data: {"type": "content_block_delta"}\n\n'


@pytest.mark.parametrize("delimiter", [b"\n\n", b"\r\n\r\n", b"\r\r"])
def test_split_complete_sse_frames_recognizes_every_sse_frame_delimiter(delimiter: bytes):
    newline: Final = delimiter[: len(delimiter) // 2]
    frame: Final = b"event: response.created" + newline + b"data: {}" + delimiter
    tail: Final = b"data: partial"

    assert split_complete_sse_frames(frame + tail) == (frame, tail)


def test_split_complete_sse_frames_holds_bytes_with_no_complete_frame():
    assert split_complete_sse_frames(b"data: unterminated") == (b"", b"data: unterminated")


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


SSE_FRAME_BYTES: Final = b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n'
BEDROCK_EVENT_STREAM_CONTENT_TYPE: Final = "application/vnd.amazon.eventstream"


@pytest.mark.asyncio
async def test_passthrough_ping_emitted_while_waiting_for_the_first_upstream_byte():
    async def slow_start_stream() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=slow_start_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == SSE_COMMENT_PING_BYTES
    assert collected[-1] == SSE_FRAME_BYTES
    assert b"".join(c for c in collected if c != SSE_COMMENT_PING_BYTES) == SSE_FRAME_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/event-stream", "text/event-stream; charset=utf-8", "TEXT/Event-Stream"])
async def test_passthrough_wraps_every_spelling_of_the_sse_content_type(content_type: str):
    async def slow_start_stream() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=slow_start_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": content_type},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES in collected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [BEDROCK_EVENT_STREAM_CONTENT_TYPE, "application/json", "application/x-ndjson", None, "text/event-streamish"],
)
async def test_passthrough_leaves_a_non_sse_transport_untouched(content_type: str | None):
    """A comment spliced into a binary transport (e.g. an AWS event stream) corrupts it."""

    async def any_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES

    stream: Final = any_stream()
    assert (
        wrap_passthrough_sse_bytes_with_keepalive_pings(
            stream=stream,
            ping_interval_seconds=0.05,
            upstream_headers={} if content_type is None else {"content-type": content_type},
        )
        is stream
    )
    await stream.aclose()


@pytest.mark.asyncio
async def test_passthrough_ping_is_never_spliced_into_a_half_delivered_frame():
    """Relayed chunks are raw transport reads, so an upstream can stall mid-frame."""

    async def stalls_mid_frame() -> AsyncGenerator[bytes, None]:
        yield b'event: content_block_delta\ndata: {"partial":'
        await asyncio.sleep(0.3)
        yield b"1}\n\n"

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=stalls_mid_frame(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES not in collected
    assert b"".join(collected) == b'event: content_block_delta\ndata: {"partial":1}\n\n'


@pytest.mark.asyncio
async def test_passthrough_ping_resumes_once_the_stalled_frame_completes():
    async def stalls_mid_frame_then_at_boundary() -> AsyncGenerator[bytes, None]:
        yield b'event: content_block_delta\ndata: {"partial":'
        await asyncio.sleep(0.2)
        yield b"1}\n\n"
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=stalls_mid_frame_then_at_boundary(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    ping_index: Final = collected.index(SSE_COMMENT_PING_BYTES)
    assert collected[:ping_index] == [b'event: content_block_delta\ndata: {"partial":', b"1}\n\n"]
    assert collected[-1] == SSE_FRAME_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_interval", [None, 0, "abc", float("inf"), float("nan"), "-3"])
async def test_passthrough_invalid_or_disabled_interval_returns_stream_unwrapped(bad_interval: float | str | None):
    async def any_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES

    stream: Final = any_stream()
    assert (
        wrap_passthrough_sse_bytes_with_keepalive_pings(
            stream=stream,
            ping_interval_seconds=bad_interval,
            upstream_headers={"content-type": "text/event-stream"},
        )
        is stream
    )
    await stream.aclose()


@pytest.mark.asyncio
async def test_passthrough_aclose_mid_silence_cancels_upstream_and_runs_its_cleanup():
    upstream_cleaned_up: Final = asyncio.Event()

    async def hung_stream() -> AsyncGenerator[bytes, None]:
        try:
            yield SSE_FRAME_BYTES
            await asyncio.Event().wait()
        finally:
            upstream_cleaned_up.set()

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=hung_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )

    assert await wrapped.__anext__() == SSE_FRAME_BYTES
    assert await wrapped.__anext__() == SSE_COMMENT_PING_BYTES
    await wrapped.aclose()

    assert upstream_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_passthrough_upstream_exception_propagates():
    async def failing_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES
        raise ValueError("upstream broke")

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=failing_stream(),
        ping_interval_seconds=5.0,
        upstream_headers={"content-type": "text/event-stream"},
    )

    assert await wrapped.__anext__() == SSE_FRAME_BYTES
    with pytest.raises(ValueError, match="upstream broke"):
        await wrapped.__anext__()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "split_frame",
    [
        (b'data: {"a": 1}\n', b"\n"),
        (b'data: {"a": 1}\r\n', b"\r\n"),
        (b'data: {"a": 1}\r', b"\n\r\n"),
        (b'data: {"a": 1}\r', b"\r"),
        (b'data: {"a": 1}\r\r', b""),
        (b'data: {"a": 1}\n\n', b""),
    ],
    ids=["lf-split", "crlf-split", "crlf-mixed-split", "cr-only-split", "cr-only-whole", "not-split"],
)
async def test_passthrough_sees_a_frame_delimiter_split_across_transport_chunks(split_frame):
    """A raw transport read can end mid-delimiter. Testing only the latest chunk
    would leave the stream looking permanently mid-frame, silently disabling the
    keepalive the operator configured."""

    async def split_delimiter_stream() -> AsyncGenerator[bytes, None]:
        for part in split_frame:
            if part:
                yield part
        await asyncio.sleep(0.3)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=split_delimiter_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES in collected
    assert b"".join(c for c in collected if c != SSE_COMMENT_PING_BYTES) == b"".join(split_frame) + SSE_FRAME_BYTES


def _deployment(keepalive_seconds=..., model="openai/gpt-4o"):
    params = {"model": model}
    if keepalive_seconds is not ...:
        params["keepalive_seconds"] = keepalive_seconds
    return {"model_name": "m", "litellm_params": params}


@pytest.mark.parametrize(
    "deployments, global_interval, expected, why",
    [
        ([], 30.0, 30.0, "no deployments known, the global applies"),
        ([_deployment()], 30.0, 30.0, "nothing configured, the global applies"),
        ([_deployment(0)], 30.0, None, "an operator's explicit 0 is a hard disable the global cannot lift"),
        ([_deployment("0")], 30.0, None, "the same, written as a yaml string"),
        ([_deployment(15)], 30.0, 15.0, "a deployment value wins over the global"),
        ([_deployment(15), _deployment(15)], 30.0, 15.0, "agreeing deployments are trusted"),
        ([_deployment(15), _deployment(60)], 30.0, 30.0, "disagreeing deployments fall back to the global"),
        ([_deployment(0), _deployment(30)], 30.0, 30.0, "a partial disable is not trusted before one is chosen"),
        ([_deployment(15)], None, 15.0, "a deployment value applies with no global set"),
        ([_deployment()], None, None, "nothing anywhere leaves it off"),
    ],
)
def test_ttft_interval_resolves_through_the_deployments_it_could_land_on(
    deployments, global_interval, expected, why
):
    assert resolve_ttft_keepalive_interval(deployments, global_interval) == expected, why

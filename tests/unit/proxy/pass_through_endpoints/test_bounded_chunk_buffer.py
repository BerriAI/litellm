import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.bounded_chunk_buffer import BoundedChunkBuffer
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _frame(name: str, payload: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


def test_buffer_under_the_limit_keeps_every_chunk_untouched():
    buffer = BoundedChunkBuffer(max_bytes=1024)
    chunks = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"]

    for chunk in chunks:
        buffer.append(chunk)

    assert list(buffer) == chunks, f"truncated={buffer.truncated}"
    assert not buffer.truncated


def test_buffer_without_a_limit_keeps_everything():
    buffer = BoundedChunkBuffer(max_bytes=None)
    chunks = [b"x" * 1_000_000 for _ in range(5)]

    for chunk in chunks:
        buffer.append(chunk)

    assert list(buffer) == chunks
    assert buffer.total_bytes == 5_000_000


def test_buffer_over_the_limit_keeps_the_first_and_last_events_whole():
    first = _frame("message_start", {"usage": {"input_tokens": 29}})
    middle = [_frame("content_block_delta", {"text": "x" * 40, "n": i}) for i in range(200)]
    last = _frame("message_delta", {"usage": {"output_tokens": 77}})
    buffer = BoundedChunkBuffer(max_bytes=600)

    for chunk in [first, *middle, last]:
        buffer.append(chunk)

    kept = b"".join(buffer)
    assert kept.startswith(first), kept
    assert kept.endswith(last), kept
    assert buffer.truncated
    assert buffer.retained_bytes <= 600
    assert buffer.total_bytes == len(first) + sum(map(len, middle)) + len(last)


def test_buffer_never_keeps_a_half_cut_event_when_chunks_split_frames():
    stream = b"".join(_frame("delta", {"i": i, "pad": "y" * 30}) for i in range(100))
    buffer = BoundedChunkBuffer(max_bytes=500)

    for start in range(0, len(stream), 7):
        buffer.append(stream[start : start + 7])

    for part in b"".join(buffer).split(b"\n\n"):
        if part:
            assert part.startswith(b"event: delta\ndata: {"), part
            assert part.endswith(b"}"), part


def test_buffer_holds_one_chunk_larger_than_the_limit_to_its_head_and_tail():
    buffer = BoundedChunkBuffer(max_bytes=100)

    buffer.append(b"event: big\ndata: " + b"z" * 10_000 + b"\n\n")

    assert buffer.truncated
    assert buffer.retained_bytes <= 100
    assert buffer.total_bytes == 10_000 + len(b"event: big\ndata: \n\n")


def test_buffer_memory_stays_bounded_while_a_huge_stream_passes():
    buffer = BoundedChunkBuffer(max_bytes=1000)

    for _ in range(10_000):
        buffer.append(b"data: " + b"q" * 1000 + b"\n\n")

    assert buffer.total_bytes > 10_000_000
    assert buffer.retained_bytes <= 1000


def test_buffer_trims_newline_delimited_streams_to_whole_lines():
    # no blank-line separators: frames are single lines, as in newline-delimited JSON
    buffer = BoundedChunkBuffer(max_bytes=40)
    lines = [json.dumps({"n": i}).encode() + b"\n" for i in range(20)]

    for line in lines:
        buffer.append(line)

    kept = b"".join(buffer)
    assert buffer.truncated
    assert kept.startswith(lines[0])
    assert kept.endswith(lines[-1])
    assert all(json.loads(part) is not None for part in kept.splitlines())


def test_buffer_supports_indexing_slicing_and_equality_with_a_list():
    buffer = BoundedChunkBuffer(max_bytes=1024)
    chunks = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"]
    for chunk in chunks:
        buffer.append(chunk)

    assert buffer[0] == chunks[0]
    assert buffer[-1] == chunks[-1]
    assert buffer[0:2] == chunks
    assert buffer == chunks
    assert buffer != chunks[:1]
    assert buffer != "not a sequence"


def _response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200

    async def _aiter_bytes():
        for chunk in chunks:
            yield chunk

    response.aiter_bytes = _aiter_bytes
    return response


@pytest.mark.asyncio
async def test_chunk_processor_relays_every_byte_but_logs_only_the_bounded_ends(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    monkeypatch.setattr(litellm, "pass_through_max_logged_stream_bytes", 400)
    first = _frame("message_start", {"usage": {"input_tokens": 29}})
    middle = [_frame("content_block_delta", {"text": "x" * 40, "n": i}) for i in range(100)]
    last = _frame("message_delta", {"usage": {"output_tokens": 77}})
    chunks = [first, *middle, last]
    logging_obj = MagicMock()
    logging_obj._on_deferred_stream_complete = None

    with patch.object(PassThroughStreamingHandler, "route_streaming_logging_to_handler", new=AsyncMock()) as route:
        relayed = [
            chunk
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=_response(chunks),
                request_body={"model": "m"},
                litellm_logging_obj=logging_obj,
                endpoint_type=EndpointType.GENERIC,
                start_time=datetime.now(),
                passthrough_success_handler_obj=MagicMock(),
                url_route="/codex/responses",
            )
        ]
        import asyncio

        await asyncio.sleep(0)

    assert relayed == chunks
    logged = b"".join(route.call_args.kwargs["raw_bytes"])
    assert logged.startswith(first) and logged.endswith(last), logged
    assert len(logged) <= 400

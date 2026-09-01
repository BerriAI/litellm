"""
Tests for restamping the public model on Anthropic Messages streaming chunks.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.anthropic_endpoints.streaming_model_restamp import (
    AnthropicStreamModelRestamper,
    restamp_anthropic_stream_chunk_model,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


def _message_start_frame(model: str, line_end: str = "\n") -> bytes:
    payload = {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": []},
    }
    return f"event: message_start{line_end}data: {json.dumps(payload)}{line_end}{line_end}".encode()


def _proxy_logging_obj_streaming(frames: list[bytes]) -> MagicMock:
    async def _iterator_hook(**_kwargs):
        for frame in frames:
            yield frame

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.async_post_call_streaming_iterator_hook = _iterator_hook
    proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["response"])
    return proxy_logging_obj


def _model_from_frame(frame: bytes | str) -> str:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else frame
    data_line = next(line for line in text.split("\n") if line.startswith("data:"))
    return json.loads(data_line[len("data:") :])["message"]["model"]


def test_restamps_sse_bytes_frame():
    restamped = restamp_anthropic_stream_chunk_model(
        _message_start_frame("claude-haiku-4-5-20251001"), "claude-auto-1"
    )

    assert isinstance(restamped, bytes)
    assert _model_from_frame(restamped) == "claude-auto-1"
    assert b"event: message_start" in restamped


def test_restamps_event_dict():
    chunk = {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6"}}

    restamped = restamp_anthropic_stream_chunk_model(chunk, "claude-auto-2")

    assert restamped == {"type": "message_start", "message": {"id": "msg_1", "model": "claude-auto-2"}}
    assert chunk["message"]["model"] == "claude-sonnet-4-6"


@pytest.mark.parametrize(
    "chunk",
    [
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_start", "message": "not-a-dict"},
        b"event: message_start\ndata: not-json\n\n",
        b"data: [DONE]\n\n",
    ],
)
def test_leaves_chunks_without_a_model_untouched(chunk):
    assert restamp_anthropic_stream_chunk_model(chunk, "claude-auto-1") == chunk


@pytest.mark.asyncio
async def test_sse_generator_publishes_requested_model_on_message_start():
    """The message_start event reports the requested model, not the provider's."""
    delta_frame = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    proxy_logging_obj = _proxy_logging_obj_streaming([_message_start_frame("claude-haiku-4-5-20251001"), delta_frame])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
            restamp_model="claude-auto-1",
        )
    ]

    assert _model_from_frame(chunks[0]) == "claude-auto-1"
    assert chunks[1] == delta_frame


@pytest.mark.asyncio
async def test_sse_generator_keeps_provider_model_when_restamping_is_off():
    proxy_logging_obj = _proxy_logging_obj_streaming([_message_start_frame("claude-haiku-4-5-20251001")])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
        )
    ]

    assert _model_from_frame(chunks[0]) == "claude-haiku-4-5-20251001"


def test_restamps_message_start_split_across_transport_chunks():
    frame = _message_start_frame("claude-haiku-4-5-20251001")
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    held = restamper.process(frame[:25])
    emitted = restamper.process(frame[25:])

    assert held == b""
    assert isinstance(emitted, bytes)
    assert _model_from_frame(emitted) == "claude-auto-1"


def test_emits_coalesced_frames_with_only_message_start_rewritten():
    delta = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    combined = _message_start_frame("claude-haiku-4-5-20251001") + delta

    emitted = restamper_output = AnthropicStreamModelRestamper("claude-auto-1").process(combined)

    assert isinstance(restamper_output, bytes)
    assert _model_from_frame(emitted) == "claude-auto-1"
    assert emitted.endswith(delta)


def test_ping_frames_keep_the_restamper_armed():
    ping = b'event: ping\ndata: {"type": "ping"}\n\n'
    frame = _message_start_frame("claude-haiku-4-5-20251001")
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    assert restamper.process(ping) == ping
    reassembled = restamper.process(frame[:10])
    reassembled += restamper.process(frame[10:])

    assert _model_from_frame(reassembled) == "claude-auto-1"


def test_first_non_ping_event_disarms_the_restamper():
    delta = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    late_message_start = _message_start_frame("claude-haiku-4-5-20251001")
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    assert restamper.process(delta) == delta
    assert restamper.process(late_message_start) == late_message_start


def test_oversized_unterminated_chunk_flushes_unmodified():
    blob = b"data: " + b"x" * 70000
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    assert restamper.process(blob) == blob
    frame = _message_start_frame("claude-haiku-4-5-20251001")
    assert restamper.process(frame) == frame


def test_dict_message_start_disarms_after_restamp():
    restamper = AnthropicStreamModelRestamper("claude-auto-1")
    first = restamper.process({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6"}})
    second = {"type": "message_start", "message": {"id": "msg_2", "model": "claude-sonnet-4-6"}}

    assert first == {"type": "message_start", "message": {"id": "msg_1", "model": "claude-auto-1"}}
    assert restamper.process(second) == second


@pytest.mark.asyncio
async def test_sse_generator_restamps_message_start_split_across_chunks():
    frame = _message_start_frame("claude-haiku-4-5-20251001")
    proxy_logging_obj = _proxy_logging_obj_streaming([frame[:30], frame[30:]])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
            restamp_model="claude-auto-1",
        )
    ]

    joined = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8") for chunk in chunks)
    assert _model_from_frame(joined) == "claude-auto-1"


def test_restamps_crlf_terminated_message_start_frame():
    frame = _message_start_frame("claude-haiku-4-5-20251001", line_end="\r\n")
    delta = b'event: content_block_delta\r\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\r\n\r\n'
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    emitted = restamper.process(frame)

    assert isinstance(emitted, bytes)
    assert _model_from_frame(emitted) == "claude-auto-1"
    assert emitted.endswith(b"\r\n\r\n")
    assert restamper.process(delta) == delta


def test_restamps_cr_terminated_message_start_frame():
    frame = _message_start_frame("claude-haiku-4-5-20251001", line_end="\r")
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    emitted = restamper.process(frame)

    assert isinstance(emitted, bytes)
    assert b'"model":"claude-auto-1"' in emitted
    assert emitted.endswith(b"\r\r")


def test_restamps_crlf_message_start_split_across_transport_chunks():
    frame = _message_start_frame("claude-haiku-4-5-20251001", line_end="\r\n")
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    held = restamper.process(frame[:25])
    emitted = restamper.process(frame[25:])

    assert held == b""
    assert isinstance(emitted, bytes)
    assert _model_from_frame(emitted) == "claude-auto-1"


def test_flush_returns_restamped_held_tail():
    unterminated = _message_start_frame("claude-haiku-4-5-20251001")[:-2]
    restamper = AnthropicStreamModelRestamper("claude-auto-1")

    assert restamper.process(unterminated) == b""
    flushed = restamper.flush()

    assert b'"model":"claude-auto-1"' in flushed
    assert restamper.flush() == b""


def test_flush_disarms_the_restamper():
    restamper = AnthropicStreamModelRestamper("claude-auto-1")
    frame = _message_start_frame("claude-haiku-4-5-20251001")

    assert restamper.flush() == b""
    assert restamper.process(frame) == frame


@pytest.mark.asyncio
async def test_sse_generator_flushes_held_tail_at_end_of_stream():
    unterminated = _message_start_frame("claude-haiku-4-5-20251001")[:-2]
    proxy_logging_obj = _proxy_logging_obj_streaming([unterminated])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
            restamp_model="claude-auto-1",
        )
    ]

    joined = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8") for chunk in chunks)
    assert b'"model":"claude-auto-1"' in joined


@pytest.mark.asyncio
async def test_sse_generator_restamps_crlf_stream():
    frame = _message_start_frame("claude-haiku-4-5-20251001", line_end="\r\n")
    delta = b'event: content_block_delta\r\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\r\n\r\n'
    proxy_logging_obj = _proxy_logging_obj_streaming([frame, delta])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
            restamp_model="claude-auto-1",
        )
    ]

    assert _model_from_frame(chunks[0]) == "claude-auto-1"
    assert chunks[1] == delta

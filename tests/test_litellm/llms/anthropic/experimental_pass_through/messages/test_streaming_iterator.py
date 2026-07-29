import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    INCOMPLETE_STREAM_ERROR_MESSAGE,
    BaseAnthropicMessagesStreamingIterator,
    _incomplete_stream_error_sse_event,
    _is_message_stop_chunk,
)


class _RecordingLoggingIterator(BaseAnthropicMessagesStreamingIterator):
    def __init__(self, litellm_logging_obj: LiteLLMLoggingObj, request_body: dict):
        super().__init__(litellm_logging_obj=litellm_logging_obj, request_body=request_body)
        self.logged_chunks: list = []

    async def _handle_streaming_logging(self, collected_chunks):
        self.logged_chunks = list(collected_chunks)


def _make_logging_obj(test_name: str) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="chat",
        start_time=datetime.now(),
        litellm_call_id=test_name,
        function_id=test_name,
    )


def _make_iterator(test_name: str) -> BaseAnthropicMessagesStreamingIterator:
    return BaseAnthropicMessagesStreamingIterator(
        litellm_logging_obj=_make_logging_obj(test_name),
        request_body={},
    )


async def _collect(iterator, stream):
    return [chunk async for chunk in iterator.async_sse_wrapper(stream)]


TRUNCATED_TOOL_USE_EVENTS = (
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tooluse_1", "name": "write", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"path": "/builder/docs/QUAL'},
    },
)


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_event_when_stream_ends_without_message_stop():
    """
    Regression test for LIT-3724: a Bedrock stream that goes silent
    mid tool_use must not be passed through as a successful, complete
    SSE stream. An `error` SSE event must be appended so strict clients
    (Anthropic SDK, Claude Code) surface the truncation instead of
    crashing on unterminated tool-call JSON.
    """

    async def _truncated_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event

    iterator = _make_iterator("test_truncated_stream_emits_error")
    chunks = await _collect(iterator, _truncated_stream())

    assert len(chunks) == len(TRUNCATED_TOOL_USE_EVENTS) + 1
    error_chunk = chunks[-1].decode()
    assert error_chunk.startswith("event: error\n")
    assert error_chunk.endswith("\n\n")

    error_payload = json.loads(error_chunk.split("data: ", 1)[1])
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert error_payload["error"]["message"] == INCOMPLETE_STREAM_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_async_sse_wrapper_no_error_event_on_complete_stream():
    async def _complete_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}
        yield {"type": "message_stop"}

    iterator = _make_iterator("test_complete_stream_no_error")
    chunks = await _collect(iterator, _complete_stream())

    assert len(chunks) == len(TRUNCATED_TOOL_USE_EVENTS) + 3
    decoded = [chunk.decode() for chunk in chunks]
    assert decoded[-1].startswith("event: message_stop\n")
    assert not any(chunk.startswith("event: error\n") for chunk in decoded)


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_event_on_empty_stream():
    async def _empty_stream():
        return
        yield

    iterator = _make_iterator("test_empty_stream_emits_error")
    chunks = await _collect(iterator, _empty_stream())

    assert len(chunks) == 1
    assert chunks[0].decode().startswith("event: error\n")


@pytest.mark.asyncio
async def test_async_sse_wrapper_treats_message_stop_bytes_as_complete():
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    iterator = _make_iterator("test_byte_stream_message_stop")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 2
    assert not any(chunk.startswith(b"event: error\n") for chunk in chunks)


def test_is_message_stop_chunk():
    assert _is_message_stop_chunk({"type": "message_stop"}) is True
    assert _is_message_stop_chunk({"type": "message_delta"}) is False
    assert _is_message_stop_chunk(b'event: message_stop\ndata: {}\n\n') is True
    assert _is_message_stop_chunk(b"raw-bytes") is False
    assert _is_message_stop_chunk("message_stop") is False


def test_is_message_stop_chunk_ignores_substring_in_payload():
    """
    Regression: a `content_block_delta` frame whose payload happens to contain
    the literal string `message_stop` (e.g. inside a tool's partial_json) must
    not be treated as a terminal stop event.
    """
    delta_frame_with_substring = (
        b'event: content_block_delta\n'
        b'data: {"type": "content_block_delta", "delta": '
        b'{"type": "input_json_delta", "partial_json": "\\"message_stop\\""}}\n\n'
    )
    assert _is_message_stop_chunk(delta_frame_with_substring) is False


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_when_bytes_stream_only_mentions_message_stop_in_payload():
    """
    Regression for the bytes-branch substring false positive: a stream whose
    payload text contains `message_stop` (but never emits the actual
    `event: message_stop` frame) must still be flagged as incomplete.
    """
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield (
            b'event: content_block_delta\n'
            b'data: {"type": "content_block_delta", "delta": '
            b'{"type": "input_json_delta", "partial_json": "\\"message_stop\\""}}\n\n'
        )

    iterator = _make_iterator("test_bytes_substring_does_not_mark_complete")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 3
    assert chunks[-1].decode().startswith("event: error\n")


@pytest.mark.asyncio
async def test_async_sse_wrapper_does_not_double_error_on_provider_error_dict():
    """
    Regression: when the provider itself terminates the stream with an
    `error` event (without a `message_stop`), the wrapper must forward that
    error and not append a second synthetic incomplete-stream error.
    """
    provider_error = {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}

    async def _error_terminated_stream():
        yield {"type": "message_start", "message": {"id": "msg_1"}}
        yield provider_error

    iterator = _make_iterator("test_provider_error_terminal_dict")
    chunks = await _collect(iterator, _error_terminated_stream())

    assert len(chunks) == 2
    error_frames = [c for c in chunks if c.startswith(b"event: error\n")]
    assert len(error_frames) == 1
    payload = json.loads(error_frames[0].decode().split("data: ", 1)[1])
    assert payload == provider_error


@pytest.mark.asyncio
async def test_async_sse_wrapper_does_not_double_error_on_provider_error_bytes():
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error"}}\n\n'

    iterator = _make_iterator("test_provider_error_terminal_bytes")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 2
    error_frames = [c for c in chunks if c.startswith(b"event: error\n")]
    assert len(error_frames) == 1


@pytest.mark.asyncio
async def test_async_sse_wrapper_excludes_synthetic_error_event_from_logged_chunks():
    async def _truncated_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_synthetic_error_not_logged"),
        request_body={},
    )
    chunks = await _collect(iterator, _truncated_stream())

    assert chunks[-1].startswith(b"event: error\n")
    assert iterator.logged_chunks == chunks[:-1]
    assert not any(chunk.startswith(b"event: error\n") for chunk in iterator.logged_chunks)


def test_incomplete_stream_error_sse_event_is_valid_anthropic_error():
    event = _incomplete_stream_error_sse_event().decode()
    lines = event.split("\n")
    assert lines[0] == "event: error"
    payload = json.loads(lines[1].removeprefix("data: "))
    assert payload == {
        "type": "error",
        "error": {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE},
    }
    assert event.endswith("\n\n")

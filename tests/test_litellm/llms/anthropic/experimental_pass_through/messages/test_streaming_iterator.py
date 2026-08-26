import asyncio
import json
from datetime import datetime

import pytest


from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    INCOMPLETE_STREAM_ERROR_MESSAGE,
    AnthropicMessagesStreamHiddenParams,
    AnthropicMessagesStreamingResponse,
    BaseAnthropicMessagesStreamingIterator,
    _incomplete_stream_error_sse_event,
    _is_message_stop_chunk,
    _is_provider_error_chunk,
    anthropic_messages_response_as_sse_events,
    is_anthropic_content_delta_chunk,
    parse_anthropic_error_event,
)


class _RecordingLoggingIterator(BaseAnthropicMessagesStreamingIterator):
    def __init__(self, litellm_logging_obj: LiteLLMLoggingObj, request_body: dict):
        super().__init__(litellm_logging_obj=litellm_logging_obj, request_body=request_body)
        self.logged_chunks: list = []
        self.logging_call_count: int = 0

    async def _handle_streaming_logging(self, collected_chunks):
        self.logged_chunks = list(collected_chunks)
        self.logging_call_count += 1


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


def test_parse_anthropic_error_event_from_dict_chunk():
    """Regression for #24004: dict-shaped error chunks parse to
    (type, message, status) so the Router can decide whether to fall back."""
    chunk = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    assert parse_anthropic_error_event(chunk) == ("overloaded_error", "Overloaded", 503)
    assert _is_provider_error_chunk(chunk) is True


def test_parse_anthropic_error_event_from_sse_bytes():
    """Regression for #24004: a raw `event: error` SSE frame (what a native
    Anthropic/Bedrock passthrough forwards verbatim today) must parse
    identically to the dict shape so the Router can raise a fallback."""
    sse_chunk = (
        b"event: error\n"
        b'data: {"type": "error", "error": {"type": "internal_server_error", "message": "boom"}}\n\n'
    )
    assert parse_anthropic_error_event(sse_chunk) == ("internal_server_error", "boom", 500)
    assert _is_provider_error_chunk(sse_chunk) is True


def test_parse_anthropic_error_event_defaults_status_for_unknown_type():
    chunk = {"type": "error", "error": {"type": "some_future_error_type", "message": "?"}}
    assert parse_anthropic_error_event(chunk) == ("some_future_error_type", "?", 500)


def test_parse_anthropic_error_event_missing_message_falls_back_to_type():
    chunk = {"type": "error", "error": {"type": "overloaded_error"}}
    assert parse_anthropic_error_event(chunk) == ("overloaded_error", "overloaded_error", 503)


def test_parse_anthropic_error_event_non_string_error_type_returns_none():
    """A malformed error body whose `type` field isn't a string (e.g. an
    upstream bug sends null or a number) must not be treated as an error
    event rather than crashing or forwarding a garbage error_type."""
    chunk = {"type": "error", "error": {"type": None, "message": "boom"}}
    assert parse_anthropic_error_event(chunk) is None


def test_decoded_sse_data_line_swallows_invalid_json():
    """A `data:` line that isn't valid JSON (a malformed/truncated frame)
    must not be treated as an error event or raise, just be ignored."""
    malformed_frame = b"event: error\ndata: {not valid json\n\n"
    assert parse_anthropic_error_event(malformed_frame) is None
    assert _is_provider_error_chunk(malformed_frame) is False


class TestIsAnthropicContentDeltaChunk:
    def test_dict_content_block_delta(self):
        assert is_anthropic_content_delta_chunk({"type": "content_block_delta"}) is True

    def test_dict_other_type(self):
        assert is_anthropic_content_delta_chunk({"type": "message_start"}) is False

    def test_bytes_content_block_delta(self):
        assert is_anthropic_content_delta_chunk(b"event: content_block_delta\ndata: {}\n\n") is True

    def test_bytes_other_event(self):
        assert is_anthropic_content_delta_chunk(b"event: message_start\ndata: {}\n\n") is False

    def test_neither_dict_nor_bytes(self):
        assert is_anthropic_content_delta_chunk("content_block_delta") is False
        assert is_anthropic_content_delta_chunk(None) is False


@pytest.mark.parametrize(
    "chunk",
    [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n',
        b"raw-bytes",
        "error",
        None,
    ],
)
def test_parse_anthropic_error_event_non_error_chunks_return_none(chunk):
    assert parse_anthropic_error_event(chunk) is None
    assert _is_provider_error_chunk(chunk) is False


def test_parse_anthropic_error_event_ignores_substring_in_payload():
    """A content_block_delta whose partial_json happens to contain the
    literal string `"type": "error"` must not be misread as an error event."""
    delta_frame_with_substring = (
        b"event: content_block_delta\n"
        b'data: {"type": "content_block_delta", "delta": '
        b'{"type": "input_json_delta", "partial_json": "\\"type\\": \\"error\\""}}\n\n'
    )
    assert parse_anthropic_error_event(delta_frame_with_substring) is None


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


async def _events_then_hang(events):
    for event in events:
        yield event
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_async_sse_wrapper_logs_partial_chunks_on_client_disconnect():
    """
    Regression test for LIT-5839: a client disconnect tears the generator
    down with GeneratorExit at the yield, which used to skip the post-loop
    logging dispatch entirely, so the partial output tokens the provider
    already generated (and billed) never reached spend tracking.
    """
    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_disconnect_logs_partial_chunks"),
        request_body={},
    )
    wrapped = iterator.async_sse_wrapper(_events_then_hang(TRUNCATED_TOOL_USE_EVENTS))
    streamed = [await wrapped.__anext__() for _ in range(len(TRUNCATED_TOOL_USE_EVENTS))]
    assert iterator.logging_call_count == 0

    await wrapped.aclose()

    assert iterator.logging_call_count == 1
    assert iterator.logged_chunks == streamed


@pytest.mark.asyncio
async def test_async_sse_wrapper_logs_partial_chunks_on_cancellation():
    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_cancellation_logs_partial_chunks"),
        request_body={},
    )
    wrapped = iterator.async_sse_wrapper(_events_then_hang(TRUNCATED_TOOL_USE_EVENTS))
    streamed = [await wrapped.__anext__() for _ in range(len(TRUNCATED_TOOL_USE_EVENTS))]

    consume_task = asyncio.ensure_future(wrapped.__anext__())
    await asyncio.sleep(0.01)
    consume_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consume_task

    assert iterator.logging_call_count == 1
    assert iterator.logged_chunks == streamed


@pytest.mark.asyncio
async def test_async_sse_wrapper_skips_logging_on_disconnect_before_first_chunk():
    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_disconnect_before_first_chunk"),
        request_body={},
    )
    wrapped = iterator.async_sse_wrapper(_events_then_hang(()))

    consume_task = asyncio.ensure_future(wrapped.__anext__())
    await asyncio.sleep(0.01)
    consume_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consume_task

    assert iterator.logging_call_count == 0


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


def _decode_sse_events(events: tuple[bytes, ...]) -> list[tuple[str, dict]]:
    decoded = []
    for event in events:
        assert isinstance(event, bytes)
        lines = event.decode().split("\n")
        assert lines[0].startswith("event: ")
        decoded.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return decoded


def test_anthropic_messages_response_as_sse_events_text_block():
    response = {
        "id": "msg_1",
        "model": "claude-haiku",
        "role": "assistant",
        "type": "message",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    types = [event_type for event_type, _ in decoded]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # message_start must not carry generated content itself, matching a real
    # streaming response - it arrives via the content_block_delta that follows.
    assert decoded[0][1]["message"]["content"] == []
    assert decoded[0][1]["message"]["id"] == "msg_1"
    # Bugbot regression: message_start must not carry the completed response's
    # final stop_reason/stop_sequence/output_tokens - a real stream keeps those
    # null/zero until message_delta, so a client could otherwise treat the
    # message as already finished, or double-count output tokens.
    assert decoded[0][1]["message"]["stop_reason"] is None
    assert decoded[0][1]["message"]["stop_sequence"] is None
    assert decoded[0][1]["message"]["usage"] == {"input_tokens": 3, "output_tokens": 0}
    assert decoded[1][1]["content_block"] == {"type": "text", "text": ""}
    assert decoded[2][1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert decoded[4][1]["delta"]["stop_reason"] == "end_turn"
    assert decoded[4][1]["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_anthropic_messages_response_as_sse_events_tool_use_block():
    response = {
        "id": "msg_2",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}}],
        "stop_reason": "tool_use",
    }
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    content_block_start = dict(decoded)["content_block_start"]
    assert content_block_start["content_block"] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {},
    }
    content_block_delta = dict(decoded)["content_block_delta"]
    assert json.loads(content_block_delta["delta"]["partial_json"]) == {"city": "NYC"}
    assert content_block_delta["delta"]["type"] == "input_json_delta"


def test_anthropic_messages_response_as_sse_events_thinking_block_emits_signature_delta():
    """Bugbot regression: a thinking block's real `signature` must reach the
    client via a trailing signature_delta, not be silently dropped - Anthropic
    rejects a replayed assistant message (a follow-up turn, a tool-use
    continuation) whose thinking block lacks its original signature."""
    response = {
        "id": "msg_5",
        "content": [{"type": "thinking", "thinking": "let me think", "signature": "sig-abc123"}],
        "stop_reason": "end_turn",
    }
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    deltas = [payload["delta"] for event_type, payload in decoded if event_type == "content_block_delta"]
    assert deltas == [
        {"type": "thinking_delta", "thinking": "let me think"},
        {"type": "signature_delta", "signature": "sig-abc123"},
    ]


def test_anthropic_messages_response_as_sse_events_thinking_block_without_signature_omits_delta():
    response = {
        "id": "msg_6",
        "content": [{"type": "thinking", "thinking": "let me think", "signature": None}],
        "stop_reason": "end_turn",
    }
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    deltas = [payload["delta"] for event_type, payload in decoded if event_type == "content_block_delta"]
    assert deltas == [{"type": "thinking_delta", "thinking": "let me think"}]


def test_anthropic_messages_response_as_sse_events_multiple_blocks_are_indexed():
    response = {
        "id": "msg_3",
        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
    }
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    starts = [payload for event_type, payload in decoded if event_type == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    deltas = [payload for event_type, payload in decoded if event_type == "content_block_delta"]
    assert [d["delta"]["text"] for d in deltas] == ["a", "b"]


def test_anthropic_messages_streaming_response_reports_withheld_output_of_its_stream():
    class _HoldingBack:
        has_buffered_provider_output = True

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    async def _bare_stream():
        yield b""

    hidden_params: AnthropicMessagesStreamHiddenParams = {"additional_headers": {}}
    assert (
        AnthropicMessagesStreamingResponse(completion_stream=_HoldingBack(), hidden_params=hidden_params)
        .has_buffered_provider_output
        is True
    )
    assert (
        AnthropicMessagesStreamingResponse(completion_stream=_bare_stream(), hidden_params=hidden_params)
        .has_buffered_provider_output
        is False
    )


def test_anthropic_messages_response_as_sse_events_no_content_blocks():
    response = {"id": "msg_4", "content": [], "stop_reason": "end_turn"}
    decoded = _decode_sse_events(anthropic_messages_response_as_sse_events(response))
    assert [event_type for event_type, _ in decoded] == ["message_start", "message_delta", "message_stop"]

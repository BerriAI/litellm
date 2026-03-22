"""Tests for LangGraph SSE stream iterator."""

import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.langgraph.chat.sse_iterator import LangGraphSSEStreamIterator


def _make_iterator():
    """Create iterator with mock response (we only call _parse_sse_line directly)."""
    mock_resp = MagicMock()
    return LangGraphSSEStreamIterator(mock_resp, model="test-model")


def test_event_header_sets_current_event_type():
    """event: line sets _current_event_type for next data line."""
    it = _make_iterator()
    assert it._parse_sse_line("event: messages") is None
    assert it._current_event_type == "messages"

    it = _make_iterator()
    assert it._parse_sse_line("event: metadata") is None
    assert it._current_event_type == "metadata"


def test_event_header_stripped():
    """event: value is stripped of whitespace."""
    it = _make_iterator()
    it._parse_sse_line("event:  messages  ")
    assert it._current_event_type == "messages"


def test_blank_line_resets_event_type():
    """Blank line (SSE boundary) resets _current_event_type so it does not leak."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    assert it._current_event_type == "messages"

    it._parse_sse_line("")
    assert it._current_event_type is None


def test_event_type_does_not_leak_across_events():
    """Event type from first event must not apply to second event's data."""
    it = _make_iterator()
    # First event: messages
    it._parse_sse_line("event: messages")
    chunk1 = it._parse_sse_line('data: [[{"type":"ai","content":"hello"}]]')
    assert chunk1 is not None

    # Blank line - boundary
    it._parse_sse_line("")

    # Second event: no event header, only data (tuple format)
    chunk2 = it._parse_sse_line('data: ["metadata", {"run_id": "x"}]')
    assert chunk2 is not None
    assert chunk2.choices[0].finish_reason == "stop"


def test_data_with_event_header_returns_chunk():
    """event: messages + data with AI content returns content chunk."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    chunk = it._parse_sse_line('data: [[{"type":"ai","content":"hi"}]]')
    assert chunk is not None
    assert chunk.choices[0].delta.content == "hi"


def test_metadata_before_messages_does_not_stop_stream():
    """Initial metadata with run_id should not end the stream."""
    it = _make_iterator()
    result = it._parse_sse_line('data: ["metadata", {"run_id": "abc", "attempt": 1}]')
    assert result is None
    assert it.finished is False


def test_metadata_after_messages_stops_stream():
    """Metadata with run_id after content has been received should stop."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    it._parse_sse_line('data: [[{"type":"ai","content":"hello"}]]')
    assert it._has_received_messages is True

    chunk = it._parse_sse_line('data: ["metadata", {"run_id": "abc"}]')
    assert chunk is not None
    assert chunk.choices[0].finish_reason == "stop"
    assert it.finished is True


def test_standard_sse_metadata_before_messages_skipped():
    """Standard SSE event: metadata before messages should not stop stream."""
    it = _make_iterator()
    it._parse_sse_line("event: metadata")
    result = it._parse_sse_line('data: {"run_id": "abc", "attempt": 1}')
    assert result is None
    assert it.finished is False


def test_full_stream_metadata_first_then_content():
    """End-to-end: metadata arrives first, then messages, then metadata ends stream."""
    lines = [
        'data: ["metadata", {"run_id": "r1", "attempt": 1}]',
        'data: ["messages", [[{"type": "ai", "content": "hi"}, {}]]]',
        'data: ["metadata", {"run_id": "r1"}]',
    ]
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    it = iter(LangGraphSSEStreamIterator(mock_resp, model="test"))

    first = next(it)
    assert first.choices[0].delta.content == "hi"
    assert first.choices[0].finish_reason is None

    second = next(it)
    assert second.choices[0].finish_reason == "stop"


def test_tool_call_only_ai_message_sets_has_received_messages():
    """Empty content with tool_calls must set _has_received_messages and emit tool deltas."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    msg = {
        "type": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "get_weather",
                "args": {"city": "Paris"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    }
    chunk = it._parse_sse_line("data: " + json.dumps([[msg]]))
    assert it._has_received_messages is True
    assert chunk is not None
    assert chunk.choices[0].delta.tool_calls is not None
    assert chunk.choices[0].delta.tool_calls[0].function.name == "get_weather"


def test_tool_calls_openai_shape_passed_through():
    """AIMessageChunk with OpenAI-style tool_calls nested under function."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    msg = {
        "type": "AIMessageChunk",
        "content": "",
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "foo", "arguments": '{"a": 1}'},
            }
        ],
    }
    chunk = it._parse_sse_line("data: " + json.dumps([[msg]]))
    assert chunk is not None
    assert chunk.choices[0].delta.tool_calls[0].function.name == "foo"
    assert '"a"' in chunk.choices[0].delta.tool_calls[0].function.arguments


def test_list_content_reasoning_and_text_blocks():
    """List-shaped content yields text in delta.content and reasoning in reasoning_content."""
    it = _make_iterator()
    it._parse_sse_line("event: messages")
    content = [
        {"type": "reasoning", "reasoning": "step"},
        {"type": "text", "text": "answer"},
    ]
    chunk = it._parse_sse_line(
        "data: " + json.dumps([[{"type": "ai", "content": content}]])
    )
    assert chunk.choices[0].delta.content == "answer"
    assert chunk.choices[0].delta.reasoning_content == "step"


def test_dict_fallback_sets_has_received_messages():
    """Top-level dict with content must flip _has_received_messages when emitting."""
    it = _make_iterator()
    it._process_data({"content": "z"}, event_type=None)
    assert it._has_received_messages is True

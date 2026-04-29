"""
Tests for LangGraphSSEStreamIterator SSE parsing - Bug #24093.

Validates that:
1. Standard SSE format (event: + data:) is parsed correctly.
2. Legacy tuple format (data: ["messages", ...]) still works.
3. Mixed / edge-case payloads are handled gracefully.
"""

import json
from typing import List
from unittest.mock import MagicMock

import pytest

from litellm.llms.langgraph.chat.sse_iterator import LangGraphSSEStreamIterator
from litellm.types.utils import ModelResponseStream

MODEL = "langgraph/my-agent"


# Helpers

def _make_iterator(lines: List[str]) -> LangGraphSSEStreamIterator:
    """Create an iterator backed by a mock httpx.Response whose iter_lines
    yields the provided *lines* one-by-one."""
    response = MagicMock()
    response.iter_lines.return_value = iter(lines)
    it = LangGraphSSEStreamIterator(response=response, model=MODEL)
    # Trigger __iter__ so line_iterator is populated
    iter(it)
    return it


def _collect_content(it: LangGraphSSEStreamIterator) -> List[str]:
    """Exhaust the iterator and return a list of content strings from chunks."""
    contents: List[str] = []
    for chunk in it:
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                contents.append(choice.delta.content)
    return contents


# Tests - Standard SSE format (event: header + data:)

class TestStandardSSEFormat:
    """Standard LangGraph SSE: ``event: messages\\ndata: [...]``."""

    def test_should_parse_ai_message_chunk(self):
        """event: messages with AIMessageChunk type extracts content."""
        lines = [
            "event: messages",
            'data: [{"content": "Hello world", "type": "AIMessageChunk"}, {}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["Hello world"]

    def test_should_parse_ai_type_message(self):
        """event: messages with 'ai' type extracts content."""
        lines = [
            "event: messages",
            'data: [{"content": "Hi there", "type": "ai"}, {"some": "meta"}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["Hi there"]

    def test_should_skip_human_messages(self):
        """Human messages should not produce content chunks."""
        lines = [
            "event: messages",
            'data: [{"content": "I am a human", "type": "human"}, {}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == []

    def test_should_handle_multiple_events(self):
        """Multiple SSE event blocks should each produce content."""
        lines = [
            "event: messages",
            'data: [{"content": "First", "type": "AIMessageChunk"}, {}]',
            "",
            "event: messages",
            'data: [{"content": "Second", "type": "AIMessageChunk"}, {}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["First", "Second"]

    def test_should_handle_metadata_event_with_run_id(self):
        """Metadata event with run_id should produce a stop chunk."""
        lines = [
            "event: messages",
            'data: [{"content": "answer", "type": "ai"}, {}]',
            "",
            "event: metadata",
            'data: {"run_id": "abc-123"}',
            "",
        ]
        it = _make_iterator(lines)
        chunks = list(it)
        # Should have a content chunk + a final stop chunk
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "answer"
        assert chunks[1].choices[0].finish_reason == "stop"

    def test_should_handle_messages_partial_event(self):
        """``event: messages/partial`` should be treated like messages."""
        lines = [
            "event: messages/partial",
            'data: [{"content": "partial", "type": "AIMessageChunk"}, {}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["partial"]


# Tests - Legacy tuple format (data: ["messages", ...])

class TestLegacyTupleFormat:
    """Legacy ``stream_mode="messages-tuple"``: ``data: ["messages", ...]``."""

    def test_should_parse_nested_tuple_payload(self):
        """Nested list payload: ["messages", [[msg, meta]]]."""
        payload = [
            "messages",
            [[{"content": "legacy hello", "type": "ai"}, {"meta": True}]],
        ]
        lines = [f"data: {json.dumps(payload)}"]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["legacy hello"]

    def test_should_parse_flat_dict_tuple_payload(self):
        """Flat dict payload: ["messages", [msg_dict, ...]]."""
        payload = [
            "messages",
            [{"content": "flat msg", "type": "AIMessageChunk"}],
        ]
        lines = [f"data: {json.dumps(payload)}"]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["flat msg"]

    def test_should_handle_metadata_tuple(self):
        """Metadata tuple: ["metadata", {run_id: ...}] should signal stop."""
        content_payload = ["messages", [{"content": "ok", "type": "ai"}]]
        meta_payload = ["metadata", {"run_id": "xyz"}]
        lines = [
            f"data: {json.dumps(content_payload)}",
            f"data: {json.dumps(meta_payload)}",
        ]
        it = _make_iterator(lines)
        chunks = list(it)
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "ok"
        assert chunks[1].choices[0].finish_reason == "stop"


# Tests - Edge cases / backward-compatibility

class TestEdgeCases:
    """Mixed formats, empty data, and malformed lines."""

    def test_should_skip_empty_lines(self):
        """Blank lines and empty data should not crash."""
        lines = ["", "data: ", "", "data: not-json", ""]
        it = _make_iterator(lines)
        chunks = list(it)
        # Only the auto-generated final stop chunk
        assert len(chunks) == 1
        assert chunks[0].choices[0].finish_reason == "stop"

    def test_should_handle_dict_with_content_key(self):
        """Dict payload with a ``content`` key should produce a chunk even
        without an event header (heuristic fallback)."""
        lines = ['data: {"content": "fallback", "type": "ai"}']
        # This hits the heuristic path since 'content' key is present
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["fallback"]

    def test_should_handle_dict_with_messages_key(self):
        """Dict payload with a ``messages`` list (heuristic fallback)."""
        lines = [
            'data: {"messages": [{"content": "inner", "type": "ai"}]}',
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["inner"]

    def test_should_reset_event_type_on_blank_line(self):
        """After a blank line the event type must be reset so a subsequent
        data line without an event header doesn't inherit the old type."""
        lines = [
            "event: messages",
            "",  # reset
            'data: [{"content": "no-header", "type": "AIMessageChunk"}, {}]',
        ]
        # Without the event header the parser should still handle the list
        # via the headerless-list fallback
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["no-header"]

    def test_should_ignore_unknown_event_types(self):
        """Unknown event types should be silently skipped."""
        lines = [
            "event: custom_event",
            'data: {"foo": "bar"}',
            "",
            "event: messages",
            'data: [{"content": "valid", "type": "ai"}, {}]',
            "",
        ]
        contents = _collect_content(_make_iterator(lines))
        assert contents == ["valid"]

    def test_should_produce_final_stop_chunk(self):
        """When the stream ends without an explicit metadata event, a final
        stop chunk should still be emitted."""
        lines = [
            "event: messages",
            'data: [{"content": "hello", "type": "ai"}, {}]',
            "",
        ]
        it = _make_iterator(lines)
        chunks = list(it)
        # content chunk + auto final stop
        assert len(chunks) == 2
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_model_response_structure(self):
        """Verify that emitted chunks have the correct ModelResponseStream
        shape expected by the rest of LiteLLM."""
        lines = [
            "event: messages",
            'data: [{"content": "check", "type": "ai"}, {}]',
            "",
        ]
        it = _make_iterator(lines)
        chunk = next(it)
        assert isinstance(chunk, ModelResponseStream)
        assert chunk.model == MODEL
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.choices[0].delta.content == "check"
        assert chunk.choices[0].finish_reason is None

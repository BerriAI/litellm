"""
Unit tests for the fix: when a provider returns tool name AND arguments in
the same streaming chunk, AnthropicStreamWrapper must forward the
content_block_delta (carrying input_json_delta) instead of silently
discarding it.

Reproduces the bug described in:
  - GitHub issue with Vertex AI Gemini streaming: tool arguments are lost,
    resulting in `input: {}`.
"""

import json
from typing import List, Optional
from unittest.mock import MagicMock

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


def _make_chunk(
    *,
    tool_call_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[str] = None,
    text: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> MagicMock:
    """Build a minimal ModelResponseStream-like object."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()

    # Defaults
    delta.content = text if text is not None else ""
    delta.thinking_blocks = None
    delta.reasoning_content = None

    if tool_name is not None or tool_args is not None:
        tc = MagicMock()
        tc.id = tool_call_id or "call_abc123"
        func = MagicMock()
        func.name = tool_name
        func.arguments = tool_args
        tc.function = func
        delta.tool_calls = [tc]
        delta.content = None
    else:
        delta.tool_calls = None

    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _collect_events(wrapper: AnthropicStreamWrapper) -> List[dict]:
    """Drain all SSE events from the wrapper."""
    events: List[dict] = []
    for evt in wrapper:
        events.append(evt)
    return events


class TestToolArgsInSameChunk:
    """Verify that tool arguments delivered in the same chunk as the tool name
    are not silently dropped."""

    def test_tool_args_forwarded_when_name_and_args_in_same_chunk(self):
        """Core regression test:
        Provider sends ONE chunk with tool_name='Read' AND
        arguments='{"file_path":"C:\\\\test.md"}'.
        The stream must contain an input_json_delta event with partial_json
        containing 'file_path'.
        """
        # First chunk: plain text (the model "speaks" before calling a tool)
        text_chunk = _make_chunk(text="Let me read that file.")
        # Second chunk: tool name + arguments in one go (Gemini-style)
        tool_chunk = _make_chunk(
            tool_name="Read",
            tool_args='{"file_path": "C:\\\\test.md"}',
        )
        # Final chunk: finish
        finish_chunk = _make_chunk(finish_reason="tool_use")

        stream = iter([text_chunk, tool_chunk, finish_chunk])
        wrapper = AnthropicStreamWrapper(
            completion_stream=stream, model="gemini-3-flash-preview"
        )

        events = _collect_events(wrapper)
        event_types = [e.get("type") for e in events]

        # Must have: message_start, content_block_start (text),
        # content_block_delta (text), content_block_stop,
        # content_block_start (tool_use), content_block_delta (input_json_delta),
        # content_block_stop, message_delta, message_stop
        assert "message_start" in event_types

        # Find the tool_use content_block_start
        tool_starts = [
            e
            for e in events
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1, f"Expected 1 tool_use start, got {tool_starts}"

        # Find input_json_delta events
        input_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert (
            len(input_deltas) >= 1
        ), f"Expected at least 1 input_json_delta, got none. Events: {event_types}"

        # The partial_json must contain the actual arguments
        combined_json = "".join(
            d["delta"]["partial_json"]
            for d in input_deltas
            if d["delta"].get("partial_json")
        )
        assert combined_json, "partial_json is empty — tool arguments were dropped"
        parsed = json.loads(combined_json)
        assert "file_path" in parsed, f"file_path not in parsed args: {parsed}"

    def test_tool_args_split_across_chunks_still_works(self):
        """Baseline: when name and args arrive in separate chunks
        (e.g. Anthropic/Bedrock), everything works as before."""
        # Chunk 1: tool name only
        name_chunk = _make_chunk(tool_name="Read", tool_args=None)
        # Chunk 2: tool arguments only
        args_chunk = _make_chunk(tool_name=None, tool_args='{"file_path": "/tmp/x"}')
        # Final
        finish_chunk = _make_chunk(finish_reason="tool_use")

        stream = iter([name_chunk, args_chunk, finish_chunk])
        wrapper = AnthropicStreamWrapper(completion_stream=stream, model="test-model")

        events = _collect_events(wrapper)

        input_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(input_deltas) >= 1, "input_json_delta missing for split-chunk case"

        combined_json = "".join(
            d["delta"]["partial_json"]
            for d in input_deltas
            if d["delta"].get("partial_json")
        )
        parsed = json.loads(combined_json)
        assert "file_path" in parsed

    def test_text_then_tool_transition_preserves_text(self):
        """When switching from text to tool_use, the text content must also
        be preserved in the output."""
        text_chunk = _make_chunk(text="I will read the file.")
        tool_chunk = _make_chunk(tool_name="Read", tool_args='{"file_path": "/a/b"}')
        finish_chunk = _make_chunk(finish_reason="tool_use")

        stream = iter([text_chunk, tool_chunk, finish_chunk])
        wrapper = AnthropicStreamWrapper(completion_stream=stream, model="test-model")

        events = _collect_events(wrapper)

        text_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert len(text_deltas) >= 1, "Text delta was lost during transition"
        assert any(
            "read the file" in d["delta"].get("text", "") for d in text_deltas
        ), "Text content was not forwarded"

"""
Unit tests for AnthropicStreamWrapper - validates the content_block_start fix
for extended thinking. Tests three scenarios:
1. Plain text response (non-thinking) - block type should be "text"
2. Thinking response - block type should be "thinking", followed by "text"
3. Tool use response - block type should be "tool_use"

Expected SSE event sequence per Anthropic docs:
  Thinking:  message_start → content_block_start(thinking) → thinking_deltas → content_block_stop
             → content_block_start(text) → text_deltas → content_block_stop → message_delta → message_stop
  Text only: message_start → content_block_start(text) → text_deltas → content_block_stop → message_delta → message_stop
  Tool use:  message_start → content_block_start(tool_use) → input_json_deltas → content_block_stop → message_delta → message_stop
"""
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath("../../../../.."))
from unittest.mock import MagicMock
from litellm.types.utils import (
    ModelResponseStream,
    Delta,
    StreamingChoices,
    Function,
    ChatCompletionDeltaToolCall,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


def make_chunk(
    content=None,
    thinking_blocks=None,
    tool_calls=None,
    finish_reason=None,
):
    """Build a ModelResponseStream chunk."""
    delta = Delta(
        content=content if content is not None else "",
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
    )
    choice = StreamingChoices(finish_reason=finish_reason, delta=delta, index=0)
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.choices = [choice]
    return chunk


def make_finish_chunk(stop_reason="end_turn"):
    """Build a finish_reason chunk (signals end of stream)."""
    delta = Delta(content="")
    choice = StreamingChoices(finish_reason=stop_reason, delta=delta, index=0)
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.choices = [choice]
    return chunk


def collect_events(wrapper):
    """Drain all events from a sync AnthropicStreamWrapper."""
    events = []
    for event in wrapper:
        events.append(event)
    return events


async def collect_events_async(wrapper):
    """Drain all events from an async AnthropicStreamWrapper."""
    events = []
    async for event in wrapper:
        events.append(event)
    return events


def make_wrapper(chunks):
    """Create a sync AnthropicStreamWrapper from a list of chunks."""
    return AnthropicStreamWrapper(iter(chunks), model="test-model")


async def async_iter(chunks):
    """Convert a list to an async iterator."""
    for chunk in chunks:
        yield chunk


def make_async_wrapper(chunks):
    """Create an async AnthropicStreamWrapper from a list of chunks."""
    return AnthropicStreamWrapper(async_iter(chunks), model="test-model")


# =============================================================================
# Sync tests (__next__ path)
# =============================================================================


class TestNonThinkingTextStream:
    """Plain text response without thinking enabled."""

    def test_content_block_start_is_text(self):
        """First content_block_start must have type=text."""
        chunks = [
            make_chunk(content="Hello"),
            make_chunk(content=" world"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        cbs = next(e for e in events if e.get("type") == "content_block_start")
        assert cbs["content_block"]["type"] == "text"
        assert cbs["index"] == 0

    def test_full_event_sequence(self):
        """message_start → content_block_start(text) → delta(s) → content_block_stop → message_delta → message_stop."""
        chunks = [
            make_chunk(content="Hi"),
            make_chunk(content=" there"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))
        types = [e.get("type") for e in events]

        assert types[0] == "message_start"
        assert types[1] == "content_block_start"
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert types[-1] == "message_stop"

    def test_text_deltas_have_correct_type(self):
        """Text deltas should have type=text_delta."""
        chunks = [
            make_chunk(content="Hello"),
            make_chunk(content=" world"),
            make_chunk(content="!"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        deltas = [e for e in events if e.get("type") == "content_block_delta"]
        assert len(deltas) >= 1
        for d in deltas:
            assert d["delta"]["type"] == "text_delta"
            assert "text" in d["delta"]

    def test_multi_chunk_text(self):
        """Multiple text chunks produce multiple deltas, single content block."""
        chunks = [
            make_chunk(content="one"),
            make_chunk(content=" two"),
            make_chunk(content=" three"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        assert len(content_block_starts) == 1
        assert content_block_starts[0]["content_block"]["type"] == "text"


class TestThinkingStream:
    """Extended thinking response: thinking block followed by text block."""

    def test_first_content_block_start_is_thinking(self):
        """First content_block_start must have type=thinking."""
        thinking_block = [{"type": "thinking", "thinking": "Let me think..."}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="", thinking_blocks=[{"type": "thinking", "thinking": " more"}]),
            make_chunk(content="The answer"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        assert content_block_starts[0]["content_block"]["type"] == "thinking"
        assert content_block_starts[0]["index"] == 0

    def test_thinking_then_text_two_content_blocks(self):
        """Should produce two content_block_start events: thinking at index 0, text at index 1."""
        thinking_block = [{"type": "thinking", "thinking": "Thinking..."}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="Answer"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        assert len(content_block_starts) == 2

        assert content_block_starts[0]["content_block"]["type"] == "thinking"
        assert content_block_starts[0]["index"] == 0

        assert content_block_starts[1]["content_block"]["type"] == "text"
        assert content_block_starts[1]["index"] == 1

    def test_thinking_deltas_have_correct_type(self):
        """Thinking deltas should have type=thinking_delta."""
        thinking_block = [{"type": "thinking", "thinking": "reasoning here"}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="", thinking_blocks=[{"type": "thinking", "thinking": " step 2"}]),
            make_chunk(content="Result part 1"),
            make_chunk(content=" part 2"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))

        deltas = [e for e in events if e.get("type") == "content_block_delta"]
        thinking_deltas = [d for d in deltas if d["delta"]["type"] == "thinking_delta"]
        text_deltas = [d for d in deltas if d["delta"]["type"] == "text_delta"]

        assert len(thinking_deltas) >= 1, f"Expected thinking deltas, got: {deltas}"
        assert len(text_deltas) >= 1, f"Expected text deltas, got: {deltas}"

    def test_content_block_stop_between_thinking_and_text(self):
        """There must be a content_block_stop between the thinking and text blocks."""
        thinking_block = [{"type": "thinking", "thinking": "..."}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="Answer"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))
        types = [e.get("type") for e in events]

        cbs_indices = [i for i, t in enumerate(types) if t == "content_block_start"]
        stop_indices = [i for i, t in enumerate(types) if t == "content_block_stop"]

        assert len(cbs_indices) == 2, f"Expected 2 content_block_starts, got types: {types}"
        assert len(stop_indices) >= 1

        # First stop must come between the two starts
        assert stop_indices[0] > cbs_indices[0]
        assert stop_indices[0] < cbs_indices[1]

    def test_full_thinking_event_sequence(self):
        """Verify full sequence matches Anthropic spec."""
        thinking_block = [{"type": "thinking", "thinking": "Let me reason"}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="", thinking_blocks=[{"type": "thinking", "thinking": " step 2"}]),
            make_chunk(content="Final answer"),
            make_finish_chunk(),
        ]
        events = collect_events(make_wrapper(chunks))
        types = [e.get("type") for e in events]

        assert types[0] == "message_start"
        assert types[1] == "content_block_start"
        assert events[1]["content_block"]["type"] == "thinking"
        assert types[-1] == "message_stop"
        assert "message_delta" in types


class TestToolUseStream:
    """Tool use response."""

    def test_content_block_start_is_tool_use(self):
        """First content_block_start must have type=tool_use with name and id."""
        tool_call = ChatCompletionDeltaToolCall(
            id="call_abc123",
            function=Function(name="get_weather", arguments=""),
            type="function",
            index=0,
        )
        chunks = [
            make_chunk(tool_calls=[tool_call]),
            make_chunk(tool_calls=[ChatCompletionDeltaToolCall(
                id=None,
                function=Function(name=None, arguments='{"city": "Sydney"}'),
                type="function",
                index=0,
            )]),
            make_finish_chunk(stop_reason="tool_use"),
        ]
        events = collect_events(make_wrapper(chunks))

        cbs = next(e for e in events if e.get("type") == "content_block_start")
        assert cbs["content_block"]["type"] == "tool_use"
        assert cbs["content_block"]["name"] == "get_weather"
        assert cbs["index"] == 0

    def test_tool_use_deltas_have_correct_type(self):
        """Tool use deltas should have type=input_json_delta."""
        tool_call = ChatCompletionDeltaToolCall(
            id="call_abc123",
            function=Function(name="get_weather", arguments=""),
            type="function",
            index=0,
        )
        chunks = [
            make_chunk(tool_calls=[tool_call]),
            make_chunk(tool_calls=[ChatCompletionDeltaToolCall(
                id=None,
                function=Function(name=None, arguments='{"city": "SF"}'),
                type="function",
                index=0,
            )]),
            make_finish_chunk(stop_reason="tool_use"),
        ]
        events = collect_events(make_wrapper(chunks))

        deltas = [e for e in events if e.get("type") == "content_block_delta"]
        for d in deltas:
            assert d["delta"]["type"] == "input_json_delta"

    def test_text_then_tool_use(self):
        """Text followed by tool call: two content blocks with correct types."""
        tool_call = ChatCompletionDeltaToolCall(
            id="call_xyz",
            function=Function(name="search", arguments=""),
            type="function",
            index=0,
        )
        chunks = [
            make_chunk(content="Let me search that for you."),
            make_chunk(tool_calls=[tool_call]),
            make_chunk(tool_calls=[ChatCompletionDeltaToolCall(
                id=None,
                function=Function(name=None, arguments='{"q": "test"}'),
                type="function",
                index=0,
            )]),
            make_finish_chunk(stop_reason="tool_use"),
        ]
        events = collect_events(make_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        block_types = [e["content_block"]["type"] for e in content_block_starts]

        assert block_types[0] == "text"
        assert "tool_use" in block_types


# =============================================================================
# Async tests (__anext__ path)
# =============================================================================


@pytest.mark.asyncio
class TestAsyncNonThinkingTextStream:
    """Async: Plain text response without thinking enabled."""

    async def test_content_block_start_is_text(self):
        """First content_block_start must have type=text."""
        chunks = [
            make_chunk(content="Hello"),
            make_chunk(content=" world"),
            make_finish_chunk(),
        ]
        events = await collect_events_async(make_async_wrapper(chunks))

        cbs = next(e for e in events if e.get("type") == "content_block_start")
        assert cbs["content_block"]["type"] == "text"
        assert cbs["index"] == 0

    async def test_full_event_sequence(self):
        """message_start → content_block_start(text) → delta(s) → content_block_stop → message_delta → message_stop."""
        chunks = [
            make_chunk(content="Hi"),
            make_chunk(content=" there"),
            make_finish_chunk(),
        ]
        events = await collect_events_async(make_async_wrapper(chunks))
        types = [e.get("type") for e in events]

        assert types[0] == "message_start"
        assert types[1] == "content_block_start"
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert types[-1] == "message_stop"


@pytest.mark.asyncio
class TestAsyncThinkingStream:
    """Async: Extended thinking response."""

    async def test_first_content_block_start_is_thinking(self):
        """First content_block_start must have type=thinking."""
        thinking_block = [{"type": "thinking", "thinking": "Let me think..."}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="", thinking_blocks=[{"type": "thinking", "thinking": " more"}]),
            make_chunk(content="The answer"),
            make_finish_chunk(),
        ]
        events = await collect_events_async(make_async_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        assert content_block_starts[0]["content_block"]["type"] == "thinking"
        assert content_block_starts[0]["index"] == 0

    async def test_thinking_then_text_two_content_blocks(self):
        """Should produce two content_block_start events: thinking at index 0, text at index 1."""
        thinking_block = [{"type": "thinking", "thinking": "Thinking..."}]
        chunks = [
            make_chunk(content="", thinking_blocks=thinking_block),
            make_chunk(content="Answer"),
            make_finish_chunk(),
        ]
        events = await collect_events_async(make_async_wrapper(chunks))

        content_block_starts = [e for e in events if e.get("type") == "content_block_start"]
        assert len(content_block_starts) == 2

        assert content_block_starts[0]["content_block"]["type"] == "thinking"
        assert content_block_starts[0]["index"] == 0

        assert content_block_starts[1]["content_block"]["type"] == "text"
        assert content_block_starts[1]["index"] == 1


@pytest.mark.asyncio
class TestAsyncToolUseStream:
    """Async: Tool use response."""

    async def test_content_block_start_is_tool_use(self):
        """First content_block_start must have type=tool_use with name and id."""
        tool_call = ChatCompletionDeltaToolCall(
            id="call_abc123",
            function=Function(name="get_weather", arguments=""),
            type="function",
            index=0,
        )
        chunks = [
            make_chunk(tool_calls=[tool_call]),
            make_chunk(tool_calls=[ChatCompletionDeltaToolCall(
                id=None,
                function=Function(name=None, arguments='{"city": "Sydney"}'),
                type="function",
                index=0,
            )]),
            make_finish_chunk(stop_reason="tool_use"),
        ]
        events = await collect_events_async(make_async_wrapper(chunks))

        cbs = next(e for e in events if e.get("type") == "content_block_start")
        assert cbs["content_block"]["type"] == "tool_use"
        assert cbs["content_block"]["name"] == "get_weather"
        assert cbs["index"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

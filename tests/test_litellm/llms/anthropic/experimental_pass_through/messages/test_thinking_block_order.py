"""
Tests for AnthropicStreamWrapper thinking block ordering.

Verifies that when extended thinking is enabled, the first content_block_start
event uses the correct block type ("thinking") rather than hardcoding "text".
This is required by the Anthropic protocol: thinking blocks must appear before
text blocks.

Fixes: https://github.com/BerriAI/litellm/issues/21128
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", ".."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponse, StreamingChoices


class MockThinkingStream:
    """Simulates a Bedrock Converse stream with extended thinking enabled.

    The stream emits thinking blocks first (reasoning), followed by text blocks
    (the final answer), which mirrors the real Anthropic/Bedrock behavior.
    """

    def __init__(self):
        self.responses = [
            # First chunk: thinking block
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(
                            content=None,
                            thinking_blocks=[
                                {"type": "thinking", "thinking": "Let me think...", "signature": ""}
                            ],
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            # Second chunk: more thinking
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(
                            content=None,
                            thinking_blocks=[
                                {"type": "thinking", "thinking": " about this problem.", "signature": ""}
                            ],
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            # Third chunk: thinking signature (end of thinking)
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(
                            content=None,
                            thinking_blocks=[
                                {"type": "thinking", "thinking": "", "signature": "abc123sig"}
                            ],
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            # Fourth chunk: text content (the answer)
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content="The answer is 4."),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            # Fifth chunk: finish
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""),
                        index=0,
                        finish_reason="end_turn",
                    )
                ],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


class AsyncMockThinkingStream:
    """Async version of MockThinkingStream."""

    def __init__(self):
        self._sync = MockThinkingStream()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._sync)
        except StopIteration:
            raise StopAsyncIteration


class MockTextOnlyStream:
    """Simulates a normal text-only stream (no thinking blocks)."""

    def __init__(self):
        self.responses = [
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" World"),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""),
                        index=0,
                        finish_reason="end_turn",
                    )
                ],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


class AsyncMockTextOnlyStream:
    """Async version of MockTextOnlyStream."""

    def __init__(self):
        self._sync = MockTextOnlyStream()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._sync)
        except StopIteration:
            raise StopAsyncIteration


def test_thinking_block_is_first_content_block_sync():
    """When thinking is enabled, the first content_block_start must be type='thinking'."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    events = list(wrapper)

    # Extract content_block_start events
    block_starts = [e for e in events if e.get("type") == "content_block_start"]

    assert len(block_starts) >= 2, f"Expected at least 2 content_block_start events, got {len(block_starts)}"

    # First content block must be "thinking" (index 0)
    first_block = block_starts[0]
    assert first_block["index"] == 0
    assert first_block["content_block"]["type"] == "thinking", (
        f"First content block should be 'thinking', got '{first_block['content_block']['type']}'"
    )

    # Second content block must be "text" (index 1)
    second_block = block_starts[1]
    assert second_block["index"] == 1
    assert second_block["content_block"]["type"] == "text", (
        f"Second content block should be 'text', got '{second_block['content_block']['type']}'"
    )


def test_text_only_first_block_remains_text_sync():
    """For non-thinking responses, the first content_block_start remains type='text'."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockTextOnlyStream(), model="claude-sonnet-4-5-20250514"
    )

    events = list(wrapper)

    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    assert len(block_starts) >= 1

    first_block = block_starts[0]
    assert first_block["index"] == 0
    assert first_block["content_block"]["type"] == "text"


def test_thinking_stream_event_sequence_sync():
    """Verify the full event sequence for a thinking stream follows Anthropic protocol."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    events = list(wrapper)
    event_types = [e.get("type") for e in events]

    # Must start with message_start
    assert event_types[0] == "message_start"

    # Must have content_block_start as second event
    assert event_types[1] == "content_block_start"

    # The first content_block_start must be followed by thinking deltas
    first_block_start = events[1]
    assert first_block_start["content_block"]["type"] == "thinking"

    # Must end with message_stop
    assert event_types[-1] == "message_stop"


def test_thinking_sse_format_sync():
    """Verify SSE output has correct event types for thinking blocks."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    sse_chunks = []
    for raw in wrapper.anthropic_sse_wrapper():
        sse_chunks.append(raw.decode("utf-8"))

    # Parse SSE events
    events = []
    for sse in sse_chunks:
        lines = sse.strip().split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type and data:
            events.append({"event": event_type, "data": data})

    # First event: message_start
    assert events[0]["event"] == "message_start"

    # Second event: content_block_start with thinking type
    assert events[1]["event"] == "content_block_start"
    assert events[1]["data"]["content_block"]["type"] == "thinking"


@pytest.mark.asyncio
async def test_thinking_block_is_first_content_block_async():
    """Async: first content_block_start must be type='thinking' when thinking is enabled."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    events = []
    async for event in wrapper:
        events.append(event)

    block_starts = [e for e in events if e.get("type") == "content_block_start"]

    assert len(block_starts) >= 2

    first_block = block_starts[0]
    assert first_block["index"] == 0
    assert first_block["content_block"]["type"] == "thinking"

    second_block = block_starts[1]
    assert second_block["index"] == 1
    assert second_block["content_block"]["type"] == "text"


@pytest.mark.asyncio
async def test_text_only_first_block_remains_text_async():
    """Async: for non-thinking responses, first content_block_start remains type='text'."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockTextOnlyStream(), model="claude-sonnet-4-5-20250514"
    )

    events = []
    async for event in wrapper:
        events.append(event)

    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    assert len(block_starts) >= 1

    first_block = block_starts[0]
    assert first_block["index"] == 0
    assert first_block["content_block"]["type"] == "text"


@pytest.mark.asyncio
async def test_thinking_stream_no_content_loss_async():
    """Async: all thinking deltas and text deltas must be present (no dropped chunks)."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    events = []
    async for event in wrapper:
        events.append(event)

    # Collect all delta events
    deltas = [e for e in events if e.get("type") == "content_block_delta"]

    # We should have thinking deltas and text deltas
    thinking_deltas = [
        d for d in deltas
        if d.get("delta", {}).get("type") in ("thinking_delta", "signature_delta")
    ]
    text_deltas = [
        d for d in deltas if d.get("delta", {}).get("type") == "text_delta"
    ]

    assert len(thinking_deltas) >= 1, "Should have at least one thinking/signature delta"
    assert len(text_deltas) >= 1, "Should have at least one text delta"


def test_thinking_stream_no_content_loss_sync():
    """Sync: all thinking deltas and text deltas must be present (no dropped chunks)."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    events = list(wrapper)

    deltas = [e for e in events if e.get("type") == "content_block_delta"]

    thinking_deltas = [
        d for d in deltas
        if d.get("delta", {}).get("type") in ("thinking_delta", "signature_delta")
    ]
    text_deltas = [
        d for d in deltas if d.get("delta", {}).get("type") == "text_delta"
    ]

    assert len(thinking_deltas) >= 1, "Should have at least one thinking/signature delta"
    assert len(text_deltas) >= 1, "Should have at least one text delta"


@pytest.mark.asyncio
async def test_thinking_sse_format_async():
    """Async: verify SSE output has correct event types for thinking blocks."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockThinkingStream(), model="claude-sonnet-4-5-20250514"
    )

    sse_chunks = []
    async for raw in wrapper.async_anthropic_sse_wrapper():
        sse_chunks.append(raw.decode("utf-8"))

    events = []
    for sse in sse_chunks:
        lines = sse.strip().split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type and data:
            events.append({"event": event_type, "data": data})

    assert events[0]["event"] == "message_start"
    assert events[1]["event"] == "content_block_start"
    assert events[1]["data"]["content_block"]["type"] == "thinking"

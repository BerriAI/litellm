"""
Tests for streaming_iterator.py fixes:

Fix 2 – Peek at first chunk to determine correct initial content_block_start type.
         Models that return reasoning_content (e.g. GLM-5 via Vertex AI) start
         the stream with a thinking block, not a text block.

Fix 3 – Queue the processed_chunk (first delta of a new block) when a content
         block transition occurs, so the first token is not silently dropped.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
)


# ---------------------------------------------------------------------------
# Mock streams
# ---------------------------------------------------------------------------


class MockSyncStream:
    """Synchronous mock completion stream yielding a fixed list of chunks."""

    def __init__(self, chunks: list[ModelResponseStream]):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


class MockAsyncStream:
    """Asynchronous mock completion stream yielding a fixed list of chunks."""

    def __init__(self, chunks: list[ModelResponseStream]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thinking_chunk(text: str) -> ModelResponseStream:
    """Create a streaming chunk with reasoning_content (no thinking_blocks)."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(
                    reasoning_content=text,
                    content="",
                    role="assistant",
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _make_text_chunk(text: str) -> ModelResponseStream:
    """Create a streaming chunk with text content."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(
                    content=text,
                    role="assistant",
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _make_stop_chunk() -> ModelResponseStream:
    """Create a streaming chunk signalling end of generation."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content=""),
                index=0,
                finish_reason="stop",
            )
        ],
    )


def _collect_all_events(wrapper) -> list[dict]:
    """Collect all events from a sync AnthropicStreamWrapper."""
    events = []
    for raw in wrapper:
        events.append(raw)
    return events


async def _collect_all_events_async(wrapper) -> list[dict]:
    """Collect all events from an async AnthropicStreamWrapper."""
    events = []
    async for raw in wrapper:
        events.append(raw)
    return events


# ---------------------------------------------------------------------------
# Fix 2 – Initial content_block_start reflects first chunk type
# ---------------------------------------------------------------------------


class TestInitialBlockTypePeek:
    """
    When the first chunk from the upstream model contains reasoning_content,
    the initial content_block_start must have type "thinking", not "text".
    """

    def test_sync_thinking_first_chunk(self):
        chunks = [
            _make_thinking_chunk("Let me think..."),
            _make_thinking_chunk(" about this."),
            _make_text_chunk("The answer is 42."),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockSyncStream(chunks), model="glm-5"
        )
        events = _collect_all_events(wrapper)

        assert events[0]["type"] == "message_start"

        content_block_start = events[1]
        assert content_block_start["type"] == "content_block_start"
        assert content_block_start["content_block"]["type"] == "thinking"

        first_delta = events[2]
        assert first_delta["type"] == "content_block_delta"
        assert first_delta["delta"]["type"] == "thinking_delta"

    def test_sync_text_first_chunk(self):
        """Text-first streams should still emit type 'text' (no regression)."""
        chunks = [
            _make_text_chunk("Hello"),
            _make_text_chunk(" world"),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockSyncStream(chunks), model="gpt-4o"
        )
        events = _collect_all_events(wrapper)

        content_block_start = events[1]
        assert content_block_start["type"] == "content_block_start"
        assert content_block_start["content_block"]["type"] == "text"

    @pytest.mark.asyncio
    async def test_async_thinking_first_chunk(self):
        chunks = [
            _make_thinking_chunk("Let me think..."),
            _make_thinking_chunk(" about this."),
            _make_text_chunk("The answer is 42."),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockAsyncStream(chunks), model="glm-5"
        )
        events = await _collect_all_events_async(wrapper)

        assert events[0]["type"] == "message_start"

        content_block_start = events[1]
        assert content_block_start["type"] == "content_block_start"
        assert content_block_start["content_block"]["type"] == "thinking"

        first_delta = events[2]
        assert first_delta["type"] == "content_block_delta"
        assert first_delta["delta"]["type"] == "thinking_delta"

    @pytest.mark.asyncio
    async def test_async_text_first_chunk(self):
        chunks = [
            _make_text_chunk("Hello"),
            _make_text_chunk(" world"),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockAsyncStream(chunks), model="gpt-4o"
        )
        events = await _collect_all_events_async(wrapper)

        content_block_start = events[1]
        assert content_block_start["type"] == "content_block_start"
        assert content_block_start["content_block"]["type"] == "text"


# ---------------------------------------------------------------------------
# Fix 3 – Block transition queues the trigger chunk
# ---------------------------------------------------------------------------


class TestBlockTransitionIncludesFirstDelta:
    """
    When the stream transitions from one block type to another (e.g. thinking
    → text), the processed chunk that triggered the transition must be queued
    and eventually yielded.  Without Fix 3 the first token of the new block
    would be silently dropped.
    """

    def test_sync_thinking_to_text_no_token_drop(self):
        chunks = [
            _make_thinking_chunk("Reasoning step."),
            _make_text_chunk("Answer text."),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockSyncStream(chunks), model="glm-5"
        )
        events = _collect_all_events(wrapper)

        text_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert len(text_deltas) >= 1, (
            "The first text delta after a thinking→text transition must not be "
            "dropped.  Got text_delta events: " + repr(text_deltas)
        )
        assert text_deltas[0]["delta"]["text"] == "Answer text."

    @pytest.mark.asyncio
    async def test_async_thinking_to_text_no_token_drop(self):
        chunks = [
            _make_thinking_chunk("Reasoning step."),
            _make_text_chunk("Answer text."),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockAsyncStream(chunks), model="glm-5"
        )
        events = await _collect_all_events_async(wrapper)

        text_deltas = [
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert len(text_deltas) >= 1, (
            "The first text delta after a thinking→text transition must not be "
            "dropped.  Got text_delta events: " + repr(text_deltas)
        )
        assert text_deltas[0]["delta"]["text"] == "Answer text."

    def test_sync_event_sequence_is_valid(self):
        """
        The full event sequence for a thinking→text stream should follow the
        Anthropic SSE spec:
          message_start →
          content_block_start (thinking) →
          content_block_delta (thinking_delta) →
          content_block_stop →
          content_block_start (text) →
          content_block_delta (text_delta) →
          content_block_stop →
          message_delta →
          message_stop
        """
        chunks = [
            _make_thinking_chunk("Think."),
            _make_text_chunk("Answer."),
            _make_stop_chunk(),
        ]
        wrapper = AnthropicStreamWrapper(
            completion_stream=MockSyncStream(chunks), model="glm-5"
        )
        events = _collect_all_events(wrapper)

        types = [e["type"] for e in events]

        assert types[0] == "message_start"
        assert types[1] == "content_block_start"
        assert events[1]["content_block"]["type"] == "thinking"
        assert "content_block_delta" in types
        idx_first_stop = types.index("content_block_stop")
        assert idx_first_stop > 1
        idx_second_start = types.index("content_block_start", idx_first_stop)
        assert events[idx_second_start]["content_block"]["type"] == "text"
        text_delta_idx = types.index("content_block_delta", idx_second_start)
        assert events[text_delta_idx]["delta"]["type"] == "text_delta"

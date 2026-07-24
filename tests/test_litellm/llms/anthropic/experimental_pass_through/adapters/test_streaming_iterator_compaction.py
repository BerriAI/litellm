"""Compaction block SSE events from AnthropicStreamWrapper (compact_20260112 polyfill)."""

import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices, Usage


def _make_text_chunk(
    text: str,
    finish_reason: str = None,
    usage: "Usage | None" = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=Delta(
                content=text, role="assistant" if text else None, tool_calls=None
            ),
            logprobs=None,
        )
    ]
    chunk.usage = usage
    chunk._hidden_params = {}
    return chunk


async def _collect_events_async(wrapper: AnthropicStreamWrapper) -> List[dict]:
    events = []
    async for event in wrapper:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_stream_emits_compaction_block_before_text():
    """Polyfill compaction_block must surface as compaction SSE events at index 0."""

    async def mock_stream():
        yield _make_text_chunk("Hi")
        yield _make_text_chunk(
            "",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    compaction_block = {
        "type": "compaction",
        "content": "Summary of prior conversation turns.",
    }
    iterations_usage = [
        {"type": "compaction", "input_tokens": 100, "output_tokens": 50},
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="claude-sonnet-4-6",
        compaction_block=compaction_block,
        iterations_usage=iterations_usage,
        applied_edits=[{"type": "compact_20260112"}],
    )

    events = await _collect_events_async(wrapper)

    compaction_start = next(
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "compaction"
    )
    assert compaction_start["index"] == 0

    compaction_delta = next(
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "compaction_delta"
    )
    assert compaction_delta["index"] == 0
    assert (
        compaction_delta["delta"]["content"] == "Summary of prior conversation turns."
    )

    compaction_stop = next(
        e
        for e in events
        if e.get("type") == "content_block_stop" and e.get("index") == 0
    )
    assert compaction_stop is not None

    text_start = next(
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "text"
    )
    assert text_start["index"] == 1

    message_delta = next(e for e in events if e.get("type") == "message_delta")
    iterations = message_delta.get("usage", {}).get("iterations")
    assert iterations is not None
    assert iterations[0]["type"] == "compaction"
    assert iterations[1]["type"] == "message"
    assert iterations[1]["input_tokens"] == 10
    assert iterations[1]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_omits_message_iteration_when_no_usage_chunk():
    """When provider sends finish_reason without usage, the held message_delta
    carries placeholder zeros — we must not emit a misleading zero-token
    ``message`` iteration entry."""

    async def mock_stream():
        yield _make_text_chunk("Hi")
        yield _make_text_chunk("", finish_reason="stop")

    iterations_usage = [
        {"type": "compaction", "input_tokens": 100, "output_tokens": 50},
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="claude-sonnet-4-6",
        iterations_usage=iterations_usage,
    )

    events = await _collect_events_async(wrapper)
    message_delta = next(e for e in events if e.get("type") == "message_delta")
    iterations = message_delta.get("usage", {}).get("iterations")
    assert iterations is not None
    assert len(iterations) == 1
    assert iterations[0]["type"] == "compaction"


@pytest.mark.asyncio
async def test_stream_omits_context_management_when_no_compaction_applied():
    """applied_edits without a compaction block must not emit context_management."""

    async def mock_stream():
        yield _make_text_chunk("Hello")
        yield _make_text_chunk("", finish_reason="stop")

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="claude-sonnet-4-6",
        applied_edits=None,
    )

    events = await _collect_events_async(wrapper)
    message_deltas = [e for e in events if e.get("type") == "message_delta"]
    assert message_deltas
    assert "context_management" not in message_deltas[-1]


@pytest.mark.asyncio
async def test_stream_without_compaction_block_unchanged():
    """No compaction_block means no compaction SSE events."""

    async def mock_stream():
        yield _make_text_chunk("Hello")
        yield _make_text_chunk("", finish_reason="stop")

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="claude-sonnet-4-6",
    )

    events = await _collect_events_async(wrapper)

    assert not any(
        e.get("content_block", {}).get("type") == "compaction"
        for e in events
        if e.get("type") == "content_block_start"
    )
    text_start = next(
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "text"
    )
    assert text_start["index"] == 0

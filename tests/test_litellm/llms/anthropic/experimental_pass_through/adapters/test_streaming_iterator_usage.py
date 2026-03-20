"""
Tests for AnthropicStreamWrapper usage and cache_creation fields in streaming API.

Covers the changes in:
- litellm/types/llms/anthropic.py: CacheCreationDelta, UsageDelta.cache_creation
- litellm/llms/anthropic/.../streaming_iterator.py: _create_initial_usage_delta(),
  usage merging with cache tokens, last_usage tracking for message_stop
"""

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAsyncStream:
    """Asynchronous mock completion stream."""

    def __init__(self, responses: List[ModelResponseStream]):
        self.responses = responses
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        resp = self.responses[self.index]
        self.index += 1
        return resp


async def _collect_async_chunks(wrapper: AnthropicStreamWrapper) -> List[Dict[str, Any]]:
    chunks = []
    async for chunk in wrapper:
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Async streaming: message_start contains initial usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_message_start_has_initial_usage():
    """Async: message_start should carry initial usage with cache fields."""
    responses = [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="Hello"),
                    index=0,
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content=""),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockAsyncStream(responses),
        model="claude-sonnet-4-5",
    )
    chunks = await _collect_async_chunks(wrapper)

    assert chunks[0]["type"] == "message_start"
    msg_usage = chunks[0]["message"]["usage"]
    assert msg_usage["cache_creation_input_tokens"] == 0
    assert msg_usage["cache_read_input_tokens"] == 0
    assert "cache_creation" in msg_usage


# ---------------------------------------------------------------------------
# Async streaming: message_stop carries last_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_message_stop_carries_usage():
    """Async: message_stop should include the last_usage."""
    responses = [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="Test"),
                    index=0,
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content=""),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockAsyncStream(responses),
        model="claude-sonnet-4-5",
    )
    chunks = await _collect_async_chunks(wrapper)

    message_stops = [c for c in chunks if c.get("type") == "message_stop"]
    assert len(message_stops) == 1
    assert "usage" in message_stops[0]

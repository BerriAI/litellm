"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import asyncio
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockSSEStream:
    """Async iterator that yields mock SSE events in order."""

    def __init__(self, events: List[Any]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def _event(event_type: str, **kwargs) -> Dict[str, Any]:
    """Build a dict-based SSE event."""
    return {"type": event_type, **kwargs}


def _reasoning_item(item_id: str = "rs_001") -> Dict[str, Any]:
    """Build a minimal reasoning output_item.added event."""
    return _event(
        "response.output_item.added",
        item={"type": "reasoning", "id": item_id},
    )


def _summary_part_added() -> Dict[str, Any]:
    return _event("response.reasoning_summary_part.added")


def _summary_text_delta(delta: str) -> Dict[str, Any]:
    return _event("response.reasoning_summary_text.delta", delta=delta)


def _summary_part_done() -> Dict[str, Any]:
    return _event("response.reasoning_summary_part.done")


def _response_created() -> Dict[str, Any]:
    return _event("response.created")


def _response_completed(
    status: str = "completed",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> Dict[str, Any]:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    usage.input_tokens_details = None
    usage.output_tokens_details = None

    response_obj = MagicMock()
    response_obj.status = status
    response_obj.usage = usage
    response_obj.output = []
    return _event("response.completed", response=response_obj)


async def _collect_all(wrapper: AnthropicResponsesStreamWrapper) -> List[Dict[str, Any]]:
    """Drain all chunks from the wrapper."""
    chunks: List[Dict[str, Any]] = []
    async for chunk in wrapper:
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Test: each reasoning summary part produces its own content block cycle
# ---------------------------------------------------------------------------


class TestPerSummaryThinkingBlocks:
    """Commit cf2024f: each reasoning summary part gets its own
    content_block_start / content_block_delta / content_block_stop cycle."""

    @pytest.mark.asyncio
    async def test_two_summaries_produce_two_thinking_blocks(self):
        """Two summary parts yield two independent thinking block cycles."""
        events = [
            _response_created(),
            # reasoning item added (no content_block_start emitted here)
            _reasoning_item(),
            # First summary part
            _summary_part_added(),
            _summary_text_delta("**Step 1**"),
            _summary_text_delta("\nAnalyze the problem."),
            _summary_part_done(),
            # Second summary part
            _summary_part_added(),
            _summary_text_delta("**Step 2**"),
            _summary_text_delta("\nFormulate answer."),
            _summary_part_done(),
            _response_completed(),
        ]

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        # Extract thinking-related chunks
        block_starts = [c for c in chunks if c["type"] == "content_block_start"]
        block_deltas = [c for c in chunks if c["type"] == "content_block_delta"]
        block_stops = [c for c in chunks if c["type"] == "content_block_stop"]

        # Two thinking block starts
        assert len(block_starts) == 2
        for bs in block_starts:
            assert bs["content_block"]["type"] == "thinking"

        # Each start has a different index
        assert block_starts[0]["index"] != block_starts[1]["index"]

        # Two block stops, matching the start indices
        assert len(block_stops) == 2
        assert block_stops[0]["index"] == block_starts[0]["index"]
        assert block_stops[1]["index"] == block_starts[1]["index"]

        # Four thinking deltas total (2 per summary)
        thinking_deltas = [
            d for d in block_deltas if d["delta"]["type"] == "thinking_delta"
        ]
        assert len(thinking_deltas) == 4

    @pytest.mark.asyncio
    async def test_single_summary_one_block_cycle(self):
        """A single summary part produces exactly one thinking block cycle."""
        events = [
            _response_created(),
            _reasoning_item(),
            _summary_part_added(),
            _summary_text_delta("Reasoning content."),
            _summary_part_done(),
            _response_completed(),
        ]

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        block_starts = [c for c in chunks if c["type"] == "content_block_start"]
        block_stops = [c for c in chunks if c["type"] == "content_block_stop"]

        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert len(block_stops) == 1
        assert block_stops[0]["index"] == block_starts[0]["index"]

    @pytest.mark.asyncio
    async def test_reasoning_item_added_does_not_emit_block_start(self):
        """response.output_item.added with type=reasoning must NOT emit
        content_block_start -- that is deferred to part.added."""
        events = [
            _response_created(),
            _reasoning_item(),
            _response_completed(),
        ]

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        block_starts = [c for c in chunks if c["type"] == "content_block_start"]
        assert len(block_starts) == 0

    @pytest.mark.asyncio
    async def test_reasoning_item_done_does_not_emit_block_stop(self):
        """response.output_item.done with type=reasoning must NOT emit
        content_block_stop -- individual part.done events handle that."""
        events = [
            _response_created(),
            _reasoning_item("rs_002"),
            _summary_part_added(),
            _summary_text_delta("Thinking..."),
            _summary_part_done(),
            # reasoning output_item.done
            _event(
                "response.output_item.done",
                item={"type": "reasoning", "id": "rs_002"},
            ),
            _response_completed(),
        ]

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        block_stops = [c for c in chunks if c["type"] == "content_block_stop"]
        # Only 1 stop from part.done, not a second from item.done
        assert len(block_stops) == 1

    @pytest.mark.asyncio
    async def test_three_summaries_indices_increment(self):
        """Block indices increase monotonically across multiple summary parts."""
        events = [
            _response_created(),
            _reasoning_item(),
        ]
        # Add 3 summary parts
        for i in range(3):
            events.extend([
                _summary_part_added(),
                _summary_text_delta(f"Summary {i}"),
                _summary_part_done(),
            ])
        events.append(_response_completed())

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        block_starts = [c for c in chunks if c["type"] == "content_block_start"]
        indices = [bs["index"] for bs in block_starts]
        assert indices == [0, 1, 2]
        assert all(bs["content_block"]["type"] == "thinking" for bs in block_starts)

    @pytest.mark.asyncio
    async def test_reasoning_then_text_output_indices_correct(self):
        """After reasoning summary blocks, a text output message gets the next index."""
        events = [
            _response_created(),
            _reasoning_item(),
            _summary_part_added(),
            _summary_text_delta("Thought."),
            _summary_part_done(),
            _summary_part_added(),
            _summary_text_delta("More thought."),
            _summary_part_done(),
            # Now a text message output item
            _event(
                "response.output_item.added",
                item={"type": "message", "id": "msg_001"},
            ),
            _event(
                "response.output_text.delta",
                item_id="msg_001",
                delta="Hello!",
            ),
            _event(
                "response.output_item.done",
                item={"type": "message", "id": "msg_001"},
            ),
            _response_completed(),
        ]

        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        block_starts = [c for c in chunks if c["type"] == "content_block_start"]
        # 2 thinking + 1 text = 3 block starts
        assert len(block_starts) == 3
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert block_starts[0]["index"] == 0
        assert block_starts[1]["content_block"]["type"] == "thinking"
        assert block_starts[1]["index"] == 1
        assert block_starts[2]["content_block"]["type"] == "text"
        assert block_starts[2]["index"] == 2


# ---------------------------------------------------------------------------
# Test: deduplicate message_start
# ---------------------------------------------------------------------------


class TestDeduplicateMessageStart:
    """Commit f4bcf05: when response.created fires and _sent_message_start
    is already True, skip the duplicate message_start."""

    @pytest.mark.asyncio
    async def test_single_response_created_emits_one_message_start(self):
        """Normal case: one response.created -> one message_start."""
        events = [
            _response_created(),
            _response_completed(),
        ]
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1

    @pytest.mark.asyncio
    async def test_duplicate_response_created_emits_one_message_start(self):
        """Two response.created events -> still only one message_start emitted."""
        events = [
            _response_created(),
            _response_created(),  # duplicate
            _response_completed(),
        ]
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1

    @pytest.mark.asyncio
    async def test_fallback_message_start_then_response_created(self):
        """Fallback message_start (from __anext__) followed by response.created
        should not produce a second message_start."""
        # If response.created never fires first, __anext__ fallback emits it.
        # Then if response.created arrives later, it should be skipped.
        events = [
            # First event is NOT response.created, so __anext__ emits fallback
            _reasoning_item(),
            _summary_part_added(),
            _summary_text_delta("think"),
            _summary_part_done(),
            # Now response.created arrives late
            _response_created(),
            _response_completed(),
        ]
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1

    @pytest.mark.asyncio
    async def test_no_response_created_fallback_still_works(self):
        """If response.created never fires, __anext__ fallback emits message_start."""
        events = [
            # Skip response.created entirely
            _event(
                "response.output_item.added",
                item={"type": "message", "id": "msg_001"},
            ),
            _event(
                "response.output_text.delta",
                item_id="msg_001",
                delta="Hello",
            ),
            _event(
                "response.output_item.done",
                item={"type": "message", "id": "msg_001"},
            ),
            _response_completed(),
        ]
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=MockSSEStream(events),
            model="test-model",
        )
        chunks = await _collect_all(wrapper)

        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1
        # Verify it has the expected structure
        msg = message_starts[0]["message"]
        assert msg["role"] == "assistant"
        assert msg["type"] == "message"

"""
Tests for AnthropicStreamWrapper's multi-tool-call splitting.

Some upstream providers (e.g. mlx_lm.server) emit multiple complete
tool_calls inside a SINGLE OpenAI streaming delta when the model produces
parallel tool calls. Anthropic's streaming format requires one
``content_block`` per ``tool_use`` and the downstream converter in
``AnthropicStreamWrapper`` indexes ``tool_calls[0]`` — so without splitting,
all but the first tool_call are silently dropped from the converted
``/v1/messages`` stream.

These tests verify that ``AnthropicStreamWrapper`` splits such chunks into
one tool_call per chunk before the converter sees them.
"""

import asyncio
import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../"))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
    _MultiToolCallSplitter,
)


class _DualProtocolStream:
    """Iterator that exposes BOTH sync and async protocols, matching the
    pattern of LiteLLM's CustomStreamWrapper. Lets a single fixture be
    consumed via either ``for`` or ``async for``.
    """

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._items):
            raise StopIteration
        v = self._items[self._idx]
        self._idx += 1
        return v

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        v = self._items[self._idx]
        self._idx += 1
        return v


def _make_chunk(tool_calls: List[dict]) -> MagicMock:
    """Build a minimal mock ModelResponseStream with the given tool_calls in delta."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.tool_calls = [
        MagicMock(
            id=tc["id"],
            type="function",
            function=MagicMock(name=tc["name"], arguments=tc["arguments"]),
            index=tc.get("index", i),
        )
        for i, tc in enumerate(tool_calls)
    ]
    chunk.choices[0].finish_reason = None
    return chunk


class TestSplitChunkByToolCalls:
    """Direct tests of the static ``_split_chunk_by_tool_calls`` helper."""

    def test_none_chunk_passthrough(self):
        assert AnthropicStreamWrapper._split_chunk_by_tool_calls(None) == [None]

    def test_string_none_passthrough(self):
        assert AnthropicStreamWrapper._split_chunk_by_tool_calls("None") == ["None"]

    def test_single_tool_call_passthrough(self):
        chunk = _make_chunk(
            [
                {"id": "call_a", "name": "Foo", "arguments": '{"x":1}'},
            ]
        )
        result = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        assert len(result) == 1
        assert result[0] is chunk  # same instance, not copied

    def test_two_parallel_tool_calls_split(self):
        chunk = _make_chunk(
            [
                {"id": "call_a", "name": "Foo", "arguments": '{"x":1}'},
                {"id": "call_b", "name": "Bar", "arguments": '{"y":2}'},
            ]
        )
        result = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        assert len(result) == 2
        # Each split chunk holds exactly one tool_call
        assert len(result[0].choices[0].delta.tool_calls) == 1
        assert len(result[1].choices[0].delta.tool_calls) == 1
        # Splits are independent copies (not aliasing the same list)
        assert (
            result[0].choices[0].delta.tool_calls
            is not result[1].choices[0].delta.tool_calls
        )
        # Each split has the correct tool_call's id
        assert result[0].choices[0].delta.tool_calls[0].id == "call_a"
        assert result[1].choices[0].delta.tool_calls[0].id == "call_b"

    def test_three_parallel_tool_calls_split(self):
        chunk = _make_chunk(
            [
                {"id": "call_a", "name": "Foo", "arguments": "{}"},
                {"id": "call_b", "name": "Bar", "arguments": "{}"},
                {"id": "call_c", "name": "Baz", "arguments": "{}"},
            ]
        )
        result = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        assert len(result) == 3
        ids = [r.choices[0].delta.tool_calls[0].id for r in result]
        assert ids == ["call_a", "call_b", "call_c"]

    def test_chunk_without_choices_passthrough(self):
        chunk = MagicMock()
        chunk.choices = []
        result = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        assert result == [chunk]

    def test_chunk_with_none_delta_passthrough(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = None
        result = AnthropicStreamWrapper._split_chunk_by_tool_calls(chunk)
        assert result == [chunk]


class TestMultiToolCallSplitterSync:
    """Sync iteration of the dual-protocol splitter."""

    def test_normal_chunks_unchanged(self):
        c1 = _make_chunk([{"id": "a", "name": "F", "arguments": "{}"}])
        c2 = _make_chunk([{"id": "b", "name": "G", "arguments": "{}"}])
        splitter = _MultiToolCallSplitter(_DualProtocolStream([c1, c2]))
        result = list(splitter)
        assert len(result) == 2

    def test_multi_tool_call_chunk_expands(self):
        single = _make_chunk([{"id": "a", "name": "F", "arguments": "{}"}])
        multi = _make_chunk(
            [
                {"id": "b", "name": "G", "arguments": "{}"},
                {"id": "c", "name": "H", "arguments": "{}"},
            ]
        )
        splitter = _MultiToolCallSplitter(_DualProtocolStream([single, multi]))
        result = list(splitter)
        # 1 (single) + 2 (multi split) = 3 chunks
        assert len(result) == 3
        ids = [r.choices[0].delta.tool_calls[0].id for r in result]
        assert ids == ["a", "b", "c"]


class TestMultiToolCallSplitterAsync:
    """Async iteration of the dual-protocol splitter."""

    def test_async_multi_tool_call_chunk_expands(self):
        async def runit():
            single = _make_chunk([{"id": "a", "name": "F", "arguments": "{}"}])
            multi = _make_chunk(
                [
                    {"id": "b", "name": "G", "arguments": "{}"},
                    {"id": "c", "name": "H", "arguments": "{}"},
                ]
            )
            splitter = _MultiToolCallSplitter(_DualProtocolStream([single, multi]))
            out = []
            async for sub in splitter:
                out.append(sub)
            return out

        result = asyncio.run(runit())
        assert len(result) == 3
        ids = [r.choices[0].delta.tool_calls[0].id for r in result]
        assert ids == ["a", "b", "c"]


class TestStreamWrapperConstructor:
    """Verify the wrapper supports both protocols on the same instance shape."""

    def test_sync_consumption_of_dual_stream(self):
        chunks = [_make_chunk([{"id": "a", "name": "F", "arguments": "{}"}])]
        wrapper = AnthropicStreamWrapper(
            _DualProtocolStream(chunks), model="test-model"
        )
        # The internal splitter must support sync iteration (this is
        # what the existing ``AnthropicStreamWrapper.__next__`` for-loop uses)
        out = list(wrapper._completion_stream_splitter)
        assert len(out) == 1

    def test_async_consumption_of_dual_stream(self):
        single = _make_chunk([{"id": "a", "name": "F", "arguments": "{}"}])
        multi = _make_chunk(
            [
                {"id": "b", "name": "G", "arguments": "{}"},
                {"id": "c", "name": "H", "arguments": "{}"},
            ]
        )

        async def drain():
            wrapper = AnthropicStreamWrapper(
                _DualProtocolStream([single, multi]), model="test-model"
            )
            out = []
            async for sub in wrapper._completion_stream_splitter:
                out.append(sub)
            return out

        result = asyncio.run(drain())
        # Multi-tool-call delta is split: 1 + 2 = 3
        assert len(result) == 3
        ids = [r.choices[0].delta.tool_calls[0].id for r in result]
        assert ids == ["a", "b", "c"]


class TestAnthropicStreamWrapperEndToEnd:
    """End-to-end SSE-event check.

    Drives the full ``AnthropicStreamWrapper.__next__`` / ``__anext__``
    pipeline (not just the splitter) and asserts the converted Anthropic
    event sequence contains one ``content_block_start`` + ``content_block_stop``
    pair per tool_call when an upstream OpenAI delta carries multiple.
    Without the splitter, only the first tool_call would surface.
    """

    def _build_chunks(self):
        """Build a 3-chunk sequence: text, then a delta with TWO parallel
        tool_calls, then a finish chunk. Mirrors ``mlx_lm.server``'s output
        shape for parallel tool calls.
        """
        from litellm.types.utils import (
            ChatCompletionDeltaToolCall,
            Delta,
            Function,
            ModelResponseStream,
            StreamingChoices,
            Usage,
        )

        text_chunk = ModelResponseStream(
            id="chatcmpl-1",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="ok", role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        multi_tool_chunk = ModelResponseStream(
            id="chatcmpl-1",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_a",
                                type="function",
                                function=Function(
                                    name="get_weather",
                                    arguments='{"city": "Tokyo"}',
                                ),
                                index=0,
                            ),
                            ChatCompletionDeltaToolCall(
                                id="call_b",
                                type="function",
                                function=Function(
                                    name="get_time",
                                    arguments='{"tz": "Asia/Tokyo"}',
                                ),
                                index=1,
                            ),
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        )

        finish_chunk = ModelResponseStream(
            id="chatcmpl-1",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(index=0, delta=Delta(), finish_reason="tool_calls")
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        return [text_chunk, multi_tool_chunk, finish_chunk]

    def test_sync_two_parallel_tool_calls_yield_two_content_blocks(self):
        chunks = self._build_chunks()
        wrapper = AnthropicStreamWrapper(
            _DualProtocolStream(chunks), model="test-model"
        )

        events = list(wrapper)

        tool_starts = [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 2, (
            f"Expected 2 content_block_start of type tool_use, got "
            f"{len(tool_starts)}. Event types: "
            f"{[e.get('type') for e in events if isinstance(e, dict)]}"
        )
        # Both tool_use blocks carry distinct ids
        ids = [e["content_block"]["id"] for e in tool_starts]
        assert "call_a" in ids
        assert "call_b" in ids

        # Each tool_use must be paired with at least one input_json_delta
        # carrying its arguments
        deltas_by_index = {}
        for e in events:
            if (
                isinstance(e, dict)
                and e.get("type") == "content_block_delta"
                and e.get("delta", {}).get("type") == "input_json_delta"
            ):
                deltas_by_index.setdefault(e["index"], []).append(e)
        # Indices for the two tool_use blocks (both > 0; index 0 was the
        # leading text block)
        tool_indices = [e["index"] for e in tool_starts]
        for idx in tool_indices:
            assert (
                idx in deltas_by_index
            ), f"No input_json_delta for tool_use at content_block index {idx}"

    def test_async_two_parallel_tool_calls_yield_two_content_blocks(self):
        async def drain():
            chunks = self._build_chunks()
            wrapper = AnthropicStreamWrapper(
                _DualProtocolStream(chunks), model="test-model"
            )
            out = []
            async for e in wrapper:
                out.append(e)
            return out

        events = asyncio.run(drain())
        tool_starts = [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

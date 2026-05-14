"""
Regression tests for AnthropicStreamWrapper's first-chunk content_block_start
type detection.

Bug history: previously the initial content_block_start was hardcoded to
{"type":"text","text":""} regardless of whether the first chunk from the
upstream stream carried thinking or tool_use content. This caused subsequent
`thinking_delta` events to be emitted inside a `text` content_block, which
violates the Anthropic Messages-streaming spec and breaks strict downstream
parsers (e.g. Claude Code).

The fix peeks at the first chunk to determine the correct initial block type.
These tests verify the block_type-on-first-chunk contract for each delta type
the upstream may emit first: text, thinking (via either `thinking_blocks` or
`reasoning_content`), and tool_use. They also cover first-chunk tool-name
de-mapping and the `_is_empty_delta` helper.
"""

import os
import sys
from typing import List


sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class MockCompletionStream:
    """Minimal completion-stream stub for testing the wrapper."""

    def __init__(self, responses: List[ModelResponseStream]):
        self.responses = responses
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def _make_thinking_chunk(thinking_text: str) -> ModelResponseStream:
    """A streaming chunk that carries a thinking_blocks payload (qwen3/qwq style)."""
    delta = Delta(content="")
    delta.thinking_blocks = [
        {"type": "thinking", "thinking": thinking_text, "signature": ""}
    ]
    return ModelResponseStream(
        choices=[StreamingChoices(delta=delta, index=0, finish_reason=None)],
    )


def _make_reasoning_content_chunk(reasoning_text: str) -> ModelResponseStream:
    """A chunk that carries thinking via `reasoning_content` (OpenRouter style)."""
    delta = Delta(content="")
    delta.reasoning_content = reasoning_text
    return ModelResponseStream(
        choices=[StreamingChoices(delta=delta, index=0, finish_reason=None)],
    )


def _make_text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content=text),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _make_tool_call_name_chunk(tool_id: str, name: str) -> ModelResponseStream:
    """Tool-call declaration chunk: id + name, empty args. Common with Ollama qwen."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=tool_id,
                            function=Function(name=name, arguments=""),
                            index=0,
                            type="function",
                        )
                    ]
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _make_finish_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content="", stop_reason="end_turn"),
                index=0,
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _collect(wrapper: AnthropicStreamWrapper):
    chunks = []
    for c in wrapper:
        chunks.append(c)
    return chunks


def test_first_chunk_thinking_opens_thinking_block():
    """
    Regression: when the very first upstream chunk carries thinking content
    (e.g. qwen3/qwq emitting their `<think>...</think>` chain-of-thought),
    the initial content_block_start MUST declare type="thinking", not "text".
    Otherwise the thinking_delta events that follow are spec-incorrectly
    parented under a text block.
    """
    stream = MockCompletionStream(
        [
            _make_thinking_chunk("Let me think about this..."),
            _make_thinking_chunk(" The answer is 42."),
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="qwen3:27b")
    chunks = _collect(wrapper)

    # First two events: message_start, content_block_start
    assert chunks[0]["type"] == "message_start"
    assert chunks[1]["type"] == "content_block_start"
    # The bug: this used to be {"type":"text","text":""}. The fix opens a
    # thinking block so subsequent thinking_delta events land in the right
    # parent.
    assert (
        chunks[1]["content_block"]["type"] == "thinking"
    ), f"Expected thinking content_block, got: {chunks[1]['content_block']}"

    # All thinking_delta events MUST appear within this thinking block (index 0).
    thinking_deltas = [
        c
        for c in chunks
        if c.get("type") == "content_block_delta"
        and isinstance(c.get("delta"), dict)
        and c["delta"].get("type") == "thinking_delta"
    ]
    assert len(thinking_deltas) >= 1, "Expected at least one thinking_delta event"
    for td in thinking_deltas:
        assert (
            td["index"] == 0
        ), "thinking_delta must reference the thinking content_block (index 0)"


def test_first_chunk_text_opens_text_block():
    """Sanity: text-first stream still opens a text block (no regression)."""
    stream = MockCompletionStream(
        [
            _make_text_chunk("Hello "),
            _make_text_chunk("world."),
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="any")
    chunks = _collect(wrapper)

    assert chunks[0]["type"] == "message_start"
    assert chunks[1]["type"] == "content_block_start"
    assert chunks[1]["content_block"]["type"] == "text"


def test_first_chunk_tool_call_opens_tool_use_block():
    """
    Tool-call-first stream opens a tool_use block directly. The spurious
    empty `text` block_start/stop pair the old code emitted is gone.
    """
    stream = MockCompletionStream(
        [
            _make_tool_call_name_chunk("call_abc", "get_weather"),
            # Tool argument chunk
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(
                            tool_calls=[
                                ChatCompletionDeltaToolCall(
                                    id=None,
                                    function=Function(
                                        name=None, arguments='{"city":"NYC"}'
                                    ),
                                    index=0,
                                    type=None,
                                )
                            ]
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            ),
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="any")
    chunks = _collect(wrapper)
    chunk_types = [c.get("type") for c in chunks]

    assert chunks[1]["type"] == "content_block_start"
    assert chunks[1]["content_block"]["type"] == "tool_use"
    assert chunks[1]["content_block"]["name"] == "get_weather"
    # No spurious empty-text block_start/stop pair:
    assert chunk_types.count("content_block_start") == 1
    assert chunk_types.count("content_block_stop") == 1


def test_first_chunk_empty_tool_args_does_not_emit_empty_delta():
    """
    The first chunk often carries only the tool name+id with empty arguments.
    The translated input_json_delta is empty; emitting it would be a spurious
    no-op event in the stream. The wrapper should suppress it (matching the
    gating already used by the block-transition handler).
    """
    stream = MockCompletionStream(
        [
            _make_tool_call_name_chunk("call_abc", "get_weather"),  # empty args
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="any")
    chunks = _collect(wrapper)
    chunk_types = [c.get("type") for c in chunks]

    # No content_block_delta events at all — the name-only chunk had no args
    # to emit, and the stop chunk produces message_delta, not block_delta.
    assert (
        "content_block_delta" not in chunk_types
    ), f"Did not expect any content_block_delta, got types: {chunk_types}"


def test_first_chunk_thinking_via_reasoning_content_opens_thinking_block():
    """
    Regression: providers like OpenRouter expose thinking content via
    `delta.reasoning_content` rather than `delta.thinking_blocks`. When such a
    chunk is first, the initial content_block_start MUST be type="thinking".
    The block-type peek has to recognize `reasoning_content` the same way the
    delta translator already does — otherwise the thinking_delta events that
    follow land inside a `text` block.
    """
    stream = MockCompletionStream(
        [
            _make_reasoning_content_chunk("Reasoning step one."),
            _make_reasoning_content_chunk(" Reasoning step two."),
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="openrouter/x")
    chunks = _collect(wrapper)

    assert chunks[0]["type"] == "message_start"
    assert chunks[1]["type"] == "content_block_start"
    assert (
        chunks[1]["content_block"]["type"] == "thinking"
    ), f"Expected thinking content_block, got: {chunks[1]['content_block']}"

    thinking_deltas = [
        c
        for c in chunks
        if c.get("type") == "content_block_delta"
        and isinstance(c.get("delta"), dict)
        and c["delta"].get("type") == "thinking_delta"
    ]
    assert len(thinking_deltas) >= 1, "Expected at least one thinking_delta event"
    for td in thinking_deltas:
        assert (
            td["index"] == 0
        ), "thinking_delta must reference the thinking block (index 0)"


def test_first_chunk_tool_call_restores_truncated_name():
    """
    Regression: OpenAI truncates tool names to its 64-char limit;
    `tool_name_mapping` holds the truncated->original mapping. When the FIRST
    chunk is a tool_use, the first-chunk path must restore the original name —
    the block-transition path (_should_start_new_content_block) already did.
    Without it, a first-chunk tool_use with a long name leaks the truncated
    name downstream.
    """
    original = "search_the_corporate_knowledge_base_for_relevant_internal_documents"
    truncated = original[:40]
    stream = MockCompletionStream(
        [
            _make_tool_call_name_chunk("call_xyz", truncated),
            _make_finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(
        completion_stream=stream,
        model="any",
        tool_name_mapping={truncated: original},
    )
    chunks = _collect(wrapper)

    assert chunks[1]["type"] == "content_block_start"
    assert chunks[1]["content_block"]["type"] == "tool_use"
    assert (
        chunks[1]["content_block"]["name"] == original
    ), f"first-chunk tool_use name was not de-mapped: {chunks[1]['content_block']}"


def test_async_first_chunk_tool_call_restores_truncated_name():
    """
    The async `__anext__` path must de-map the first-chunk tool name too — the
    fix touches both the sync and async iterators identically.
    """
    import asyncio

    original = "search_the_corporate_knowledge_base_for_relevant_internal_documents"
    truncated = original[:40]

    class AsyncMockCompletionStream:
        def __init__(self, responses: List[ModelResponseStream]):
            self.responses = responses
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    async def _run():
        stream = AsyncMockCompletionStream(
            [
                _make_tool_call_name_chunk("call_xyz", truncated),
                _make_finish_chunk(),
            ]
        )
        wrapper = AnthropicStreamWrapper(
            completion_stream=stream,
            model="any",
            tool_name_mapping={truncated: original},
        )
        collected = []
        while True:
            try:
                collected.append(await wrapper.__anext__())
            except StopAsyncIteration:
                break
        return collected

    chunks = asyncio.run(_run())
    block_starts = [c for c in chunks if c.get("type") == "content_block_start"]
    assert (
        block_starts
    ), f"no content_block_start emitted; got: {[c.get('type') for c in chunks]}"
    assert block_starts[0]["content_block"]["type"] == "tool_use"
    assert (
        block_starts[0]["content_block"]["name"] == original
    ), f"async first-chunk tool_use name was not de-mapped: {block_starts[0]['content_block']}"


def test_is_empty_delta_branches():
    """
    Direct coverage for the `_is_empty_delta` static helper across every delta
    type it recognizes, plus the non-content_block_delta and non-dict guards.
    """
    f = AnthropicStreamWrapper._is_empty_delta

    # Non-dict / wrong-shape inputs are never "empty deltas".
    assert f(None) is False
    assert f("not-a-dict") is False
    assert f({"type": "message_delta"}) is False
    assert f({"type": "content_block_delta", "delta": None}) is False

    # input_json_delta: empty/missing partial_json -> empty.
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ""},
            }
        )
        is True
    )
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            }
        )
        is False
    )

    # text_delta
    assert (
        f({"type": "content_block_delta", "delta": {"type": "text_delta", "text": ""}})
        is True
    )
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hi"},
            }
        )
        is False
    )

    # thinking_delta
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": ""},
            }
        )
        is True
    )
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "x"},
            }
        )
        is False
    )

    # signature_delta
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": ""},
            }
        )
        is True
    )
    assert (
        f(
            {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "sig"},
            }
        )
        is False
    )

    # Unknown delta types are not treated as empty.
    assert (
        f({"type": "content_block_delta", "delta": {"type": "some_other_delta"}})
        is False
    )

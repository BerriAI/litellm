"""
Tests that AnthropicStreamWrapper preserves tool call arguments
when a provider (e.g. Ollama) sends the complete tool call in a
single streaming chunk.

Regression test for https://github.com/BerriAI/litellm/issues/25605
"""

import json

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


def _make_text_chunk(text: str) -> ModelResponseStream:
    """Create a streaming chunk with text content."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=text, role="assistant"),
            )
        ],
        model="ollama/qwen3-coder",
    )


def _make_tool_call_chunk(
    name: str, arguments: str, tool_call_id: str = "call_123"
) -> ModelResponseStream:
    """Create a streaming chunk with a complete tool call (Ollama-style)."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=tool_call_id,
                            function=Function(arguments=arguments, name=name),
                            type="function",
                            index=0,
                        )
                    ],
                ),
            )
        ],
        model="ollama/qwen3-coder",
    )


def _make_finish_chunk(finish_reason: str = "tool_calls") -> ModelResponseStream:
    """Create a streaming chunk with finish_reason."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(content=None, role=None),
            )
        ],
        model="ollama/qwen3-coder",
    )


def _collect_all_chunks(wrapper: AnthropicStreamWrapper) -> list:
    """Iterate through the wrapper and collect all emitted chunks."""
    chunks = []
    for chunk in wrapper:
        chunks.append(chunk)
    return chunks


class TestOllamaStreamingToolArgs:
    """
    Ollama sends the complete tool call (name + full arguments) in a single
    streaming chunk. The AnthropicStreamWrapper must emit an input_json_delta
    content_block_delta after the content_block_start so the arguments are
    not lost.
    """

    def test_tool_arguments_preserved_in_single_chunk(self):
        """
        When a tool call arrives in one chunk (Ollama-style), the arguments
        must appear in an input_json_delta event after content_block_start.
        """
        tool_args = json.dumps({"pattern": "*.py"})
        stream = iter(
            [
                _make_text_chunk("Let me find those files."),
                _make_tool_call_chunk("Glob", tool_args),
                _make_finish_chunk(),
            ]
        )

        wrapper = AnthropicStreamWrapper(
            completion_stream=stream, model="ollama/qwen3-coder"
        )
        chunks = _collect_all_chunks(wrapper)

        # Find all input_json_delta events
        input_deltas = [
            c
            for c in chunks
            if c.get("type") == "content_block_delta"
            and c.get("delta", {}).get("type") == "input_json_delta"
        ]

        # There must be at least one input_json_delta carrying the arguments
        assert (
            len(input_deltas) > 0
        ), f"No input_json_delta events found. Chunks: {chunks}"

        # Concatenate all partial_json values
        combined_json = "".join(
            d["delta"]["partial_json"]
            for d in input_deltas
            if d["delta"].get("partial_json")
        )
        assert combined_json == tool_args

    def test_content_block_start_has_tool_name(self):
        """
        The content_block_start for a tool_use block must contain the tool name.
        """
        stream = iter(
            [
                _make_text_chunk("Searching..."),
                _make_tool_call_chunk("Glob", '{"pattern": "*.py"}'),
                _make_finish_chunk(),
            ]
        )

        wrapper = AnthropicStreamWrapper(
            completion_stream=stream, model="ollama/qwen3-coder"
        )
        chunks = _collect_all_chunks(wrapper)

        # Find content_block_start for tool_use
        tool_starts = [
            c
            for c in chunks
            if c.get("type") == "content_block_start"
            and c.get("content_block", {}).get("type") == "tool_use"
        ]

        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "Glob"

    def test_event_ordering_text_then_tool(self):
        """
        Events must follow Anthropic SSE protocol ordering:
        message_start -> content_block_start(text) -> content_block_delta(text) ->
        content_block_stop -> content_block_start(tool_use) ->
        content_block_delta(input_json_delta) -> ...
        """
        stream = iter(
            [
                _make_text_chunk("Hello"),
                _make_tool_call_chunk("Glob", '{"pattern": "*.py"}'),
                _make_finish_chunk(),
            ]
        )

        wrapper = AnthropicStreamWrapper(
            completion_stream=stream, model="ollama/qwen3-coder"
        )
        chunks = _collect_all_chunks(wrapper)

        types = [c.get("type") for c in chunks]

        # Verify the key ordering: after content_block_start(tool_use),
        # there must be a content_block_delta before content_block_stop
        tool_start_idx = None
        for i, c in enumerate(chunks):
            if (
                c.get("type") == "content_block_start"
                and c.get("content_block", {}).get("type") == "tool_use"
            ):
                tool_start_idx = i
                break

        assert tool_start_idx is not None, "No tool_use content_block_start found"

        # The next event(s) after tool_use start should include at least
        # one content_block_delta with input_json_delta before any stop
        remaining = chunks[tool_start_idx + 1 :]
        found_delta = False
        for c in remaining:
            if c.get("type") == "content_block_stop":
                break
            if (
                c.get("type") == "content_block_delta"
                and c.get("delta", {}).get("type") == "input_json_delta"
            ):
                found_delta = True

        assert found_delta, (
            "No input_json_delta found between content_block_start(tool_use) "
            f"and content_block_stop. Event types: {types}"
        )

import os
import sys
from typing import Any, AsyncIterator, Dict, List

import pytest

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


def _make_text_chunk(text: str, finish_reason=None) -> ModelResponseStream:
    """Helper to build a streaming chunk with text content."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(
                    content=text,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
            )
        ],
        model="test-model",
    )


def _make_tool_call_chunk(
    name: str, arguments: str, tool_call_id: str = "call_abc123", finish_reason=None
) -> ModelResponseStream:
    """
    Helper to build a streaming chunk with a tool call that carries BOTH
    name and arguments in the same chunk (atomic delivery).
    """
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=tool_call_id,
                            function=Function(arguments=arguments, name=name),
                            type="function",
                            index=0,
                        )
                    ],
                    audio=None,
                ),
            )
        ],
        model="test-model",
    )


def _make_finish_chunk(stop_reason: str = "tool_use") -> ModelResponseStream:
    """Helper to build a streaming chunk with only a finish reason."""
    return ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=stop_reason,
                index=0,
                delta=Delta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
            )
        ],
        model="test-model",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


# ---------------------------------------------------------------------------
# Sync iterator tests
# ---------------------------------------------------------------------------


def test_emit_input_json_delta_for_atomic_tool_call():
    """
    When a provider delivers name + arguments in a single streaming chunk
    (atomic delivery), the adapter must emit an input_json_delta event
    between content_block_start and content_block_stop.

    Regression test for https://github.com/BerriAI/litellm/issues/25561
    """
    chunks = [
        _make_text_chunk("Let me read that file for you."),
        _make_tool_call_chunk("Read", '{"file_path": "/etc/hosts"}'),
        _make_finish_chunk("tool_use"),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks),
        model="test-model",
    )

    events: List[Dict[str, Any]] = list(wrapper)

    # Gather the event types
    event_types = [e.get("type") for e in events]

    # There must be at least one content_block_delta with input_json_delta
    input_json_deltas: List[Dict[str, Any]] = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(input_json_deltas) >= 1, (
        f"Expected at least one input_json_delta event, got 0. "
        f"Event types: {event_types}"
    )

    # The concatenated partial_json must produce the correct arguments
    combined_json = "".join(d["delta"]["partial_json"] for d in input_json_deltas)
    assert combined_json == '{"file_path": "/etc/hosts"}'


def test_correct_event_sequence_for_atomic_tool_call():
    """
    Verify the full event sequence: message_start, text block (start/delta/stop),
    tool_use block (start/delta/stop), message_delta, message_stop.
    """
    chunks = [
        _make_text_chunk("Hello"),
        _make_tool_call_chunk("Read", '{"file_path": "/etc/hosts"}'),
        _make_finish_chunk("tool_use"),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks),
        model="test-model",
    )

    events: List[Dict[str, Any]] = list(wrapper)
    event_types = [e.get("type") for e in events]

    # Expected sequence pattern
    assert event_types[0] == "message_start"

    # Text content block
    assert "content_block_start" in event_types
    assert "content_block_stop" in event_types

    # tool_use content block
    tool_block_starts: List[Dict[str, Any]] = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_block_starts) == 1
    assert tool_block_starts[0]["content_block"]["name"] == "Read"

    # input_json_delta must appear between the tool_use start and stop
    tool_start_idx = events.index(tool_block_starts[0])
    input_deltas_after_tool_start: List[Dict[str, Any]] = [
        e
        for e in events[tool_start_idx:]
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(input_deltas_after_tool_start) >= 1

    # message_stop at the end
    assert event_types[-1] == "message_stop"


def test_split_tool_call_across_chunks():
    """
    For providers that split name and arguments across multiple chunks,
    the existing behavior should still work correctly.
    """
    # Chunk 1: tool call with name, empty arguments
    name_chunk = ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id="call_abc123",
                            function=Function(arguments="", name="Read"),
                            type="function",
                            index=0,
                        )
                    ],
                    audio=None,
                ),
            )
        ],
        model="gpt-4",
    )

    # Chunk 2: arguments only (no name)
    args_chunk = ModelResponseStream(
        id="chatcmpl-test",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=None,
                            function=Function(
                                arguments='{"file_path": "/etc/hosts"}', name=None
                            ),
                            type="function",
                            index=0,
                        )
                    ],
                    audio=None,
                ),
            )
        ],
        model="gpt-4",
    )

    chunks = [
        _make_text_chunk("Let me look."),
        name_chunk,
        args_chunk,
        _make_finish_chunk("tool_use"),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks),
        model="gpt-4",
    )

    events: List[Dict[str, Any]] = list(wrapper)

    # Should still produce input_json_delta events
    input_json_deltas: List[Dict[str, Any]] = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(input_json_deltas) >= 1

    combined_json = "".join(d["delta"]["partial_json"] for d in input_json_deltas)
    assert combined_json == '{"file_path": "/etc/hosts"}'


def test_has_meaningful_delta():
    """Verify the _has_meaningful_delta helper classifies delta types correctly."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter([]),
        model="test-model",
    )

    # input_json_delta with content -> meaningful
    assert (
        wrapper._has_meaningful_delta(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'},
            }
        )
        is True
    )

    # input_json_delta with empty string -> not meaningful
    assert (
        wrapper._has_meaningful_delta(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ""},
            }
        )
        is False
    )

    # text_delta is not emitted during block transitions
    assert (
        wrapper._has_meaningful_delta(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            }
        )
        is False
    )

    # message_delta (not a content_block_delta) -> not meaningful
    assert (
        wrapper._has_meaningful_delta(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        )
        is False
    )

    # thinking_delta is not emitted during block transitions
    assert (
        wrapper._has_meaningful_delta(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            }
        )
        is False
    )


# ---------------------------------------------------------------------------
# Async iterator tests
# ---------------------------------------------------------------------------


async def _async_iter(chunks: list) -> AsyncIterator:
    """Wrap a list into an async iterator."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_async_emit_input_json_delta_for_atomic_tool_call():
    """
    Async counterpart of test_emit_input_json_delta_for_atomic_tool_call.
    Verifies the __anext__ path also emits input_json_delta when a provider
    delivers name + arguments in a single streaming chunk.

    Regression test for https://github.com/BerriAI/litellm/issues/25561
    """
    chunks = [
        _make_text_chunk("Let me read that file for you."),
        _make_tool_call_chunk("Read", '{"file_path": "/etc/hosts"}'),
        _make_finish_chunk("tool_use"),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=_async_iter(chunks),
        model="test-model",
    )

    events: List[Dict[str, Any]] = []
    async for event in wrapper:
        events.append(event)

    event_types = [e.get("type") for e in events]

    input_json_deltas: List[Dict[str, Any]] = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(input_json_deltas) >= 1, (
        f"Expected at least one input_json_delta event, got 0. "
        f"Event types: {event_types}"
    )

    combined_json = "".join(d["delta"]["partial_json"] for d in input_json_deltas)
    assert combined_json == '{"file_path": "/etc/hosts"}'

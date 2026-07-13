"""
Test that AnthropicStreamWrapper emits input_json_delta when tool arguments
are bundled in the same streaming chunk as the function name/id.

Providers like xAI and Gemini include tool_call function arguments in
the first chunk rather than streaming them separately (OpenAI-style).
Without the fix, the AnthropicStreamWrapper silently dropped these
arguments, causing tool_use blocks to arrive with empty input {}.
"""

import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    StreamingChoices,
)


def _make_chunk(
    delta: Delta,
    finish_reason: str = None,
) -> MagicMock:
    """Create a minimal streaming chunk with the given delta and finish_reason."""
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=delta,
            logprobs=None,
        )
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


def _collect_events_sync(wrapper: AnthropicStreamWrapper) -> List[dict]:
    """Drain all events from a sync AnthropicStreamWrapper."""
    events = []
    for event in wrapper:
        events.append(event)
    return events


async def _collect_events_async(wrapper: AnthropicStreamWrapper) -> List[dict]:
    """Drain all events from an async AnthropicStreamWrapper."""
    events = []
    async for event in wrapper:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_async_stream_emits_input_json_delta_for_bundled_tool_args():
    """
    When a provider bundles tool_call arguments in the first streaming chunk
    (same chunk as name/id), the async wrapper must emit an input_json_delta
    content_block_delta after the tool_use content_block_start.
    """
    # Chunk 1: text content
    text_chunk = _make_chunk(Delta(content="Hello", role="assistant", tool_calls=None))

    # Chunk 2: tool call with name AND arguments in the same chunk (xAI/Gemini style)
    tool_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call_abc123",
                    function=Function(
                        name="get_weather",
                        arguments='{"location": "Boston"}',
                    ),
                    type="function",
                    index=0,
                )
            ],
        )
    )

    # Chunk 3: finish
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="tool_calls",
    )

    async def mock_stream():
        for c in [text_chunk, tool_chunk, finish_chunk]:
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="test-model",
    )

    events = await _collect_events_async(wrapper)
    event_types = [e.get("type") if isinstance(e, dict) else str(e) for e in events]

    # Find the tool_use content_block_start and subsequent input_json_delta
    tool_start_idx = None
    input_json_delta_idx = None

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("type") == "content_block_start"
            and isinstance(event.get("content_block"), dict)
            and event["content_block"].get("type") == "tool_use"
        ):
            tool_start_idx = i
        if (
            event.get("type") == "content_block_delta"
            and isinstance(event.get("delta"), dict)
            and event["delta"].get("type") == "input_json_delta"
        ):
            input_json_delta_idx = i

    assert (
        tool_start_idx is not None
    ), f"Expected content_block_start with type=tool_use; events: {event_types}"
    assert (
        input_json_delta_idx is not None
    ), f"Expected content_block_delta with input_json_delta; events: {event_types}"
    assert (
        input_json_delta_idx == tool_start_idx + 1
    ), "input_json_delta should immediately follow the tool_use content_block_start"

    # Verify the delta carries the tool arguments
    delta_event = events[input_json_delta_idx]
    assert delta_event["delta"][
        "partial_json"
    ], "input_json_delta should have non-empty partial_json"


@pytest.mark.asyncio
async def test_async_stream_no_extra_delta_when_tool_args_empty():
    """
    When a provider sends tool name/id WITHOUT arguments in the first chunk
    (OpenAI-style), the wrapper should NOT emit an extra input_json_delta
    after content_block_start. This verifies backward compatibility.
    """
    # Chunk 1: text
    text_chunk = _make_chunk(Delta(content="Hi", role="assistant", tool_calls=None))

    # Chunk 2: tool call with name but NO arguments (OpenAI-style)
    tool_name_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call_xyz789",
                    function=Function(name="get_weather", arguments=""),
                    type="function",
                    index=0,
                )
            ],
        )
    )

    # Chunk 3: arguments streamed separately
    tool_args_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id=None,
                    function=Function(name=None, arguments='{"location": "NYC"}'),
                    type="function",
                    index=0,
                )
            ],
        )
    )

    # Chunk 4: finish
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="tool_calls",
    )

    async def mock_stream():
        for c in [text_chunk, tool_name_chunk, tool_args_chunk, finish_chunk]:
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="test-model",
    )

    events = await _collect_events_async(wrapper)

    # Find tool_use content_block_start
    tool_start_idx = None
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("type") == "content_block_start"
            and isinstance(event.get("content_block"), dict)
            and event["content_block"].get("type") == "tool_use"
        ):
            tool_start_idx = i
            break

    assert tool_start_idx is not None

    # Count how many input_json_delta events appear after the tool_use block start.
    # With empty args in the trigger chunk, only the subsequent tool_args_chunk
    # should produce one — not the trigger chunk itself.
    input_json_deltas = [
        e
        for e in events[tool_start_idx + 1 :]
        if isinstance(e, dict)
        and e.get("type") == "content_block_delta"
        and isinstance(e.get("delta"), dict)
        and e["delta"].get("type") == "input_json_delta"
    ]
    assert len(input_json_deltas) == 1, (
        f"Expected exactly 1 input_json_delta (from the follow-up chunk), "
        f"got {len(input_json_deltas)}"
    )
    assert input_json_deltas[0]["delta"]["partial_json"] == '{"location": "NYC"}'


def test_sync_stream_emits_input_json_delta_for_bundled_tool_args():
    """
    Sync counterpart: when a provider bundles tool_call arguments in the first
    streaming chunk, the sync wrapper must also emit the input_json_delta.
    """
    text_chunk = _make_chunk(Delta(content="Hello", role="assistant", tool_calls=None))
    tool_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call_abc123",
                    function=Function(
                        name="get_weather",
                        arguments='{"location": "Boston"}',
                    ),
                    type="function",
                    index=0,
                )
            ],
        )
    )
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="tool_calls",
    )

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter([text_chunk, tool_chunk, finish_chunk]),
        model="test-model",
    )

    events = _collect_events_sync(wrapper)
    event_types = [e.get("type") if isinstance(e, dict) else str(e) for e in events]

    tool_start_idx = None
    input_json_delta_idx = None

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("type") == "content_block_start"
            and isinstance(event.get("content_block"), dict)
            and event["content_block"].get("type") == "tool_use"
        ):
            tool_start_idx = i
        if (
            event.get("type") == "content_block_delta"
            and isinstance(event.get("delta"), dict)
            and event["delta"].get("type") == "input_json_delta"
        ):
            input_json_delta_idx = i

    assert (
        tool_start_idx is not None
    ), f"Expected content_block_start with type=tool_use; events: {event_types}"
    assert (
        input_json_delta_idx is not None
    ), f"Expected content_block_delta with input_json_delta; events: {event_types}"
    assert (
        input_json_delta_idx == tool_start_idx + 1
    ), "input_json_delta should immediately follow the tool_use content_block_start"
    assert events[input_json_delta_idx]["delta"]["partial_json"]


def test_sync_stream_no_extra_delta_when_tool_args_empty():
    """
    Sync counterpart: empty args (OpenAI-style) should not emit an extra
    input_json_delta from the trigger chunk.
    """
    text_chunk = _make_chunk(Delta(content="Hi", role="assistant", tool_calls=None))
    tool_name_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call_xyz789",
                    function=Function(name="get_weather", arguments=""),
                    type="function",
                    index=0,
                )
            ],
        )
    )
    tool_args_chunk = _make_chunk(
        Delta(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id=None,
                    function=Function(name=None, arguments='{"location": "NYC"}'),
                    type="function",
                    index=0,
                )
            ],
        )
    )
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="tool_calls",
    )

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(
            [text_chunk, tool_name_chunk, tool_args_chunk, finish_chunk]
        ),
        model="test-model",
    )

    events = _collect_events_sync(wrapper)

    tool_start_idx = None
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("type") == "content_block_start"
            and isinstance(event.get("content_block"), dict)
            and event["content_block"].get("type") == "tool_use"
        ):
            tool_start_idx = i
            break

    assert tool_start_idx is not None

    input_json_deltas = [
        e
        for e in events[tool_start_idx + 1 :]
        if isinstance(e, dict)
        and e.get("type") == "content_block_delta"
        and isinstance(e.get("delta"), dict)
        and e["delta"].get("type") == "input_json_delta"
    ]
    assert len(input_json_deltas) == 1, (
        f"Expected exactly 1 input_json_delta (from the follow-up chunk), "
        f"got {len(input_json_deltas)}"
    )
    assert input_json_deltas[0]["delta"]["partial_json"] == '{"location": "NYC"}'

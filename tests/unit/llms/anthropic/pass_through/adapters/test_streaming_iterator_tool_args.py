"""
Test that AnthropicStreamWrapper emits input_json_delta when tool arguments
are bundled in the same streaming chunk as the function name/id.

Providers like xAI and Gemini include tool_call function arguments in
the first chunk rather than streaming them separately (OpenAI-style).
Without the fix, the AnthropicStreamWrapper silently dropped these
arguments, causing tool_use blocks to arrive with empty input {}.
"""

import asyncio
import json

from typing import List
from unittest.mock import MagicMock

import pytest


from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import (
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

    assert tool_start_idx is not None, f"Expected content_block_start with type=tool_use; events: {event_types}"
    assert input_json_delta_idx is not None, (
        f"Expected content_block_delta with input_json_delta; events: {event_types}"
    )
    assert input_json_delta_idx == tool_start_idx + 1, (
        "input_json_delta should immediately follow the tool_use content_block_start"
    )

    # Verify the delta carries the tool arguments
    delta_event = events[input_json_delta_idx]
    assert json.loads(delta_event["delta"]["partial_json"]) == {"location": "Boston"}


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
        f"Expected exactly 1 input_json_delta (from the follow-up chunk), got {len(input_json_deltas)}"
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

    assert tool_start_idx is not None, f"Expected content_block_start with type=tool_use; events: {event_types}"
    assert input_json_delta_idx is not None, (
        f"Expected content_block_delta with input_json_delta; events: {event_types}"
    )
    assert input_json_delta_idx == tool_start_idx + 1, (
        "input_json_delta should immediately follow the tool_use content_block_start"
    )
    assert json.loads(events[input_json_delta_idx]["delta"]["partial_json"]) == {"location": "Boston"}


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
        completion_stream=iter([text_chunk, tool_name_chunk, tool_args_chunk, finish_chunk]),
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
        f"Expected exactly 1 input_json_delta (from the follow-up chunk), got {len(input_json_deltas)}"
    )
    assert input_json_deltas[0]["delta"]["partial_json"] == '{"location": "NYC"}'


# --- Several tool calls in one chunk (https://github.com/BerriAI/litellm/issues/44029) ---

_TMP_ARGS = '{"command": "ls -d /tmp", "description": "List the /tmp directory"}'
_VAR_ARGS = '{"command": "ls -d /var", "description": "List the /var directory"}'


def _tool_call(index: int, arguments: str, call_id: str = None, name: str = None) -> ChatCompletionDeltaToolCall:
    return ChatCompletionDeltaToolCall(
        id=call_id,
        type="function" if call_id else None,
        function=Function(name=name, arguments=arguments),
        index=index,
    )


def _stream_chunk(delta: Delta, finish_reason: str = None, usage: Usage = None) -> ModelResponseStream:
    chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)])
    if usage is not None:
        chunk.usage = usage
    return chunk


def _drain(chunks: list[ModelResponseStream], use_async: bool) -> list[dict]:
    async def _agen():
        for chunk in chunks:
            yield chunk

    if use_async:
        wrapper = AnthropicStreamWrapper(completion_stream=_agen(), model="mistral-large-2512")
        return asyncio.run(_collect_events_async(wrapper))
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="mistral-large-2512")
    return _collect_events_sync(wrapper)


def _blocks(events: list[dict]) -> list[dict]:
    """Rebuild each content block from its start, deltas and stop events."""
    blocks: dict = {}
    for event in events:
        if event.get("type") == "content_block_start":
            assert event["index"] not in blocks, f"block {event['index']} opened twice"
            blocks[event["index"]] = {"start": event["content_block"], "json": "", "text": "", "stopped": False}
        elif event.get("type") == "content_block_delta":
            block = blocks[event["index"]]
            assert not block["stopped"], f"delta after stop on block {event['index']}"
            block["json"] += event["delta"].get("partial_json", "")
            block["text"] += event["delta"].get("text", "")
        elif event.get("type") == "content_block_stop":
            blocks[event["index"]]["stopped"] = True
    assert list(blocks) == list(range(len(blocks))), f"block indexes not contiguous: {list(blocks)}"
    assert all(block["stopped"] for block in blocks.values())
    return [blocks[index] for index in sorted(blocks)]


def _message_delta(events: list[dict]) -> dict:
    deltas = [event for event in events if event.get("type") == "message_delta"]
    assert len(deltas) == 1
    last_stop = max(i for i, event in enumerate(events) if event.get("type") == "content_block_stop")
    assert events.index(deltas[0]) > last_stop
    return deltas[0]


@pytest.mark.parametrize("use_async", [False, True])
def test_two_complete_tool_calls_in_one_chunk_with_finish_and_usage(use_async):
    """Mistral bundles both calls, their full arguments, finish_reason and usage into one chunk."""
    chunk = _stream_chunk(
        Delta(
            role="assistant",
            tool_calls=[
                _tool_call(0, _TMP_ARGS, call_id="M7B4U5UJs", name="Bash"),
                _tool_call(1, _VAR_ARGS, call_id="jWEVB6J9J", name="Read"),
            ],
        ),
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=40, completion_tokens=12, total_tokens=52),
    )

    events = _drain([chunk], use_async)
    blocks = _blocks(events)

    assert [block["start"]["type"] for block in blocks] == ["tool_use", "tool_use"]
    assert [block["start"]["id"] for block in blocks] == ["M7B4U5UJs", "jWEVB6J9J"]
    assert [block["start"]["name"] for block in blocks] == ["Bash", "Read"]
    assert [json.loads(block["json"]) for block in blocks] == [json.loads(_TMP_ARGS), json.loads(_VAR_ARGS)]
    message_delta = _message_delta(events)
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"]["output_tokens"] == 12


@pytest.mark.parametrize("use_async", [False, True])
def test_text_chunk_then_chunk_with_two_tool_calls(use_async):
    chunks = [
        _stream_chunk(Delta(role="assistant", content="Listing both.")),
        _stream_chunk(
            Delta(
                tool_calls=[
                    _tool_call(0, _TMP_ARGS, call_id="call_a", name="Bash"),
                    _tool_call(1, _VAR_ARGS, call_id="call_b", name="Bash"),
                ]
            )
        ),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["type"] for block in blocks] == ["text", "tool_use", "tool_use"]
    assert blocks[0]["text"] == "Listing both."
    assert [block["start"]["id"] for block in blocks[1:]] == ["call_a", "call_b"]
    assert [json.loads(block["json"]) for block in blocks[1:]] == [json.loads(_TMP_ARGS), json.loads(_VAR_ARGS)]
    assert _message_delta(events)["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("use_async", [False, True])
def test_two_calls_opened_together_then_interleaved_continuations(use_async):
    """Anthropic blocks are sequential, so call 1 is held until call 0 can no longer grow.

    Call 0 keeps streaming live; call 1's fragments are gathered by ``index`` and
    its block is written whole once a chunk that is not a fragment arrives.
    """
    chunks = [
        _stream_chunk(
            Delta(
                role="assistant",
                tool_calls=[
                    _tool_call(0, '{"command": ', call_id="call_a", name="Bash"),
                    _tool_call(1, '{"command": ', call_id="call_b", name="Bash"),
                ],
            )
        ),
        _stream_chunk(Delta(tool_calls=[_tool_call(1, '"ls -d /var"')])),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, '"ls -d /tmp"')])),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, "}"), _tool_call(1, "}")])),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["id"] for block in blocks] == ["call_a", "call_b"]
    assert [json.loads(block["json"]) for block in blocks] == [{"command": "ls -d /tmp"}, {"command": "ls -d /var"}]
    assert _message_delta(events)["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("use_async", [False, True])
def test_chunk_continuing_open_call_and_opening_next_one(use_async):
    """Entry 0 finishes the open call, entry 1 starts a new one in the same chunk."""
    chunks = [
        _stream_chunk(
            Delta(role="assistant", tool_calls=[_tool_call(0, '{"command": ', call_id="call_a", name="Bash")])
        ),
        _stream_chunk(
            Delta(
                tool_calls=[
                    _tool_call(0, '"ls -d /tmp"}'),
                    _tool_call(1, '{"command": ', call_id="call_b", name="Bash"),
                ]
            )
        ),
        _stream_chunk(Delta(tool_calls=[_tool_call(1, '"ls -d /var"}')])),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["id"] for block in blocks] == ["call_a", "call_b"]
    assert [json.loads(block["json"]) for block in blocks] == [{"command": "ls -d /tmp"}, {"command": "ls -d /var"}]


@pytest.mark.parametrize("use_async", [False, True])
def test_next_call_opening_beside_a_fragment_of_the_open_call_waits_for_it(use_async):
    """Call 1 opens in a chunk that also extends call 0, and call 0 keeps streaming after it."""
    chunks = [
        _stream_chunk(
            Delta(role="assistant", tool_calls=[_tool_call(0, '{"command": ', call_id="call_a", name="Bash")])
        ),
        _stream_chunk(
            Delta(
                tool_calls=[
                    _tool_call(0, '"ls'),
                    _tool_call(1, '{"command": ', call_id="call_b", name="Bash"),
                ]
            )
        ),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, ' -d /tmp"}')])),
        _stream_chunk(Delta(tool_calls=[_tool_call(1, '"ls -d /var"}')])),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["id"] for block in blocks] == ["call_a", "call_b"]
    assert [json.loads(block["json"]) for block in blocks] == [{"command": "ls -d /tmp"}, {"command": "ls -d /var"}]
    assert _message_delta(events)["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("use_async", [False, True])
def test_three_calls_in_one_chunk_come_out_in_index_order(use_async):
    chunks = [
        _stream_chunk(
            Delta(
                role="assistant",
                tool_calls=[
                    _tool_call(0, '{"command": "ls -d /tmp"}', call_id="call_a", name="Bash"),
                    _tool_call(1, '{"command": "ls -d /var"}', call_id="call_b", name="Bash"),
                    _tool_call(2, '{"command": "ls -d /etc"}', call_id="call_c", name="Bash"),
                ],
            )
        ),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    blocks = _blocks(_drain(chunks, use_async))

    assert [block["start"]["id"] for block in blocks] == ["call_a", "call_b", "call_c"]
    assert [json.loads(block["json"])["command"] for block in blocks] == ["ls -d /tmp", "ls -d /var", "ls -d /etc"]


@pytest.mark.parametrize("use_async", [False, True])
def test_held_call_is_written_when_the_stream_ends_without_a_finish_chunk(use_async):
    chunks = [
        _stream_chunk(
            Delta(
                role="assistant",
                tool_calls=[
                    _tool_call(0, '{"command": "ls -d /tmp"}', call_id="call_a", name="Bash"),
                    _tool_call(1, '{"command": "ls -d /var"}', call_id="call_b", name="Bash"),
                ],
            )
        ),
    ]

    events = _drain(chunks, use_async)
    starts = [event for event in events if event.get("type") == "content_block_start"]
    arguments = [
        "".join(
            event["delta"]["partial_json"]
            for event in events
            if event.get("type") == "content_block_delta" and event["index"] == start["index"]
        )
        for start in starts
    ]

    assert [start["content_block"]["id"] for start in starts] == ["call_a", "call_b"]
    assert [json.loads(argument) for argument in arguments] == [{"command": "ls -d /tmp"}, {"command": "ls -d /var"}]


@pytest.mark.parametrize("use_async", [False, True])
def test_call_opening_alone_while_another_is_held_waits_behind_the_live_call(use_async):
    chunks = [
        _stream_chunk(
            Delta(
                role="assistant",
                tool_calls=[
                    _tool_call(0, '{"command": ', call_id="call_a", name="Bash"),
                    _tool_call(1, '{"command": "ls -d /var"}', call_id="call_b", name="Bash"),
                ],
            )
        ),
        _stream_chunk(Delta(tool_calls=[_tool_call(2, '{"command": ', call_id="call_c", name="Bash")])),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, '"ls -d /tmp"}')])),
        _stream_chunk(Delta(tool_calls=[_tool_call(2, '"ls -d /etc"}')])),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["id"] for block in blocks] == ["call_a", "call_b", "call_c"]
    assert [json.loads(block["json"])["command"] for block in blocks] == ["ls -d /tmp", "ls -d /var", "ls -d /etc"]
    assert _message_delta(events)["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("use_async", [False, True])
def test_one_call_split_across_chunks_is_unchanged(use_async):
    chunks = [
        _stream_chunk(Delta(role="assistant", tool_calls=[_tool_call(0, "", call_id="call_a", name="Bash")])),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, '{"command": ')])),
        _stream_chunk(Delta(tool_calls=[_tool_call(0, '"ls -d /tmp"}')])),
        _stream_chunk(Delta(), finish_reason="tool_calls"),
    ]

    events = _drain(chunks, use_async)
    blocks = _blocks(events)

    assert [block["start"]["id"] for block in blocks] == ["call_a"]
    assert json.loads(blocks[0]["json"]) == {"command": "ls -d /tmp"}
    deltas = [event for event in events if event.get("type") == "content_block_delta"]
    assert [delta["delta"]["partial_json"] for delta in deltas] == ['{"command": ', '"ls -d /tmp"}']

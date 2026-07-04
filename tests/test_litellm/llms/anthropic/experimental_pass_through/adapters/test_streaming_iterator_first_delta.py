"""
Regression tests for issue #30014.

When LiteLLM proxies ``client -> /v1/messages -> /v1/chat/completions`` and a
streaming chunk both *triggers* a new Anthropic content block (its type differs
from the active block) and *carries* the first delta of that new block, the
trigger chunk's delta must be re-emitted as a ``content_block_delta``.

The synthesized ``content_block_start`` always carries an empty body, so before
the fix the first non-empty ``text_delta`` of every transitioned block was
silently dropped — e.g. text resuming after a tool call started from the second
token ("The weather is nice." was lost, "Hi" rendered as ""). Bundled
``input_json_delta`` tool arguments were already preserved and must stay
preserved, and empty trigger deltas must not produce spurious events.

Also covers the inverse regression: a chunk whose translated delta carries no
payload must not be emitted at all. The translate fallback types empty deltas
as ``text_delta`` regardless of the active block, so an empty reasoning delta
mid-thinking-block (Bedrock Converse sends these) used to emit ``text_delta``
into an open ``thinking`` block, crashing Anthropic SDK clients (Claude Code)
with "Content block is not a text block".
"""

import os
import sys
from typing import List, Optional
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
    PromptTokensDetailsWrapper,
    StreamingChoices,
    Usage,
)


def _make_chunk(delta: Delta, finish_reason: Optional[str] = None) -> MagicMock:
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


def _thinking_chunk(thinking: str, signature: str = "") -> MagicMock:
    block = {"type": "thinking", "thinking": thinking}
    if signature:
        block["signature"] = signature
    return _make_chunk(Delta(content=None, thinking_blocks=[block]))


def _tool_chunk(call_id: str, name: Optional[str], arguments: Optional[str]) -> MagicMock:
    return _make_chunk(
        Delta(
            content=None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id=call_id,
                    function=Function(name=name, arguments=arguments),
                    type="function",
                    index=0,
                )
            ],
        )
    )


class _AsyncStream:
    def __init__(self, items: List[MagicMock]):
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _drain_sync(wrapper: AnthropicStreamWrapper) -> List[dict]:
    return list(wrapper)


async def _drain_async(wrapper: AnthropicStreamWrapper) -> List[dict]:
    return [event async for event in wrapper]


def _text_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]


def _input_json_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["partial_json"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "input_json_delta"
    ]


def _thinking_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    ]


def _signature_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["signature"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "signature_delta"
    ]


_DELTA_TYPES_PER_BLOCK_TYPE = {
    "text": {"text_delta"},
    "thinking": {"thinking_delta", "signature_delta"},
    "tool_use": {"input_json_delta"},
}


def _assert_deltas_match_their_block_type(events: List[dict]) -> None:
    """Enforce the invariant the Anthropic SDK enforces client-side: every
    ``content_block_delta`` must be of a type valid for the block opened by
    the most recent ``content_block_start`` at the same index.
    """
    block_types = {}
    for event in events:
        if event.get("type") == "content_block_start":
            block_types[event["index"]] = event["content_block"]["type"]
        if event.get("type") == "content_block_delta":
            block_type = block_types[event["index"]]
            assert event["delta"]["type"] in _DELTA_TYPES_PER_BLOCK_TYPE[block_type], (
                f"{event['delta']['type']} emitted into a {block_type} block: {event}"
            )


def test_held_stop_reason_usage_merge_preserves_openai_cache_token_details():
    """OpenAI-compatible usage chunks carry cache reads in prompt_tokens_details."""
    wrapper = AnthropicStreamWrapper(completion_stream=iter([]), model="claude-x")
    wrapper.holding_stop_reason_chunk = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    usage_chunk = MagicMock()
    usage_chunk.usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )

    merged_chunk = wrapper._merge_usage_into_held_stop_reason_chunk(usage_chunk)

    assert merged_chunk["usage"]["input_tokens"] == 70
    assert merged_chunk["usage"]["output_tokens"] == 50
    assert merged_chunk["usage"]["cache_read_input_tokens"] == 30
    assert merged_chunk["usage"]["cache_creation_input_tokens"] == 20


def test_first_text_delta_after_tool_use_is_not_dropped_sync():
    """A tool_use -> text transition (text resuming after a tool call) carries
    the resumed text's first token in the trigger chunk. Without the fix it was
    dropped, so "The weather is nice." vanished and the answer began at " Bye.".
    """
    chunks = [
        _make_chunk(Delta(content="Let me check.")),
        _tool_chunk("call_1", "get_weather", '{"city":'),
        _tool_chunk("call_1", None, ' "NY"}'),
        _make_chunk(Delta(content="The weather is nice.")),
        _make_chunk(Delta(content=" Bye.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _input_json_deltas(events) == ['{"city":', ' "NY"}']
    assert _text_deltas(events) == [
        "Let me check.",
        "The weather is nice.",
        " Bye.",
    ]


@pytest.mark.asyncio
async def test_first_text_delta_after_tool_use_is_not_dropped_async():
    """Async path mirrors the sync regression — the proxy serves the async
    iterator, so it must preserve the first resumed text delta too.
    """
    chunks = [
        _make_chunk(Delta(content="Let me check.")),
        _tool_chunk("call_1", "get_weather", '{"city": "NY"}'),
        _make_chunk(Delta(content="The weather is nice.")),
        _make_chunk(Delta(content=" Bye.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=_AsyncStream(chunks), model="claude-x")
    events = await _drain_async(wrapper)

    assert _input_json_deltas(events) == ['{"city": "NY"}']
    assert _text_deltas(events) == [
        "Let me check.",
        "The weather is nice.",
        " Bye.",
    ]


def test_mixed_reasoning_and_text_chunk_keeps_delta_types_on_matching_blocks_sync():
    chunks = [
        _make_chunk(Delta(thinking_blocks=[{"type": "thinking", "thinking": "first thought", "signature": ""}])),
        _make_chunk(Delta(content="FINAL: ABC", reasoning_content="last thought")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    start_types_by_index = {
        e["index"]: e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"
    }
    delta_types_by_index = [(e["index"], e["delta"]["type"]) for e in events if e.get("type") == "content_block_delta"]

    assert start_types_by_index[1] == "thinking"
    assert start_types_by_index[2] == "text"
    assert (2, "thinking_delta") not in delta_types_by_index
    assert _thinking_deltas(events) == ["first thought", "last thought"]
    assert _text_deltas(events) == ["FINAL: ABC"]


@pytest.mark.asyncio
async def test_mixed_reasoning_and_text_chunk_keeps_delta_types_on_matching_blocks_async():
    chunks = [
        _make_chunk(Delta(thinking_blocks=[{"type": "thinking", "thinking": "first thought", "signature": ""}])),
        _make_chunk(Delta(content="FINAL: ABC", reasoning_content="last thought")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=_AsyncStream(chunks), model="claude-x")
    events = await _drain_async(wrapper)

    start_types_by_index = {
        e["index"]: e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"
    }
    delta_types_by_index = [(e["index"], e["delta"]["type"]) for e in events if e.get("type") == "content_block_delta"]

    assert start_types_by_index[1] == "thinking"
    assert start_types_by_index[2] == "text"
    assert (2, "thinking_delta") not in delta_types_by_index
    assert _thinking_deltas(events) == ["first thought", "last thought"]
    assert _text_deltas(events) == ["FINAL: ABC"]


def test_mixed_reasoning_and_text_chunk_with_empty_thinking_blocks_keeps_reasoning_sync():
    chunks = [
        _make_chunk(Delta(thinking_blocks=[{"type": "thinking", "thinking": "first thought", "signature": ""}])),
        _make_chunk(Delta(content="FINAL: ABC", reasoning_content="last thought", thinking_blocks=[])),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _thinking_deltas(events) == ["first thought", "last thought"]
    assert _text_deltas(events) == ["FINAL: ABC"]


def test_reasoning_content_with_non_thinking_block_type_is_not_dropped_sync():
    chunks = [
        _make_chunk(Delta(reasoning_content="fallback thought", thinking_blocks=[{"type": "redacted_thinking"}])),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    start_types_by_index = {
        e["index"]: e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"
    }

    assert start_types_by_index[1] == "thinking"
    assert _thinking_deltas(events) == ["fallback thought"]
    assert _text_deltas(events) == []


def test_single_first_text_token_after_tool_use_preserved_sync():
    """Minimal reproduction of the issue's example: a single short text token
    ("Hi") resuming after a tool call. Without the fix the whole answer is
    dropped because its only delta sits in the transition trigger chunk.
    """
    chunks = [
        _tool_chunk("call_1", "get_weather", '{"city": "NY"}'),
        _make_chunk(Delta(content="Hi")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Hi"]


def test_multiple_text_deltas_after_tool_use_preserved_sync():
    """Multiple-delta edge case: only the *first* text delta sits in the
    transition trigger chunk; the rest stream normally. All of them — leading
    one included — must reach the client in order.
    """
    chunks = [
        _tool_chunk("call_1", "get_weather", '{"city": "NY"}'),
        _make_chunk(Delta(content="Hi")),
        _make_chunk(Delta(content=", how ")),
        _make_chunk(Delta(content="can I help ")),
        _make_chunk(Delta(content="you?")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Hi", ", how ", "can I help ", "you?"]
    assert "".join(_text_deltas(events)) == "Hi, how can I help you?"


def test_empty_trigger_delta_is_not_re_emitted_sync():
    """A transition whose trigger chunk carries no content (empty text) must
    NOT produce a spurious empty ``content_block_delta`` — only the synthesized
    ``content_block_start`` is emitted for the new block. Here a ``tool_use ->
    text`` transition is triggered by an empty-content chunk; the re-emit guard
    must reject it so the new text block opens without a leading empty delta.
    """
    chunks = [
        _tool_chunk("call_1", "get_weather", '{"city": "NY"}'),
        # tool_use -> text transition triggered by an empty content chunk; the
        # real text arrives in the following chunk.
        _make_chunk(Delta(content="")),
        _make_chunk(Delta(content="real text")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    # No empty-string text_delta should be present.
    assert "" not in _text_deltas(events)
    assert "".join(_text_deltas(events)) == "real text"


def test_bundled_tool_args_on_transition_still_preserved_sync():
    """Existing behavior guard: when the trigger chunk that opens a tool_use
    block also carries arguments (xAI/Gemini style), the ``input_json_delta``
    must still be emitted after ``content_block_start``.
    """
    chunks = [
        _make_chunk(Delta(content="Calling a tool.")),
        _tool_chunk("call_1", "get_weather", '{"city": "NY"}'),
        _make_chunk(Delta(content=None), finish_reason="tool_calls"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Calling a tool."]
    assert _input_json_deltas(events) == ['{"city": "NY"}']


@pytest.mark.parametrize(
    "processed_chunk, expected",
    [
        # Non-empty deltas of every type must be re-emitted.
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "x"},
            },
            True,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
            True,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "t"},
            },
            True,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "s"},
            },
            True,
        ),
        # Empty deltas must NOT be re-emitted (no spurious events).
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": ""},
            },
            False,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ""},
            },
            False,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": ""},
            },
            False,
        ),
        (
            {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": ""},
            },
            False,
        ),
        # Unknown delta type / non-content_block_delta / malformed delta.
        (
            {"type": "content_block_delta", "delta": {"type": "other_delta"}},
            False,
        ),
        ({"type": "message_delta", "delta": {"stop_reason": "stop"}}, False),
        ({"type": "content_block_delta", "delta": None}, False),
    ],
)
def test_delta_has_content_branches(processed_chunk, expected):
    """Directly exercise the emission predicate across all delta types and the
    empty/malformed guards, so the helper's behavior is pinned independently of
    upstream chunk-translation details.
    """
    assert AnthropicStreamWrapper._delta_has_content(processed_chunk) is expected


def _empty_reasoning_delta_mid_thinking_chunks() -> List[MagicMock]:
    return [
        _thinking_chunk("Let me think"),
        _thinking_chunk(""),
        _thinking_chunk("", signature="sig123"),
        _make_chunk(Delta(content="Hello")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def _assert_empty_reasoning_delta_suppressed(events: List[dict]) -> None:
    _assert_deltas_match_their_block_type(events)
    assert _thinking_deltas(events) == ["Let me think"]
    assert _signature_deltas(events) == ["sig123"]
    assert _text_deltas(events) == ["Hello"]


def test_empty_reasoning_delta_mid_thinking_block_is_suppressed_sync():
    """Bedrock Converse repro: an empty reasoning delta arriving inside an open
    thinking block used to be emitted as ``text_delta {"text": ""}`` at the
    thinking block's index (no block transition), which crashes Claude Code's
    Anthropic SDK with "Content block is not a text block". It must be dropped,
    while the surrounding thinking/signature/text deltas all still flow.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_empty_reasoning_delta_mid_thinking_chunks()),
        model="claude-x",
    )
    _assert_empty_reasoning_delta_suppressed(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_empty_reasoning_delta_mid_thinking_block_is_suppressed_async():
    """Async twin of the Bedrock Converse repro — the proxy serves the async
    iterator, so the skip must exist on that path too.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_empty_reasoning_delta_mid_thinking_chunks()),
        model="claude-x",
    )
    _assert_empty_reasoning_delta_suppressed(await _drain_async(wrapper))


def test_empty_content_chunk_mid_text_block_is_suppressed_sync():
    """An empty-content chunk arriving mid-text-block (no transition) used to
    emit a pointless ``text_delta {"text": ""}``; it must be dropped without
    affecting the surrounding text deltas.
    """
    chunks = [
        _make_chunk(Delta(content="Hi")),
        _make_chunk(Delta(content="")),
        _make_chunk(Delta(content=" there")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Hi", " there"]
    _assert_deltas_match_their_block_type(events)

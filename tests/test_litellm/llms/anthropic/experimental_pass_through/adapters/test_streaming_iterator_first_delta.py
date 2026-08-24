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

from typing import List, Optional
from unittest.mock import MagicMock

import pytest


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


def _full_snapshot_signature_chunks() -> List[MagicMock]:
    """Mirror litellm's real Anthropic streaming: incremental ``thinking_delta``
    chunks (empty signature), then a terminal chunk whose ``thinking_blocks`` entry
    re-states the *full accumulated thinking text* together with the signature
    (anthropic/chat/handler.py builds the signature_delta event this way), then the
    answer text.
    """
    return [
        _thinking_chunk("Let me "),
        _thinking_chunk("think about it."),
        _thinking_chunk("Let me think about it.", signature="sig-abc"),
        _make_chunk(Delta(content="42")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def _assert_full_snapshot_signature_handled(events: List[dict]) -> None:
    _assert_deltas_match_their_block_type(events)
    # The full-text snapshot on the signature chunk must NOT be re-emitted as an
    # extra thinking_delta (it was already streamed incrementally) - otherwise the
    # client renders the reasoning twice.
    assert _thinking_deltas(events) == ["Let me ", "think about it."]
    assert "".join(_thinking_deltas(events)) == "Let me think about it."
    assert _signature_deltas(events) == ["sig-abc"]
    assert _text_deltas(events) == ["42"]


def test_full_thinking_snapshot_with_signature_emits_signature_only_sync():
    """Regression: a terminal thinking chunk carrying both the full thinking text
    and the signature used to raise ``ValueError`` (500) mid-stream, breaking every
    Claude Code request routed through the proxy with an extended-thinking model. It
    must instead emit a single ``signature_delta`` without duplicating the thinking.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_full_snapshot_signature_chunks()),
        model="claude-x",
    )
    _assert_full_snapshot_signature_handled(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_full_thinking_snapshot_with_signature_emits_signature_only_async():
    """Async twin - the proxy serves the async iterator, so the crash must be gone
    on that path too.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_full_snapshot_signature_chunks()),
        model="claude-x",
    )
    _assert_full_snapshot_signature_handled(await _drain_async(wrapper))


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


def _thinking_first_chunks() -> List[MagicMock]:
    return [
        _thinking_chunk("Let me think"),
        _thinking_chunk("about it."),
        _make_chunk(Delta(content="42")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def _assert_thinking_first_block_opens_at_index_zero(events: List[dict]) -> None:
    starts = [
        (e["index"], e["content_block"]["type"])
        for e in events
        if e.get("type") == "content_block_start"
    ]
    assert starts == [(0, "thinking"), (1, "text")], starts
    assert "" not in _text_deltas(events)
    assert _thinking_deltas(events) == ["Let me think", "about it."]
    assert _text_deltas(events) == ["42"]
    _assert_deltas_match_their_block_type(events)


def test_thinking_first_stream_opens_thinking_block_at_index_zero_sync():
    """Bug A regression: when the model's first output is reasoning the adapter
    must open the first content block as ``thinking`` at index 0. The previous
    code pre-emitted a hardcoded empty ``text`` block at index 0 before
    inspecting any upstream chunk, then opened ``thinking`` at index 1; strict
    Anthropic SDK clients with thinking enabled reject that stream with
    "Content block is not a thinking block".
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_thinking_first_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_thinking_first_stream_opens_thinking_block_at_index_zero_async():
    """Async twin of the Bug A regression; the proxy serves the async iterator,
    so the first block must be ``thinking`` at index 0 on this path too.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_thinking_first_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(await _drain_async(wrapper))


def _reasoning_content_chunk(reasoning: str) -> MagicMock:
    return _make_chunk(Delta(content=None, reasoning_content=reasoning))


def _reasoning_first_chunks() -> List[MagicMock]:
    return [
        _reasoning_content_chunk("Let me think"),
        _reasoning_content_chunk("about it."),
        _make_chunk(Delta(content="42")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def test_reasoning_content_first_stream_opens_thinking_block_at_index_zero_sync():
    """The reported backend (hosted_vllm; vLLM and SGLang reasoning parsers)
    surfaces reasoning as OpenAI ``reasoning_content`` with no
    ``thinking_blocks``. Such a stream must also open the first content block as
    ``thinking`` at index 0, exercising the reasoning_content branch of the
    chunk translator rather than the thinking_blocks branch the other twins use.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_reasoning_first_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(_drain_sync(wrapper))


def _blank_lead_chunks() -> List[MagicMock]:
    return [
        _make_chunk(Delta(content=None)),
        _thinking_chunk("Let me think"),
        _thinking_chunk("about it."),
        _make_chunk(Delta(content="42")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def _role_only_reasoning_content_lead_chunks() -> List[MagicMock]:
    return [
        _make_chunk(Delta(role="assistant", content=None, tool_calls=[])),
        _reasoning_content_chunk("Let me think"),
        _reasoning_content_chunk("about it."),
        _make_chunk(Delta(content="42")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]


def test_contentless_lead_chunk_does_not_open_text_block_before_thinking_sync():
    """OpenAI-compatible streaming backends open the response with a contentless
    priming chunk (an empty delta, e.g. the {role: assistant} lead-in) before the
    first real token. Such a lead chunk must NOT commit index 0 to an empty text
    block; the following thinking chunk must still open thinking at index 0, or
    strict Anthropic SDK clients reject the stream.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_blank_lead_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(_drain_sync(wrapper))


def test_role_only_lead_chunk_does_not_open_text_block_before_reasoning_content_sync():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_role_only_reasoning_content_lead_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_contentless_lead_chunk_does_not_open_text_block_before_thinking_async():
    """Async twin; the proxy serves the async iterator, so the contentless lead
    chunk must be skipped on this path too.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_blank_lead_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(await _drain_async(wrapper))


@pytest.mark.asyncio
async def test_role_only_lead_chunk_does_not_open_text_block_before_reasoning_content_async():
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_role_only_reasoning_content_lead_chunks()),
        model="claude-x",
    )
    _assert_thinking_first_block_opens_at_index_zero(await _drain_async(wrapper))


def test_finish_first_chunk_is_not_deferred_sync():
    """A stream whose first upstream chunk is already the finish event must not
    be skipped by the blank-delta deferral. ``_is_blank_delta`` returns False
    for a finish chunk so the message_delta still flows (with an empty text
    block opened and closed first); without that guard the deferral would drop
    the terminal event entirely.
    """
    chunks = [_make_chunk(Delta(content=None), finish_reason="stop")]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def _mixed_reasoning_and_text_chunks() -> List[MagicMock]:
    return [
        _make_chunk(Delta(content=None, reasoning_content="First thought.")),
        _make_chunk(
            Delta(content="Answer.", reasoning_content=" Last thought."),
            finish_reason="stop",
        ),
    ]


def _assert_mixed_reasoning_and_text_chunk_is_split(events: List[dict]) -> None:
    _assert_deltas_match_their_block_type(events)
    assert _thinking_deltas(events) == ["First thought.", " Last thought."]
    assert _text_deltas(events) == ["Answer."]
    assert [event["type"] for event in events].count("message_delta") == 1


def test_mixed_reasoning_and_text_chunk_is_split_sync():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_mixed_reasoning_and_text_chunks()),
        model="claude-x",
    )

    _assert_mixed_reasoning_and_text_chunk_is_split(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_mixed_reasoning_and_text_chunk_is_split_async():
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_mixed_reasoning_and_text_chunks()),
        model="claude-x",
    )

    _assert_mixed_reasoning_and_text_chunk_is_split(await _drain_async(wrapper))


def _mixed_chunk_with_tool_call() -> List[MagicMock]:
    return [
        _make_chunk(
            Delta(
                content="Answer.",
                reasoning_content="Thought.",
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        function=Function(name="get_weather", arguments='{"city": "NY"}'),
                        type="function",
                        index=0,
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
    ]


def _assert_each_payload_kind_emitted_once_in_anthropic_order(events: List[dict]) -> None:
    starts = [(e["index"], e["content_block"]["type"]) for e in events if e.get("type") == "content_block_start"]
    assert [block_type for _, block_type in starts] == ["thinking", "text", "tool_use"], starts
    assert _thinking_deltas(events) == ["Thought."]
    assert _text_deltas(events) == ["Answer."]
    assert _input_json_deltas(events) == ['{"city": "NY"}']
    assert [e["type"] for e in events].count("message_delta") == 1
    _assert_deltas_match_their_block_type(events)


def test_mixed_chunk_with_tool_call_emits_tool_use_once_sync():
    """A collapsed chunk carrying reasoning, text, AND a tool call must emit the
    tool_use block exactly once. The previous split cleared only the fields it
    knew about, so ``tool_calls`` survived on both pieces and the tool_use block
    (same id) was emitted twice; clients executed the tool twice or rejected the
    follow-up turn.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_mixed_chunk_with_tool_call()),
        model="claude-x",
    )
    _assert_each_payload_kind_emitted_once_in_anthropic_order(_drain_sync(wrapper))


@pytest.mark.asyncio
async def test_mixed_chunk_with_tool_call_emits_tool_use_once_async():
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(_mixed_chunk_with_tool_call()),
        model="claude-x",
    )
    _assert_each_payload_kind_emitted_once_in_anthropic_order(await _drain_async(wrapper))


def test_mixed_thinking_blocks_and_text_chunk_is_split_sync():
    """A mixed chunk whose reasoning arrives as ``thinking_blocks`` with no
    ``reasoning_content`` must split too. The previous predicate gated on
    ``reasoning_content`` only, so this shape skipped the split and emitted a
    ``thinking_delta`` inside a text block while dropping the answer text.
    """
    chunks = [
        _make_chunk(
            Delta(
                content="Answer.",
                thinking_blocks=[{"type": "thinking", "thinking": "Thought."}],
            )
        ),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _thinking_deltas(events) == ["Thought."]
    assert _text_deltas(events) == ["Answer."]
    _assert_deltas_match_their_block_type(events)


def test_mixed_chunk_with_both_reasoning_fields_keeps_text_sync():
    """LiteLLM bridges often set ``reasoning_content`` AND ``thinking_blocks``
    together. Both fields are one payload kind, so the split must emit the
    thinking once and still deliver the text; the previous split cleared only
    ``reasoning_content`` on the text piece, so the surviving ``thinking_blocks``
    won the translator's priority and the answer text was dropped.
    """
    chunks = [
        _make_chunk(
            Delta(
                content="Answer.",
                reasoning_content="Thought.",
                thinking_blocks=[{"type": "thinking", "thinking": "Thought."}],
            )
        ),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _thinking_deltas(events) == ["Thought."]
    assert _text_deltas(events) == ["Answer."]
    _assert_deltas_match_their_block_type(events)


def test_mixed_thinking_start_body_is_empty_and_thinking_not_doubled_sync():
    """SSE accumulators seed a block from the ``content_block_start`` body and
    append every delta, so a thinking start body that already carries the text
    doubles it client-side. A signature-less thinking_blocks piece must open
    with an empty body and deliver the text exactly once, via the delta.
    """
    chunks = [
        _make_chunk(
            Delta(
                content="Answer.",
                thinking_blocks=[{"type": "thinking", "thinking": "Thought.", "signature": ""}],
            )
        ),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    accumulated = ""
    for event in events:
        if event.get("type") == "content_block_start" and event["content_block"].get("type") == "thinking":
            assert not event["content_block"].get("thinking"), event["content_block"]
            accumulated += event["content_block"].get("thinking") or ""
        if event.get("type") == "content_block_delta" and event["delta"].get("type") == "thinking_delta":
            accumulated += event["delta"]["thinking"]
    assert accumulated == "Thought."
    assert _text_deltas(events) == ["Answer."]


def test_mixed_chunk_with_tool_argument_continuation_is_not_split_sync():
    """Streaming providers send a tool call's name only on its first chunk;
    later chunks carry argument fragments with ``name=None``. Splitting a
    mixed chunk around such a continuation would close the in-flight tool_use
    block mid-arguments and fabricate a second block with truncated JSON, so
    continuation chunks must pass through the splitter untouched.
    """
    chunks = [
        _tool_chunk("call_1", "get_weather", '{"ci'),
        _make_chunk(
            Delta(
                content="Answer.",
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=None,
                        function=Function(name=None, arguments='ty": "NY"}'),
                        type="function",
                        index=0,
                    )
                ],
            )
        ),
        _make_chunk(Delta(content=None), finish_reason="tool_calls"),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    starts = [e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"]
    assert starts.count("tool_use") == 1, starts
    assert "".join(_input_json_deltas(events)) == '{"city": "NY"}'


def test_multi_choice_mixed_chunk_is_not_split_sync():
    """The translators read every choice, so slicing a multi-choice chunk into
    per-kind pieces would drop or repeat the secondary choices' payload. A
    chunk with more than one choice must pass through the splitter untouched.
    """
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(content="Answer.", reasoning_content="Thought."),
            logprobs=None,
        ),
        StreamingChoices(
            finish_reason=None,
            index=1,
            delta=Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        function=Function(name="get_weather", arguments='{"city": "NY"}'),
                        type="function",
                        index=0,
                    )
                ],
            ),
            logprobs=None,
        ),
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    chunks = [chunk, _make_chunk(Delta(content=None), finish_reason="stop")]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    assert _input_json_deltas(events) == ['{"city": "NY"}']


def test_mixed_finish_chunk_emits_usage_once_sync():
    """Usage riding on a mixed finish chunk must surface exactly once, on the
    final ``message_delta``, never duplicated onto the intermediate pieces.
    """
    chunks = [
        _make_chunk(Delta(content=None, reasoning_content="T.")),
        _make_chunk(
            Delta(content="Hi", reasoning_content=" T2."),
            finish_reason="stop",
        ),
    ]
    chunks[1].usage = Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = _drain_sync(wrapper)

    message_deltas = [e for e in events if e.get("type") == "message_delta"]
    assert len(message_deltas) == 1
    assert message_deltas[0]["usage"]["output_tokens"] == 7
    assert _text_deltas(events) == ["Hi"]
    _assert_deltas_match_their_block_type(events)


class _CountingSyncStream:
    """Sync stream recording how many upstream chunks have been pulled."""

    def __init__(self, items: List[MagicMock]):
        self._items = list(items)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pulled >= len(self._items):
            raise StopIteration
        item = self._items[self.pulled]
        self.pulled += 1
        return item


class _CountingAsyncStream(_CountingSyncStream):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration


def _bedrock_tool_open_then_args() -> List[MagicMock]:
    """The Bedrock Converse shape: ``contentBlockStart`` names the tool and
    carries empty arguments, the arguments arrive in later events.
    """
    return [
        _tool_chunk("call_1", "Write", ""),
        _tool_chunk("call_1", None, '{"file_text":'),
        _tool_chunk("call_1", None, ' "hello"}'),
        _make_chunk(Delta(content=None), finish_reason="tool_calls"),
    ]


def test_tool_block_start_emitted_without_awaiting_the_next_chunk_sync():
    """Regression test for issue #32004.

    A tool_use block opened by a chunk whose delta is empty (Bedrock Converse
    sends the tool id/name and its arguments in separate events) must emit
    ``content_block_start`` off that chunk alone. Holding it until the next
    upstream chunk arrives means a provider that delivers tool arguments as a
    trailing burst leaves the client with nothing after ``message_start`` for
    the whole generation, tripping client and load-balancer idle timeouts.
    """
    stream = _CountingSyncStream(_bedrock_tool_open_then_args())
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="claude-x")

    assert next(wrapper)["type"] == "message_start"
    assert stream.pulled == 0

    start = next(wrapper)
    assert start["type"] == "content_block_start"
    assert start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "Write",
        "input": {},
    }
    assert stream.pulled == 1, (
        f"content_block_start was withheld until {stream.pulled} upstream chunks had arrived"
    )


@pytest.mark.asyncio
async def test_tool_block_start_emitted_without_awaiting_the_next_chunk_async():
    """Async twin of the sync regression test above (issue #32004)."""
    stream = _CountingAsyncStream(_bedrock_tool_open_then_args())
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="claude-x")

    assert (await wrapper.__anext__())["type"] == "message_start"
    assert stream.pulled == 0

    start = await wrapper.__anext__()
    assert start["type"] == "content_block_start"
    assert start["content_block"]["name"] == "Write"
    assert stream.pulled == 1, (
        f"content_block_start was withheld until {stream.pulled} upstream chunks had arrived"
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_tool_block_start_flush_does_not_duplicate_or_drop_events(is_async: bool):
    """Flushing the queued ``content_block_start`` early must not duplicate it,
    lose the empty opening delta's successors, or break event ordering.
    """
    chunks = _bedrock_tool_open_then_args()
    if is_async:
        wrapper = AnthropicStreamWrapper(completion_stream=_AsyncStream(chunks), model="claude-x")
        events = await _drain_async(wrapper)
    else:
        wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
        events = _drain_sync(wrapper)

    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert _input_json_deltas(events) == ['{"file_text":', ' "hello"}']
    _assert_deltas_match_their_block_type(events)

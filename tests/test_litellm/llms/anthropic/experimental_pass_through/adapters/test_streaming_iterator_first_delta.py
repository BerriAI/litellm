"""
Regression tests for issue #30014.

When LiteLLM proxies ``client -> /v1/messages -> /v1/chat/completions`` and a
streaming chunk both *triggers* a new Anthropic content block (its type differs
from the active block) and *carries* the first delta of that new block, the
trigger chunk's delta must be re-emitted as a ``content_block_delta``.

The synthesized ``content_block_start`` always carries an empty body, so before
the fix the first non-empty ``text_delta`` / ``thinking_delta`` of every
transitioned block was silently dropped — the client output started from the
second token (e.g. ``"Hi, how can I help you?"`` rendered as
``", how can I help you?"``). Bundled ``input_json_delta`` tool arguments were
already preserved and must stay preserved.
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
    StreamingChoices,
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


def _reasoning_chunk(reasoning: str) -> MagicMock:
    """OpenAI-compatible reasoning backends populate ``reasoning_content``."""
    return _make_chunk(Delta(role="assistant", content=None, reasoning_content=reasoning))


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
        if e.get("type") == "content_block_delta"
        and e["delta"].get("type") == "text_delta"
    ]


def _thinking_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e["delta"].get("type") == "thinking_delta"
    ]


def _input_json_deltas(events: List[dict]) -> List[str]:
    return [
        e["delta"]["partial_json"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e["delta"].get("type") == "input_json_delta"
    ]


def test_first_text_delta_after_thinking_is_not_dropped_sync():
    """The first text token of the answer triggers a thinking -> text
    transition; without the fix it was dropped, so the answer rendered as
    ``" Done."`` instead of ``"The answer is 42. Done."``.
    """
    chunks = [
        _reasoning_chunk("Let me think"),
        _reasoning_chunk(" hard."),
        _make_chunk(Delta(content="The answer is 42.")),
        _make_chunk(Delta(content=" Done.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    assert _thinking_deltas(events) == ["Let me think", " hard."]
    assert _text_deltas(events) == ["The answer is 42.", " Done."]
    assert "".join(_text_deltas(events)) == "The answer is 42. Done."


@pytest.mark.asyncio
async def test_first_text_delta_after_thinking_is_not_dropped_async():
    """Async path mirrors the sync regression — the proxy serves the async
    iterator, so it must preserve the first text delta too.
    """
    chunks = [
        _reasoning_chunk("Let me think"),
        _make_chunk(Delta(content="The answer is 42.")),
        _make_chunk(Delta(content=" Done.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(chunks), model="claude-x"
    )
    events = await _drain_async(wrapper)

    assert _text_deltas(events) == ["The answer is 42.", " Done."]
    assert "".join(_text_deltas(events)) == "The answer is 42. Done."


def test_first_thinking_delta_is_not_dropped_sync():
    """The first reasoning token triggers the initial text -> thinking
    transition and must not be dropped.
    """
    chunks = [
        _reasoning_chunk("Step one"),
        _reasoning_chunk(" step two."),
        _make_chunk(Delta(content="Answer.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    assert _thinking_deltas(events) == ["Step one", " step two."]
    assert "".join(_thinking_deltas(events)) == "Step one step two."


def test_empty_trigger_delta_is_not_re_emitted_sync():
    """A transition whose trigger chunk carries no content (empty text) must
    NOT produce a spurious empty ``content_block_delta`` — only the synthesized
    ``content_block_start`` is emitted for the new block.
    """
    chunks = [
        _reasoning_chunk("thinking"),
        # text block opens with an empty content chunk (role/keepalive style),
        # then the real text arrives in the following chunk.
        _make_chunk(Delta(content="")),
        _make_chunk(Delta(content="real text")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    # No empty-string text_delta should be present.
    assert "" not in _text_deltas(events)
    assert "".join(_text_deltas(events)) == "real text"


def test_multiple_text_deltas_after_thinking_preserved_sync():
    """Multiple-delta edge case: only the *first* text delta sits in the
    transition trigger chunk; the rest stream normally. All of them — leading
    one included — must reach the client in order.
    """
    chunks = [
        _reasoning_chunk("reasoning"),
        _make_chunk(Delta(content="Hi")),
        _make_chunk(Delta(content=", how ")),
        _make_chunk(Delta(content="can I help ")),
        _make_chunk(Delta(content="you?")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Hi", ", how ", "can I help ", "you?"]
    assert "".join(_text_deltas(events)) == "Hi, how can I help you?"


@pytest.mark.asyncio
async def test_first_thinking_delta_is_not_dropped_async():
    """Async variant: the first reasoning token opens the thinking block via a
    text -> thinking transition and must not be dropped.
    """
    chunks = [
        _reasoning_chunk("Step one"),
        _reasoning_chunk(" step two."),
        _make_chunk(Delta(content="Answer.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(chunks), model="claude-x"
    )
    events = await _drain_async(wrapper)

    assert _thinking_deltas(events) == ["Step one", " step two."]
    assert _text_deltas(events) == ["Answer."]


def test_text_after_tool_use_first_delta_preserved_sync():
    """A tool_use -> text transition (text resuming after a tool call) carries
    the resumed text's first token in the trigger chunk; it must be re-emitted.
    """
    chunks = [
        _make_chunk(Delta(content="Let me check.")),
        _make_chunk(
            Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        function=Function(name="get_weather", arguments='{"city":'),
                        type="function",
                        index=0,
                    )
                ],
            )
        ),
        _make_chunk(
            Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        function=Function(name=None, arguments=' "NY"}'),
                        type="function",
                        index=0,
                    )
                ],
            )
        ),
        _make_chunk(Delta(content="The weather is nice.")),
        _make_chunk(Delta(content=" Bye.")),
        _make_chunk(Delta(content=None), finish_reason="stop"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    assert _input_json_deltas(events) == ['{"city":', ' "NY"}']
    assert _text_deltas(events) == [
        "Let me check.",
        "The weather is nice.",
        " Bye.",
    ]


def test_bundled_tool_args_on_transition_still_preserved_sync():
    """Existing behavior guard: when the trigger chunk that opens a tool_use
    block also carries arguments (xAI/Gemini style), the ``input_json_delta``
    must still be emitted after ``content_block_start``.
    """
    chunks = [
        _make_chunk(Delta(content="Calling a tool.")),
        _make_chunk(
            Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        function=Function(
                            name="get_weather",
                            arguments='{"city": "NY"}',
                        ),
                        type="function",
                        index=0,
                    )
                ],
            )
        ),
        _make_chunk(Delta(content=None), finish_reason="tool_calls"),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks), model="claude-x"
    )
    events = _drain_sync(wrapper)

    assert _text_deltas(events) == ["Calling a tool."]
    assert _input_json_deltas(events) == ['{"city": "NY"}']

"""
Regression tests for AnthropicStreamWrapper dropping the first chunk of content
when a new content block is started.

Context:
Before the fix, `__next__` / `__anext__` emitted the sequence
`content_block_stop` -> `content_block_start` on a detected block transition
(e.g. text -> thinking, thinking -> text, text -> tool_use), but the `processed_chunk`
that actually *triggered* the transition was silently discarded. Because
`_translate_streaming_openai_chunk_to_anthropic_content_block()` returns a block
with an empty body for text transitions (`TextBlock(text="")`), dropping the
trigger chunk meant the first character(s) of every new block were lost on the
wire. For Bedrock Converse reasoning providers (MiniMax, Kimi, Claude extended
thinking), this typically manifested as responses starting mid-sentence or, if
the model emitted the text as a single chunk, as an empty text block with zero
`content_block_delta` events.

The fix enqueues the trigger chunk's `content_block_delta` after the
`content_block_start` so that no content is lost.
"""

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage


def _text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content=text, role="assistant"),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _thinking_chunk(thinking: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(
                    content="",
                    role="assistant",
                    reasoning_content=thinking,
                    thinking_blocks=[
                        {
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": None,
                        }
                    ],
                    provider_specific_fields={
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": thinking,
                                "signature": None,
                            }
                        ]
                    },
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _stop_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content="", role="assistant"),
                index=0,
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


class _MockStreamThinkingThenText:
    """Simulates a Bedrock Converse reasoning model stream.

    The first chunk is a thinking block (which forces a text->thinking transition
    on the wrapper's initial state) and the first text chunk forces a
    thinking->text transition. Both transition chunks carry real content that
    must not be dropped.
    """

    def __init__(self) -> None:
        self.responses: List[ModelResponseStream] = [
            _thinking_chunk("The user says hi. I should"),
            _thinking_chunk(" greet them back."),
            _text_chunk("안녕하세요, 저는 MiniMax입니다."),
            _text_chunk(" 무엇을 도와드릴까요?"),
            _stop_chunk(),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def _collect_deltas(chunks: List[dict]) -> dict:
    """Aggregate text_delta / thinking_delta content from a stream of wrapper chunks."""
    text = ""
    thinking = ""
    for chunk in chunks:
        if chunk.get("type") != "content_block_delta":
            continue
        delta = chunk.get("delta", {}) or {}
        if delta.get("type") == "text_delta":
            text += delta.get("text", "") or ""
        elif delta.get("type") == "thinking_delta":
            thinking += delta.get("thinking", "") or ""
    return {"text": text, "thinking": thinking}


EXPECTED_THINKING = "The user says hi. I should greet them back."
EXPECTED_TEXT = "안녕하세요, 저는 MiniMax입니다. 무엇을 도와드릴까요?"


def test_sync_first_chunk_preserved_on_block_transitions():
    """Sync path: the first chunk of each new content block must not be dropped."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_MockStreamThinkingThenText(),
        model="bedrock/converse/minimax.minimax-m2.5",
    )

    chunks = list(wrapper)
    aggregated = _collect_deltas(chunks)

    assert aggregated["thinking"] == EXPECTED_THINKING, (
        "Thinking content was dropped on the text->thinking transition. "
        f"Expected {EXPECTED_THINKING!r}, got {aggregated['thinking']!r}"
    )
    assert aggregated["text"] == EXPECTED_TEXT, (
        "Text content was dropped on the thinking->text transition. "
        f"Expected {EXPECTED_TEXT!r}, got {aggregated['text']!r}"
    )


@pytest.mark.asyncio
async def test_async_first_chunk_preserved_on_block_transitions():
    """Async path: same guarantee as the sync path."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_MockStreamThinkingThenText(),
        model="bedrock/converse/minimax.minimax-m2.5",
    )

    chunks: List[dict] = []
    async for chunk in wrapper:
        chunks.append(chunk)

    aggregated = _collect_deltas(chunks)

    assert aggregated["thinking"] == EXPECTED_THINKING
    assert aggregated["text"] == EXPECTED_TEXT


def test_sync_block_transition_event_ordering():
    """On a block transition the wrapper must emit
    content_block_stop -> content_block_start -> content_block_delta in that order,
    with the delta carrying the trigger chunk's content."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_MockStreamThinkingThenText(),
        model="bedrock/converse/minimax.minimax-m2.5",
    )

    chunks = list(wrapper)
    types = [c.get("type") for c in chunks]

    # Find the first text->thinking transition: content_block_stop then start then delta.
    # The wrapper emits an initial content_block_start(index=0,text) before the first
    # real chunk, so the first stop we see is the index=0 text block being closed.
    first_stop = types.index("content_block_stop")
    assert types[first_stop + 1] == "content_block_start"
    assert (
        types[first_stop + 2] == "content_block_delta"
    ), "First trigger chunk's delta must immediately follow content_block_start."

    # And the delta must carry the *first* trigger chunk's content verbatim -
    # i.e. the text that would otherwise have been dropped. This is the
    # specific regression this test guards against.
    delta_chunk = chunks[first_stop + 2]
    inner_delta = delta_chunk.get("delta", {}) or {}
    assert inner_delta.get("type") == "thinking_delta"
    assert inner_delta.get("thinking") == "The user says hi. I should", (
        "Expected the first thinking chunk's content to be emitted as the first "
        "delta after content_block_start, but got "
        f"{inner_delta.get('thinking')!r}"
    )

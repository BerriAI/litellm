"""The stream fold's dialect/event mismatch contract.

Wire dialects (openai, azure) fold ``wire_chunk`` events, gemini folds
composite ``chunk`` events, and only the block dialects (anthropic,
bedrock_converse) carry per-block deltas. A mismatched pairing is a
dialect/parser registration bug and must surface as a LOUD error value
through the fold, never a fabricated placeholder body (critic-google M5,
critic-azure M3).
"""

import pytest
from expression import Nothing, Some

from litellm.translation.engine.stream import fold_events
from litellm.translation.errors import TranslationError
from litellm.translation.inbound.openai_chat.stream import initial_state, step
from litellm.translation.ir import (
    CompositeChunk,
    JsonBlob,
    StreamEvent,
    ThinkingDelta,
)


def _thinking_event() -> StreamEvent:
    return StreamEvent.of_thinking_delta(ThinkingDelta(index=0, thinking="hmm"))


@pytest.mark.parametrize("dialect", ["openai", "azure", "gemini"])
def test_thinking_delta_on_non_block_dialect_is_a_loud_error(dialect) -> None:
    result = step(initial_state(model="m", dialect=dialect), _thinking_event())
    assert isinstance(result, TranslationError)
    assert "mismatch" in result.summary


@pytest.mark.parametrize("dialect", ["anthropic", "bedrock_converse", "gemini"])
def test_wire_chunk_on_non_wire_dialect_is_a_loud_error(dialect) -> None:
    event = StreamEvent.of_wire_chunk(JsonBlob(value={"choices": []}))
    result = step(initial_state(model="m", dialect=dialect), event)
    assert isinstance(result, TranslationError)
    assert "mismatch" in result.summary


@pytest.mark.parametrize(
    "dialect", ["anthropic", "bedrock_converse", "openai", "azure"]
)
def test_composite_chunk_on_non_gemini_dialect_is_a_loud_error(dialect) -> None:
    from expression.collections import Block

    event = StreamEvent.of_chunk(
        CompositeChunk(
            id="c1",
            text=Some("hi"),
            reasoning=Nothing,
            signatures=Block.empty(),
            tool_calls=Block.empty(),
            finish=Nothing,
            usage=Nothing,
        )
    )
    result = step(initial_state(model="m", dialect=dialect), event)
    assert isinstance(result, TranslationError)
    assert "mismatch" in result.summary


def test_fold_surfaces_the_mismatch_error() -> None:
    def parse(event):
        from expression import Ok

        return Ok(_thinking_event())

    folded = fold_events([{}], parse, initial_state(model="m", dialect="azure"))
    assert folded.is_error()
    assert "mismatch" in folded.error.summary

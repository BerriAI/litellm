"""
Tests for stream_chunk_builder handling of the raw 'reasoning' delta field.

Some providers (e.g. Scaleway) return {"delta": {"reasoning": "..."}} in
streaming chunks instead of {"delta": {"reasoning_content": "..."}}.

stream_chunk_builder must handle both field names so that:
  - Chunks with 'reasoning' are not silently treated as empty text chunks
    by the fast-path (is_simple_text_stream would wrongly stay True).
  - Reasoning content is accumulated into the final response's
    reasoning_content field regardless of which key the provider used.

Regression guard for: https://github.com/BerriAI/litellm/issues/27670
"""

import pytest

from litellm import stream_chunk_builder
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_delta_chunk(
    model: str = "test-model",
    reasoning_content: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> ModelResponseStream:
    """Build a chunk backed by a proper Delta object (reasoning_content key)."""
    delta = Delta(reasoning_content=reasoning_content, content=content)
    return ModelResponseStream(
        id="chatcmpl-test",
        created=1700000000,
        model=model,
        object="chat.completion.chunk",
        choices=[StreamingChoices(delta=delta, finish_reason=finish_reason, index=0)],
    )


def _make_raw_reasoning_chunk(
    model: str = "test-model",
    reasoning: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> ModelResponseStream:
    """Build a chunk where the delta is a *dict* with the raw 'reasoning' key.

    This simulates chunks created from providers like Scaleway that use
    {"delta": {"reasoning": "..."}} instead of {"reasoning_content": "..."}.
    """
    delta_dict: dict = {}
    if reasoning is not None:
        delta_dict["reasoning"] = reasoning
    if content is not None:
        delta_dict["content"] = content
    delta = Delta(**delta_dict)
    return ModelResponseStream(
        id="chatcmpl-test",
        created=1700000000,
        model=model,
        object="chat.completion.chunk",
        choices=[StreamingChoices(delta=delta, finish_reason=finish_reason, index=0)],
    )


# ---------------------------------------------------------------------------
# tests — Delta objects (reasoning_content key) — existing behaviour
# ---------------------------------------------------------------------------


def test_stream_chunk_builder_reasoning_content_accumulated():
    """Chunks with reasoning_content in Delta are accumulated correctly."""
    chunks = [
        _make_delta_chunk(reasoning_content="I need to think"),
        _make_delta_chunk(reasoning_content=" about this"),
        _make_delta_chunk(content="The answer is 42"),
        _make_delta_chunk(finish_reason="stop"),
    ]
    result = stream_chunk_builder(chunks=chunks)

    assert result is not None
    assert result.choices[0].message.content == "The answer is 42"
    assert result.choices[0].message.reasoning_content == "I need to think about this"


# ---------------------------------------------------------------------------
# tests — raw dict with 'reasoning' key (provider like Scaleway)
# ---------------------------------------------------------------------------


def test_stream_chunk_builder_raw_reasoning_not_treated_as_simple_stream():
    """
    A stream where the delta carries 'reasoning' (not 'reasoning_content')
    must NOT be classified as a simple text stream — otherwise reasoning
    chunks are silently discarded.
    """
    chunks = [
        _make_raw_reasoning_chunk(reasoning="I need to think"),
        _make_raw_reasoning_chunk(reasoning=" about this"),
        _make_raw_reasoning_chunk(content="The answer is 42"),
        _make_raw_reasoning_chunk(finish_reason="stop"),
    ]
    result = stream_chunk_builder(chunks=chunks)

    assert result is not None, "stream_chunk_builder must not return None"
    assert result.choices[0].message.content == "The answer is 42"
    assert (
        result.choices[0].message.reasoning_content == "I need to think about this"
    ), (
        "reasoning content from 'reasoning' field must be accumulated into "
        "reasoning_content on the final message"
    )


def test_stream_chunk_builder_raw_reasoning_only_stream():
    """Reasoning-only stream (no content) with raw 'reasoning' key works."""
    chunks = [
        _make_raw_reasoning_chunk(reasoning="step one"),
        _make_raw_reasoning_chunk(reasoning=" step two"),
        _make_raw_reasoning_chunk(finish_reason="stop"),
    ]
    result = stream_chunk_builder(chunks=chunks)

    assert result is not None
    assert result.choices[0].message.reasoning_content == "step one step two"


def test_stream_chunk_builder_mixed_reasoning_keys():
    """Chunks with both 'reasoning_content' and raw 'reasoning' are merged."""
    chunks = [
        _make_delta_chunk(reasoning_content="part A"),
        _make_raw_reasoning_chunk(reasoning=" part B"),
        _make_delta_chunk(content="answer"),
        _make_delta_chunk(finish_reason="stop"),
    ]
    result = stream_chunk_builder(chunks=chunks)

    assert result is not None
    assert result.choices[0].message.content == "answer"
    rc = result.choices[0].message.reasoning_content
    assert rc is not None
    assert "part A" in rc
    assert "part B" in rc

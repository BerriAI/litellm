"""
Tests for stream_chunk_builder reasoning item reassembly.

Previously, stream_chunk_builder dropped delta.reasoning_items entirely: the
simple-text fast path returned before any aggregation ran, and the full path
had no reasoning_items branch. The encrypted reasoning state a provider sends
back was therefore lost from the assembled message, which is what gets cached
and what the Responses API bridge reads to rebuild its input.
"""

from litellm import stream_chunk_builder
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(**delta_kwargs) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        created=1700000000,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(**delta_kwargs),
            )
        ],
    )


def test_reasoning_items_survive_a_text_only_stream():
    """
    A stream carrying only content and reasoning_items must keep the reasoning
    items. This is the fast path, which used to return before aggregating.
    """
    reasoning_item = {
        "id": "rs_abc123",
        "type": "reasoning",
        "encrypted_content": "ENCRYPTED-REASONING-BLOB",
        "summary": [{"type": "summary_text", "text": "Thinking"}],
    }

    chunks = [
        _chunk(content="Hello", role="assistant"),
        _chunk(content="", reasoning_items=[reasoning_item]),
    ]

    response = stream_chunk_builder(chunks, messages=MESSAGES)

    assert response is not None
    message = response.choices[0].message
    assert message.content == "Hello"
    assert getattr(message, "reasoning_items", None) == [reasoning_item]


def test_reasoning_items_are_merged_across_chunks():
    """
    Reasoning items arriving in more than one chunk are concatenated, matching
    how annotations and images are merged. thinking_blocks here also forces the
    full aggregation path rather than the text-only fast path.
    """
    item_a = {"id": "rs_a", "type": "reasoning", "encrypted_content": "BLOB-A"}
    item_b = {"id": "rs_b", "type": "reasoning", "encrypted_content": "BLOB-B"}

    chunks = [
        _chunk(
            content="Part one. ",
            role="assistant",
            thinking_blocks=[{"type": "thinking", "thinking": "step 1", "signature": "sig"}],
            reasoning_items=[item_a],
        ),
        _chunk(content="Part two.", reasoning_items=[item_b]),
    ]

    response = stream_chunk_builder(chunks, messages=MESSAGES)

    assert response is not None
    message = response.choices[0].message
    assert message.content == "Part one. Part two."
    assert getattr(message, "reasoning_items", None) == [item_a, item_b]


def test_no_reasoning_items_leaves_the_message_alone():
    """A stream without reasoning items must not grow a reasoning_items field."""
    chunks = [_chunk(content="Hello", role="assistant"), _chunk(content=" there")]

    response = stream_chunk_builder(chunks, messages=MESSAGES)

    assert response is not None
    message = response.choices[0].message
    assert message.content == "Hello there"
    assert getattr(message, "reasoning_items", None) is None

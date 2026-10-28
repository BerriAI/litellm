"""
Tests for stream_chunk_builder annotation merging.

Previously, stream_chunk_builder only took annotations from the FIRST
annotation chunk, losing any annotations that arrived in later chunks.
This fix merges annotations from ALL chunks.
"""

from litellm import stream_chunk_builder
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def test_stream_chunk_builder_merges_annotations_from_multiple_chunks():
    """
    stream_chunk_builder must merge annotations from ALL streaming chunks,
    not just take them from the first annotation chunk.

    Providers may spread annotations across multiple chunks (e.g. Gemini
    sends grounding metadata in the final chunk, while intermediate chunks
    may carry different annotations).
    """
    annotation_a = {
        "type": "url_citation",
        "url_citation": {
            "url": "https://example.com/a",
            "title": "Source A",
            "start_index": 0,
            "end_index": 10,
        },
    }
    annotation_b = {
        "type": "url_citation",
        "url_citation": {
            "url": "https://example.com/b",
            "title": "Source B",
            "start_index": 20,
            "end_index": 30,
        },
    }

    chunks = [
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Part one. ",
                        role="assistant",
                        annotations=[annotation_a],
                    ),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Part two."),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason="stop",
                    index=0,
                    delta=Delta(
                        content=None,
                        annotations=[annotation_b],
                    ),
                )
            ],
        ),
    ]

    response = stream_chunk_builder(chunks=chunks)
    assert response is not None

    message = response["choices"][0]["message"]
    assert message.annotations is not None
    assert len(message.annotations) == 2
    assert message.annotations[0] == annotation_a
    assert message.annotations[1] == annotation_b


def test_stream_chunk_builder_single_annotation_chunk_still_works():
    """
    When annotations come from a single chunk (most common case),
    stream_chunk_builder must still work correctly (no regression).
    """
    annotation = {
        "type": "url_citation",
        "url_citation": {
            "url": "https://example.com/only",
            "title": "Only Source",
            "start_index": 0,
            "end_index": 5,
        },
    }

    chunks = [
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Hello", role="assistant"),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason="stop",
                    index=0,
                    delta=Delta(content=None, annotations=[annotation]),
                )
            ],
        ),
    ]

    response = stream_chunk_builder(chunks=chunks)
    assert response is not None

    message = response["choices"][0]["message"]
    assert message.annotations is not None
    assert len(message.annotations) == 1
    assert message.annotations[0] == annotation


def test_stream_chunk_builder_no_annotations():
    """
    When no chunks contain annotations, the message should not have
    an annotations key (no regression).
    """
    chunks = [
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Hello", role="assistant"),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-test",
            created=1700000000,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason="stop",
                    index=0,
                    delta=Delta(content=None),
                )
            ],
        ),
    ]

    response = stream_chunk_builder(chunks=chunks)
    assert response is not None

    message = response["choices"][0]["message"]
    assert not hasattr(message, "annotations") or message.annotations is None

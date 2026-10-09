from typing import Final

from litellm import stream_chunk_builder
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

_CITATION_ONE: Final = {
    "type": "char_location",
    "cited_text": "The grass is green.",
    "document_index": 0,
    "document_title": "My Document",
    "start_char_index": 0,
    "end_char_index": 20,
}
_CITATION_TWO: Final = {
    "type": "char_location",
    "cited_text": "The sky is blue.",
    "document_index": 0,
    "document_title": "My Document",
    "start_char_index": 20,
    "end_char_index": 36,
}


def _chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-citations",
        created=1724900000,
        model="claude-opus-5",
        object="chat.completion.chunk",
        choices=[StreamingChoices(finish_reason=finish_reason, index=0, delta=delta)],
    )


def test_stream_chunk_builder_collects_every_streamed_citation():
    chunks: Final = [
        _chunk(Delta(content="The grass is green", role="assistant")),
        _chunk(Delta(content="", provider_specific_fields={"citation": _CITATION_ONE})),
        _chunk(Delta(content=" and the sky is blue.")),
        _chunk(Delta(content="", provider_specific_fields={"citation": _CITATION_TWO})),
        _chunk(Delta(content=""), finish_reason="stop"),
    ]

    response: Final = stream_chunk_builder(chunks=chunks)

    assert response is not None
    fields: Final = response.choices[0].message.provider_specific_fields
    assert fields is not None
    assert fields["citations"] == [[_CITATION_ONE, _CITATION_TWO]]
    assert "citation" not in fields
    assert response.choices[0].message.content == "The grass is green and the sky is blue."


def test_stream_chunk_builder_keeps_other_provider_fields_alongside_citations():
    thinking_blocks: Final = [{"type": "thinking", "thinking": "checking the document", "signature": "sig"}]
    chunks: Final = [
        _chunk(Delta(content="Green.", role="assistant")),
        _chunk(Delta(content="", provider_specific_fields={"citation": _CITATION_ONE})),
        _chunk(Delta(content="", provider_specific_fields={"thinking_blocks": thinking_blocks})),
        _chunk(Delta(content=""), finish_reason="stop"),
    ]

    response: Final = stream_chunk_builder(chunks=chunks)

    assert response is not None
    fields: Final = response.choices[0].message.provider_specific_fields
    assert fields is not None
    assert fields["citations"] == [[_CITATION_ONE]]
    assert fields["thinking_blocks"] == thinking_blocks
    assert "citation" not in fields


def test_stream_chunk_builder_without_citation_deltas_sets_no_citations_key():
    chunks: Final = [
        _chunk(Delta(content="Hello", role="assistant")),
        _chunk(Delta(content="", provider_specific_fields={"web_search_results": [{"url": "https://example.com"}]})),
        _chunk(Delta(content=""), finish_reason="stop"),
    ]

    response: Final = stream_chunk_builder(chunks=chunks)

    assert response is not None
    fields: Final = response.choices[0].message.provider_specific_fields
    assert fields is not None
    assert "citations" not in fields
    assert fields["web_search_results"] == [{"url": "https://example.com"}]


def test_stream_chunk_builder_keeps_block_list_citation_deltas_unnested():
    block_one: Final = [dict(_CITATION_ONE), dict(_CITATION_TWO)]
    block_two: Final = [dict(_CITATION_ONE)]
    chunks: Final = [
        _chunk(Delta(content="Green sky.", role="assistant")),
        _chunk(Delta(content="", provider_specific_fields={"citation": block_one})),
        _chunk(Delta(content="", provider_specific_fields={"citation": block_two})),
        _chunk(Delta(content=""), finish_reason="stop"),
    ]

    response: Final = stream_chunk_builder(chunks=chunks)

    assert response is not None
    fields: Final = response.choices[0].message.provider_specific_fields
    assert fields is not None
    assert fields["citations"] == [block_one, block_two]
    assert "citation" not in fields

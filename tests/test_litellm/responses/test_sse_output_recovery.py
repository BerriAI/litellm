"""Tests for litellm.responses.sse_output_recovery helpers."""

from litellm.responses.sse_output_recovery import (
    _MAX_CONTENT_INDEX,
    record_output_text_chunk,
    record_output_text_delta_chunk,
)


def test_text_chunk_with_oversized_content_index_is_dropped():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": _MAX_CONTENT_INDEX + 1,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    item = text_only_items[0]
    assert item["content"] == []


def test_text_chunk_with_negative_content_index_is_dropped():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": -1,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    assert text_only_items[0]["content"] == []


def test_text_chunk_at_max_content_index_is_recorded():
    output_items: dict = {}
    text_only_items: dict = {}
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": _MAX_CONTENT_INDEX,
            "text": "kept",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    content = text_only_items[0]["content"]
    assert len(content) == _MAX_CONTENT_INDEX + 1
    assert content[_MAX_CONTENT_INDEX]["text"] == "kept"


def test_text_delta_chunks_are_accumulated():
    output_items: dict = {}
    text_only_items: dict = {}

    record_output_text_delta_chunk(
        parsed_chunk={
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hel",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    record_output_text_delta_chunk(
        parsed_chunk={
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "lo",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"][0]["text"] == "Hello"


def test_text_done_replaces_accumulated_delta_with_final_text():
    output_items: dict = {}
    text_only_items: dict = {}

    record_output_text_delta_chunk(
        parsed_chunk={
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )
    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "final",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"][0]["text"] == "final"


def test_text_delta_with_invalid_delta_is_ignored():
    output_items: dict = {}
    text_only_items: dict = {}

    record_output_text_delta_chunk(
        parsed_chunk={
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": {"not": "text"},
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items == {}


def test_text_delta_does_not_override_output_item_done():
    output_items = {
        0: {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "final"}],
        }
    }
    text_only_items: dict = {}

    record_output_text_delta_chunk(
        parsed_chunk={
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": " ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items == {}
    assert output_items[0]["content"][0]["text"] == "final"


def test_text_chunk_without_indices_uses_next_available_item_and_content():
    output_items: dict = {}
    text_only_items: dict = {}

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "text": "fallback",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"][0]["text"] == "fallback"


def test_text_chunk_with_invalid_existing_content_is_ignored():
    output_items: dict = {}
    text_only_items = {0: {"type": "message", "content": "not-a-list"}}

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"] == "not-a-list"


def test_text_chunk_replaces_invalid_existing_content_item():
    output_items: dict = {}
    text_only_items = {0: {"type": "message", "content": [None]}}

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "recovered",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"][0]["text"] == "recovered"


def test_text_chunk_with_invalid_text_is_ignored():
    output_items: dict = {}
    text_only_items: dict = {}

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": None,
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items == {}


def test_text_chunk_does_not_override_output_item_done():
    output_items = {
        0: {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "authoritative"}],
        }
    }
    text_only_items: dict = {}

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "ignored",
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items == {}
    assert output_items[0]["content"][0]["text"] == "authoritative"


def test_text_chunk_preserves_annotations():
    output_items: dict = {}
    text_only_items: dict = {}
    annotations = [{"type": "url_citation", "url": "https://example.com"}]

    record_output_text_chunk(
        parsed_chunk={
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "with annotations",
            "annotations": annotations,
        },
        output_items=output_items,
        text_only_items=text_only_items,
    )

    assert text_only_items[0]["content"][0]["annotations"] == annotations

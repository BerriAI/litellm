"""Assembly contract for raw Anthropic SSE streams.

Four guardrails (bedrock, model_armor, tool_permission, thirdlaw) and the OTel request/response
recorder all scan the body this module assembles, so anything it drops is invisible to every one
of them.
"""

import json
from collections.abc import Sequence

from litellm.proxy.guardrails.anthropic_sse import anthropic_sse_chunks_from_body, assemble_anthropic_sse_body


def _frames(*events: tuple[str, dict[str, object]]) -> list[bytes]:
    return [f"event: {name}\ndata: {json.dumps(body)}\n\n".encode() for name, body in events]


def _message_start() -> tuple[str, dict[str, object]]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
    )


def _tool_block_start(index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
    )


def _input_delta(partial_json: str, index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def _assembled_tool_input(*events: tuple[str, dict[str, object]]) -> object:
    """The tool block as it lands in the native Anthropic body, which is the shape the /v1/messages
    scan is handed. The ModelResponse conversion keeps the raw partial JSON by another route, so
    asserting through that one would pass either way."""
    body = assemble_anthropic_sse_body(_frames(*events))
    assert body is not None
    content = body["content"]
    assert isinstance(content, Sequence)
    return content[0]["input"]


def test_complete_tool_arguments_assemble_as_the_object_the_model_sent():
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "weather'),
        _input_delta(' in sf"}'),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {"query": "weather in sf"}


def test_truncated_tool_arguments_keep_the_text_instead_of_dropping_it():
    """max_tokens can cut a tool argument mid-JSON. That fragment still reached the client, so
    discarding it here would hand a caller a way to smuggle text past every scanner above."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "sk-live-TRUNC'),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    assert "sk-live-TRUNC" in json.dumps(arguments)


def test_a_tool_block_with_no_arguments_assembles_as_an_empty_object():
    """A tool called with no arguments is not truncation, so it must not pick up a raw fragment."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {}


def _text_block_start(index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
    )


def _block_delta(delta: dict[str, object], index: int = 0) -> tuple[str, dict[str, object]]:
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def _assembled_content(*events: tuple[str, dict[str, object]]) -> Sequence[dict[str, object]]:
    body = assemble_anthropic_sse_body(_frames(*events))
    assert body is not None
    content = body["content"]
    assert isinstance(content, Sequence)
    return content


def test_citation_deltas_land_on_the_text_block_they_annotate():
    citation = {"type": "char_location", "cited_text": "the sky is blue", "document_index": 0}
    content = _assembled_content(
        _message_start(),
        _text_block_start(),
        _block_delta({"type": "text_delta", "text": "Blue."}),
        _block_delta({"type": "citations_delta", "citation": citation}),
        _block_delta({"type": "citations_delta"}),
        _block_delta({"type": "some_future_delta", "text": "ignored"}),
    )
    assert list(content) == [{"type": "text", "text": "Blue.", "citations": (citation,)}]


def test_a_block_type_with_no_deltas_keeps_its_start_payload():
    redacted = {"type": "redacted_thinking", "data": "opaque-blob"}
    content = _assembled_content(
        _message_start(),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": redacted}),
    )
    assert list(content) == [redacted]


def test_malformed_block_events_are_skipped_without_corrupting_the_rest():
    content = _assembled_content(
        _message_start(),
        ("content_block_start", {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
        _text_block_start(),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": "not-a-mapping"}),
        _block_delta({"type": "text_delta", "text": "kept"}),
    )
    assert list(content) == [{"type": "text", "text": "kept"}]


def test_an_undecodable_stream_assembles_to_nothing():
    """Nothing scannable comes out of bytes that are not UTF-8, so callers must treat the stream as
    unassembled rather than scanning a partial body."""
    frames = _frames(_message_start(), _text_block_start())
    assert assemble_anthropic_sse_body([*frames[:1], b"\xff\xfe\xfd", *frames[1:]]) is None


def test_a_replayed_body_keeps_its_citations_and_redacted_thinking():
    """A guardrail rewrite is replayed to the client from the body, so anything the replay drops is
    lost to the caller even though the scan saw it: citations lose their sources, and a
    redacted_thinking block without its data cannot be sent back on the next turn."""
    citation = {"type": "char_location", "cited_text": "the sky is blue", "document_index": 0}
    content = [
        {"type": "redacted_thinking", "data": "opaque-blob"},
        {"type": "text", "text": "The sky is blue.", "citations": [citation]},
        {"type": "text", "text": "No source here."},
    ]
    body: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 7},
    }
    replayed = assemble_anthropic_sse_body(list(anthropic_sse_chunks_from_body(body)))
    assert replayed is not None
    assert json.loads(json.dumps(replayed["content"])) == content

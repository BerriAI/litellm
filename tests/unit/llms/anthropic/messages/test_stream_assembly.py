import json
from typing import Final

import pytest

from litellm.llms.anthropic.messages.stream_assembly import (
    anthropic_message_delta_fields,
    anthropic_sse_error_events,
    assemble_anthropic_sse_stream,
    has_unreadable_json_event,
    is_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)
from litellm.types.utils import Choices


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


MESSAGE_ID: Final = "msg_upstream_1"
MODEL: Final = "claude-upstream"

ANTHROPIC_STREAM: Final = (
    _frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": MESSAGE_ID,
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ),
    _frame(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}},
    ),
    _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
    _frame(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ),
    _frame("message_stop", {"type": "message_stop"}),
)

SAFEGUARD_RESULTS: Final = [
    {
        "type": "dangerous_tool_use",
        "status": {
            "type": "available",
            "tool_uses": {
                "toolu_01": {"type": "evaluated", "outcome": "flagged", "explanation": "deletes the filesystem root"}
            },
        },
    }
]
SAFEGUARDED_STREAM: Final = (
    *ANTHROPIC_STREAM[:5],
    _frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "stop_details": None,
                "safeguard_results": SAFEGUARD_RESULTS,
            },
            "usage": {"output_tokens": 2},
        },
    ),
    _frame("message_stop", {"type": "message_stop"}),
)

GOOGLE_STREAM: Final = (b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n',)
ANTHROPIC_ERROR: Final = _frame("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
CHAT_ERROR: Final = b'data: {"error": {"message": "blocked by guardrail"}}\n\n'
INVALID_UTF8: Final = (b"\xff\xfe data",)
TOOL_CALL_OVER_TWO_DATA_LINES: Final = (
    b'event: content_block_start\ndata: {"type": "content_block_start", "index": 0,\n'
    b'data: "content_block": {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}}\n\n'
)


def _wrapped_in_ignored_fields(prefix: bytes) -> bytes:
    return (
        prefix + b'data: {"type": "error", "error": {}, "w": [\n'
        b'data: {"type": "content_block_start"}\n' + prefix + b"data: ]}\n\n"
    )


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((b"data: {}\n\n",), True),
        (("data: {}\n\n",), True),
        ((object(), b"data: {}\n\n"), True),
        ((object(),), False),
        ((), False),
    ],
)
def test_is_raw_sse_stream_is_true_only_when_a_chunk_is_str_or_bytes(chunks: tuple[object, ...], expected: bool):
    assert is_raw_sse_stream(chunks) is expected


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (ANTHROPIC_STREAM, True),
        ((ANTHROPIC_ERROR,), True),
        (tuple(chunk.decode() for chunk in ANTHROPIC_STREAM), True),
        (GOOGLE_STREAM, False),
        (INVALID_UTF8, False),
        ((), False),
    ],
)
def test_is_anthropic_sse_stream_decides_on_the_event_types_present(chunks: tuple[object, ...], expected: bool):
    assert is_anthropic_sse_stream(chunks) is expected


def test_assemble_anthropic_sse_stream_joins_the_text_deltas():
    assembled = assemble_anthropic_sse_stream(ANTHROPIC_STREAM)

    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "Hello there"


def test_assemble_anthropic_sse_stream_restores_upstream_identity_only_when_asked():
    plain = assemble_anthropic_sse_stream(ANTHROPIC_STREAM)
    restored = assemble_anthropic_sse_stream(ANTHROPIC_STREAM, restore_identity=True)

    assert plain is not None and restored is not None
    assert plain.id != MESSAGE_ID
    assert (restored.id, restored.model) == (MESSAGE_ID, MODEL)


def test_anthropic_message_delta_fields_reads_the_fields_the_openai_shape_drops():
    assert dict(anthropic_message_delta_fields(ANTHROPIC_STREAM)) == {}
    assert dict(anthropic_message_delta_fields(SAFEGUARDED_STREAM)) == {"safeguard_results": SAFEGUARD_RESULTS}
    assert dict(anthropic_message_delta_fields(GOOGLE_STREAM)) == {}


def test_anthropic_message_delta_fields_reads_a_message_delta_whose_json_holds_unicode_line_separators() -> None:
    safeguard_results: Final = [
        {
            "type": "dangerous_tool_use",
            "status": {
                "type": "available",
                "tool_uses": {
                    "toolu_01": {"type": "evaluated", "outcome": "flagged", "explanation": "deletes\u2028the\u2029root"}
                },
            },
        }
    ]
    message_delta: Final = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "safeguard_results": safeguard_results},
        "usage": {"output_tokens": 2},
    }
    raw_message_delta_frame: Final = f"event: message_delta\ndata: {json.dumps(message_delta, ensure_ascii=False)}\n\n"

    fields: Final = anthropic_message_delta_fields((*ANTHROPIC_STREAM[:5], raw_message_delta_frame.encode()))

    assert dict(fields) == {"safeguard_results": safeguard_results}


CUT_OFF_TOOL_USE: Final = (
    _frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_cut_off", "name": "Bash", "input": {}},
        },
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"command": "l'}},
    ),
)


@pytest.mark.parametrize(
    "chunks",
    [(*ANTHROPIC_STREAM[:4], ANTHROPIC_ERROR), (*ANTHROPIC_STREAM[:5], *CUT_OFF_TOOL_USE, ANTHROPIC_ERROR)],
    ids=["text-cut-off", "tool-input-cut-off"],
)
def test_stream_cut_off_by_an_upstream_error_assembles_its_text_only_through_the_error(chunks: tuple[bytes, ...]):
    assert assemble_anthropic_sse_stream(chunks) is None, "a caller recording what was served must not log a reply the upstream never finished"

    assembled: Final = assemble_anthropic_sse_stream(chunks, through_upstream_error=True)

    assert assembled is not None
    choice: Final = assembled.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "Hello there"
    assert not choice.message.tool_calls, "a tool call whose input the upstream cut off is not a call the client can run"


@pytest.mark.parametrize(
    "chunks",
    [GOOGLE_STREAM, INVALID_UTF8, (ANTHROPIC_ERROR,), ()],
    ids=["no-message-start", "undecodable", "error-only", "empty"],
)
def test_assemble_anthropic_sse_stream_returns_none_without_a_message_to_build(chunks: tuple[object, ...]):
    assert assemble_anthropic_sse_stream(chunks) is None


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((ANTHROPIC_ERROR,), True),
        ((CHAT_ERROR,), True),
        ((ANTHROPIC_ERROR, CHAT_ERROR), True),
        ((ANTHROPIC_ERROR, ANTHROPIC_STREAM[0]), False),
        ((object(), ANTHROPIC_ERROR), False),
        (ANTHROPIC_STREAM, False),
        (INVALID_UTF8, False),
        ((b"\n\n",), False),
        ((TOOL_CALL_OVER_TWO_DATA_LINES, ANTHROPIC_ERROR), False),
        ((b'data:\ndata: {"type": "message_start"}\n\n', ANTHROPIC_ERROR), False),
        ((b"data: some words\n\n", ANTHROPIC_ERROR), False),
        ((b"data: 5\n\n", CHAT_ERROR), False),
        ((b'data: {"type": "message_start", "error": {}}\n\n', ANTHROPIC_ERROR), False),
        ((b'event: error\ndata: {"type": "error",\ndata: "error": {"type": "overloaded_error"}}\n\n',), False),
        ((b": keepalive\n\n", b"event: ping\n\n", CHAT_ERROR, b"data: [DONE]\n\n"), True),
        ((b'event: error\ndata: {"type": "error", "error": {"type": "api_error"}, "request_id": "req_1"}\n\n',), True),
        ((_wrapped_in_ignored_fields(b" "),), False),
        ((_wrapped_in_ignored_fields("\u00a0".encode()),), False),
        ((b' data: {"type": "error", "error": {}}\n\n',), False),
        ((b'event: message_start\ndata: {"error": {}, "message": {"id": "m1"}}\n\n', ANTHROPIC_ERROR), False),
        ((b'data: {"error": {"code": 0}, "candidates": []}\n\n',), False),
        ((b'data: {"type": "error", "error": {}, "choices": []}\n\n',), False),
        ((b'id: 1\ndata: {"type": "error", "error": {}}\n\n',), False),
        ((b'\xef\xbb\xbfdata: {"type": "message_start"}\n\n', ANTHROPIC_ERROR), False),
        ((b'event: message_stop\ndata: {"error": {}}\n\n',), False),
        ((b"data:\n\n", ANTHROPIC_ERROR), False),
        ((b': x\x0cdata: {"type": "content_block_start"}\n\n', ANTHROPIC_ERROR), False),
        ((b'event: ping\x0cdata: {"type": "content_block_start"}\n\n', ANTHROPIC_ERROR), False),
    ],
    ids=[
        "anthropic-error",
        "chat-error",
        "both-error-forms",
        "error-then-message",
        "typed-chunk-mixed-in",
        "normal-stream",
        "undecodable",
        "no-events",
        "tool-call-over-two-data-lines",
        "message-start-behind-an-empty-data-line",
        "words-beside-an-error",
        "non-object-beside-an-error",
        "message-start-with-an-error-field",
        "error-over-two-data-lines",
        "keepalive-bare-event-and-done-marker",
        "error-with-a-request-id",
        "content-between-space-led-data-lines",
        "content-between-nbsp-led-data-lines",
        "space-led-error",
        "content-named-by-its-event-line",
        "error-key-beside-gemini-candidates",
        "typed-error-beside-chat-choices",
        "error-with-an-id-field",
        "content-behind-a-byte-order-mark",
        "error-under-a-content-event-name",
        "error-behind-an-empty-data-line",
        "content-behind-a-form-feed-in-a-comment",
        "content-behind-a-form-feed-in-an-event-line",
    ],
)
def test_is_sse_error_stream_is_true_only_for_frames_that_are_all_errors(chunks: tuple[object, ...], expected: bool):
    assert is_sse_error_stream(chunks) is expected


@pytest.mark.parametrize(
    "frame",
    [
        b'event: error\ndata: {"type": "error"}\ndata: {"type": "content_block_start"}\n\n',
        b'event: error\ndata: {"type": "error"}\x0cdata: {"type": "content_block_start"}\n\n',
        b'event: error\rdata: {"type": "error"}\r\rdata: {"type": "content_block_start"}\r\r',
        b'data: {"type": "error"}\n\n',
    ],
    ids=["second-data-line", "behind-a-form-feed", "glued-frame-with-cr-line-ends", "no-event-line"],
)
def test_error_event_is_the_event_name_and_the_data_line_assembly_read(frame: bytes):
    assert anthropic_sse_error_events((frame,)) == ('event: error\ndata: {"type": "error"}',)


def test_line_separators_inside_a_json_string_do_not_make_frames():
    text = "a\u2028\u2028data: 5\u2028\u2028event: error\u2028data: upstream broke\u2028\u2028b"
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    frame = f"event: content_block_delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n".encode()

    assert not has_unreadable_json_event((frame,))
    assert anthropic_sse_error_events((frame,)) == ()


@pytest.mark.parametrize("name", ["error\u2028", "error\u0085", "error\x0b", "error\x0c", "error\x1c", " error"])
def test_event_name_no_client_reads_as_error_is_no_error_event(name: str):
    assert anthropic_sse_error_events((f"event: {name}\n\n".encode(),)) == ()


@pytest.mark.parametrize(
    "frame",
    [
        "event: error\nevent: ping\n\n",
        "event: error\nevent:\n\n",
        "event: error\ndata: {}\nevent: foo\n\n",
        "event: error\revent: ping\r\r",
        "event: error",
        "event: error\n",
        'event: error\ndata: {"type":"error"}\n',
        'data: {"type":"error"}',
    ],
    ids=[
        "later-name",
        "later-empty-name",
        "later-name-after-data",
        "later-name-cr",
        "no-line-end",
        "no-blank-line",
        "data-with-no-blank-line",
        "typed-with-no-blank-line",
    ],
)
def test_frame_no_client_dispatches_as_an_error_is_no_error_event(frame: str):
    assert anthropic_sse_error_events((*ANTHROPIC_STREAM, frame.encode())) == ()


def test_a_later_event_name_of_error_is_an_error_event():
    assert anthropic_sse_error_events((b"event: ping\nevent: error\n\n",)) == ("event: error",)


@pytest.mark.parametrize("line_break", ["\u2028", "\u2029", "\u0085"], ids=["u2028", "u2029", "u0085"])
def test_line_break_in_an_error_message_keeps_the_json_whole_for_every_reader(line_break: str):
    error: Final = {"type": "error", "error": {"type": "overloaded_error", "message": f"Over{line_break}loaded"}}
    frame: Final = f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n".encode()

    name_line, data_line = anthropic_sse_error_events((frame,))[0].splitlines()

    assert name_line == "event: error"
    assert json.loads(data_line.removeprefix("data: ")) == error


def test_error_frame_data_that_is_not_json_stops_at_the_first_line_any_reader_sees():
    assert anthropic_sse_error_events(("event: error\ndata: Overloaded\u2028data: hidden\n\n".encode(),)) == (
        "event: error\ndata: Overloaded",
    )


def test_error_event_with_no_data_stays_a_bare_event_line():
    assert anthropic_sse_error_events((b"event: error\n\n",)) == ("event: error",)


@pytest.mark.parametrize(
    "error_frame",
    [ANTHROPIC_ERROR, b"event: error\ndata: upstream broke\n\n"],
    ids=["typed", "named-with-unreadable-data"],
)
def test_assembly_through_an_upstream_error_reads_past_the_error_frame(error_frame: bytes):
    chunks: Final = (*ANTHROPIC_STREAM[:4], error_frame)

    assert assemble_anthropic_sse_stream(chunks, through_upstream_error=True) is not None

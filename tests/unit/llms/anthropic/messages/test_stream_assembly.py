import json
from typing import Final

import pytest

from litellm.llms.anthropic.messages.stream_assembly import (
    anthropic_message_delta_fields,
    anthropic_sse_error_events,
    anthropic_sse_reads_alike,
    assemble_anthropic_sse_stream,
    collapse_pure_text_chunks,
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


TOOL_USE_STREAM: Final = (
    *ANTHROPIC_STREAM[:5],
    _frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_ls", "name": "Bash", "input": {}},
        },
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'}},
    ),
    _frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
    _frame(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
    ),
    _frame("message_stop", {"type": "message_stop"}),
)


@pytest.mark.parametrize(
    "non_object_frame",
    [
        b"data: 5\n\n",
        b"event: ping\ndata: null\n\n",
        b'data: "x"\n\n',
        b"data: [1]\n\n",
        b"data: some words\n\n",
        b"data: " + b"9" * 5000 + b"\n\n",
        b"data: " + b"[" * 100_000 + b"]" * 100_000 + b"\n\n",
    ],
    ids=["number", "null", "string", "array", "not-json", "huge-number", "deep-array"],
)
def test_assembling_a_tool_use_stream_skips_a_data_line_that_is_not_a_json_object(non_object_frame: bytes):
    assembled: Final = assemble_anthropic_sse_stream((*TOOL_USE_STREAM[:3], non_object_frame, *TOOL_USE_STREAM[3:]))

    assert assembled is not None
    choice: Final = assembled.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "Hello there"
    assert [(call.function.name, call.function.arguments) for call in choice.message.tool_calls or []] == [
        ("Bash", '{"command": "ls"}')
    ]


@pytest.mark.parametrize(
    "undecodable_object_frame",
    [
        b"data: " + b'{"a": ' * 100_000 + b"1" + b"}" * 100_000 + b"\n\n",
        b'event: ping\ndata: {"type": "ping", "n": ' + b"9" * 5000 + b"}\n\n",
    ],
    ids=["deep-object", "huge-number-in-object"],
)
def test_assembling_refuses_a_stream_with_an_object_event_too_deep_or_too_long_to_decode(
    undecodable_object_frame: bytes,
):
    assert (
        assemble_anthropic_sse_stream((*TOOL_USE_STREAM[:3], undecodable_object_frame, *TOOL_USE_STREAM[3:])) is None
    )


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


@pytest.mark.parametrize(
    "payload, unreadable",
    [
        (b"[" * 100_000 + b"]" * 100_000, True),
        (b'{"a": ' * 100_000 + b"1" + b"}" * 100_000, True),
        (b'{"type": "ping", "n": ' + b"9" * 5000 + b"}", True),
        (b"not json", False),
    ],
    ids=["deep-array", "deep-object", "huge-number-in-object", "not-json"],
)
def test_json_too_deep_or_too_long_to_decode_counts_as_unreadable(payload: bytes, unreadable: bool):
    assert has_unreadable_json_event((b"event: ping\ndata: " + payload + b"\n\n",)) is unreadable


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


def test_collapsing_a_text_run_rejects_a_delta_whose_text_is_not_a_string():
    chunks: Final = (
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
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": 7}},
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )

    with pytest.raises(TypeError, match="sequence item 1: expected str instance, int found"):
        collapse_pure_text_chunks(chunks)


def _one_block_stream(block: dict[str, object], delta: dict[str, object]) -> tuple[bytes, ...]:
    return (
        ANTHROPIC_STREAM[0],
        _frame("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        _frame("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}),
        _frame("message_stop", {"type": "message_stop"}),
    )


_TEXT_BLOCK: Final = {"type": "text", "text": ""}
_TOOL_USE_BLOCK: Final = {"type": "tool_use", "id": "toolu_1", "name": "call", "input": {}}
_THINKING_BLOCK: Final = {"type": "thinking", "thinking": "", "signature": ""}
_COMPACTION_BLOCK: Final = {"type": "compaction", "content": ""}
_DELTAS_WITH_THE_FIELD_THEIR_TYPE_NAMES: Final = (
    (_TEXT_BLOCK, {"type": "text_delta", "text": "hi"}),
    (_TOOL_USE_BLOCK, {"type": "input_json_delta", "partial_json": "{}"}),
    (_TEXT_BLOCK, {"type": "citations_delta", "citation": {"type": "char_location", "cited_text": "hi"}}),
    (_THINKING_BLOCK, {"type": "thinking_delta", "thinking": "hm"}),
    (_THINKING_BLOCK, {"type": "signature_delta", "signature": "sig"}),
    (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary"}),
)


@pytest.mark.parametrize(("block", "delta"), _DELTAS_WITH_THE_FIELD_THEIR_TYPE_NAMES)
def test_block_delta_carrying_the_field_its_type_names_reads_alike(
    block: dict[str, object], delta: dict[str, object]
):
    assert anthropic_sse_reads_alike(_one_block_stream(block, delta))


@pytest.mark.parametrize(
    ("block", "delta"),
    [
        (_TEXT_BLOCK, {"type": "text_delta", "text": "hi", "partial_json": "{}"}),
        (_THINKING_BLOCK, {"type": "signature_delta", "signature": "sig", "thinking": "hm"}),
        (_THINKING_BLOCK, {"type": "thinking_delta", "thinking": "hm", "signature": "sig"}),
        (_COMPACTION_BLOCK, {"type": "unknown_delta", "text": "hi"}),
    ],
    ids=["text-read-first", "thinking-read-beside-signature", "signature-read-beside-thinking", "unknown-type"],
)
def test_block_delta_whose_other_fields_assembly_reads_after_or_beside_its_own_reads_alike(
    block: dict[str, object], delta: dict[str, object]
):
    assert anthropic_sse_reads_alike(_one_block_stream(block, delta))


@pytest.mark.parametrize(
    ("block", "delta"),
    [
        (_TOOL_USE_BLOCK, {"type": "input_json_delta", "partial_json": "{}", "text": "hi"}),
        (_TEXT_BLOCK, {"type": "citations_delta", "citation": {}, "text": "hi"}),
        (_TEXT_BLOCK, {"type": "citations_delta", "citation": {}, "partial_json": "{}"}),
        (_THINKING_BLOCK, {"type": "thinking_delta", "thinking": "hm", "text": "hi"}),
        (_THINKING_BLOCK, {"type": "thinking_delta", "thinking": "hm", "partial_json": "{}"}),
        (_THINKING_BLOCK, {"type": "thinking_delta", "thinking": "hm", "citation": {}}),
        (_THINKING_BLOCK, {"type": "signature_delta", "signature": "sig", "text": "hi"}),
        (_THINKING_BLOCK, {"type": "signature_delta", "signature": "sig", "partial_json": "{}"}),
        (_THINKING_BLOCK, {"type": "signature_delta", "signature": "sig", "citation": {}}),
        (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary", "text": "hi"}),
        (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary", "partial_json": "{}"}),
        (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary", "citation": {}}),
        (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary", "thinking": "hm"}),
        (_COMPACTION_BLOCK, {"type": "compaction_delta", "content": "summary", "signature": "sig"}),
    ],
)
def test_block_delta_carrying_a_field_assembly_reads_before_its_own_reads_differently(
    block: dict[str, object], delta: dict[str, object]
):
    assert not anthropic_sse_reads_alike(_one_block_stream(block, delta))


def _block_start(index: object, block: dict[str, object]) -> bytes:
    return _frame("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def _tool_start(index: int) -> bytes:
    return _block_start(index, {"type": "tool_use", "id": f"toolu_{index}", "name": f"call_{index}", "input": {}})


def _delta(index: object, delta: dict[str, object]) -> bytes:
    return _frame("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def _text(index: int, text: str) -> bytes:
    return _delta(index, {"type": "text_delta", "text": text})


def _tool_input(index: int, partial_json: str) -> bytes:
    return _delta(index, {"type": "input_json_delta", "partial_json": partial_json})


def _stop(index: int) -> bytes:
    return _frame("content_block_stop", {"type": "content_block_stop", "index": index})


def _message(*block_frames: bytes) -> tuple[bytes, ...]:
    return (
        ANTHROPIC_STREAM[0],
        *block_frames,
        _frame("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}}),
        _frame("message_stop", {"type": "message_stop"}),
    )


_OVERLAPPED_BLOCKS: Final = {
    "every-call-started-before-any-input": _message(
        _block_start(0, _TEXT_BLOCK),
        _text(0, "Hello"),
        _tool_start(1),
        _tool_start(2),
        _tool_input(1, '{"n": 1}'),
        _stop(1),
        _tool_input(2, '{"n": 2}'),
        _stop(2),
    ),
    "text-stopped-after-the-calls": _message(
        _block_start(0, _TEXT_BLOCK),
        _text(0, "Hello"),
        _tool_start(1),
        _tool_start(2),
        _tool_input(2, '{"n": 2}'),
        _tool_input(1, '{"n": 1}'),
        _stop(2),
        _stop(1),
        _stop(0),
    ),
    "inputs-split-and-interleaved": _message(
        _block_start(0, _TEXT_BLOCK),
        _tool_start(1),
        _text(0, "Hello"),
        _tool_start(2),
        _tool_input(1, '{"n"'),
        _tool_input(2, '{"n": '),
        _tool_input(1, ": 1}"),
        _stop(0),
        _tool_input(2, "2}"),
        _stop(1),
        _stop(2),
    ),
}


@pytest.mark.parametrize("chunks", _OVERLAPPED_BLOCKS.values(), ids=_OVERLAPPED_BLOCKS)
def test_overlapped_blocks_read_alike_and_assemble_by_index(chunks: tuple[bytes, ...]):
    assert anthropic_sse_reads_alike(chunks)
    assembled: Final = assemble_anthropic_sse_stream(chunks)
    assert assembled is not None and isinstance(assembled.choices[0], Choices)
    message: Final = assembled.choices[0].message
    assert message.content == "Hello"
    assert [(call.id, call.function.name, call.function.arguments) for call in message.tool_calls or ()] == [
        ("toolu_1", "call_1", '{"n": 1}'),
        ("toolu_2", "call_2", '{"n": 2}'),
    ]


def test_a_delta_at_a_reused_index_assembles_onto_the_block_at_that_list_position_as_the_sdks_apply_it():
    assembled: Final = assemble_anthropic_sse_stream(
        _message(
            _block_start(0, _TEXT_BLOCK),
            _text(0, "Call "),
            _stop(0),
            _block_start(1, _TEXT_BLOCK),
            _text(1, "the office at "),
            _stop(1),
            _block_start(0, _TEXT_BLOCK),
            _text(0, "noon"),
            _stop(0),
        )
    )
    assert assembled is not None and isinstance(assembled.choices[0], Choices)
    assert assembled.choices[0].message.content == "Call noonthe office at "


_BLOCKS_CLIENTS_READ_DIFFERENTLY: Final = {
    "delta-after-its-block-stopped": _message(
        _tool_start(0), _tool_start(1), _stop(0), _tool_input(0, "{}"), _stop(1)
    ),
    "delta-before-its-block-starts": _message(_tool_start(0), _tool_input(1, "{}"), _tool_start(1), _stop(0), _stop(1)),
    "delta-of-a-block-that-never-starts": _message(_tool_start(0), _tool_input(1, "{}"), _stop(0)),
    "stop-before-its-block-starts": _message(_tool_start(0), _stop(1), _tool_start(1), _stop(0)),
    "block-stopped-twice": _message(_tool_start(0), _stop(0), _stop(0)),
    "block-started-twice": _message(_tool_start(0), _tool_start(0), _stop(0)),
    "blocks-started-out-of-order": _message(_tool_start(1), _tool_start(0), _stop(0), _stop(1)),
    "block-index-skipped": _message(_tool_start(0), _stop(0), _tool_start(2), _stop(2)),
    "index-that-is-a-bool": _message(_block_start(False, _TEXT_BLOCK), _stop(0)),
    "index-that-is-a-string": _message(_block_start("0", _TEXT_BLOCK), _stop(0)),
    "delta-without-an-index": _message(_block_start(0, _TEXT_BLOCK), _delta(None, {"type": "text_delta"}), _stop(0)),
    "text-back-in-an-earlier-block": _message(
        _block_start(0, _TEXT_BLOCK),
        _block_start(1, _TEXT_BLOCK),
        _text(0, "Call "),
        _text(1, "the office"),
        _text(0, "212-555"),
        _stop(0),
        _stop(1),
    ),
    "text-start-after-later-text": _message(
        _block_start(0, _TEXT_BLOCK),
        _block_start(1, {"type": "text", "text": "the office"}),
        _text(0, "Call "),
        _stop(0),
        _stop(1),
    ),
}


@pytest.mark.parametrize(
    "chunks", _BLOCKS_CLIENTS_READ_DIFFERENTLY.values(), ids=_BLOCKS_CLIENTS_READ_DIFFERENTLY
)
def test_blocks_out_of_the_order_clients_read_them_in_do_not_read_alike(chunks: tuple[bytes, ...]):
    assert not anthropic_sse_reads_alike(chunks)


_LONG_NUMBER: Final = b"9" * 5000
_OBJECTS_PYTHON_CANNOT_DECODE: Final = {
    "nested-too-deep": b'data: {"type":"ping","a":' + b'{"a":' * 100_000 + b"1" + b"}" * 100_001 + b"\n\n",
    "number-too-long": b'data: {"type":"ping","n":' + _LONG_NUMBER + b"}\n\n",
    "number-too-long-on-a-later-data-line": b'data: {"type":"ping","n":\ndata: ' + _LONG_NUMBER + b"}\n\n",
}


@pytest.mark.parametrize("frame", _OBJECTS_PYTHON_CANNOT_DECODE.values(), ids=_OBJECTS_PYTHON_CANNOT_DECODE)
def test_a_stream_with_an_object_python_cannot_decode_neither_reads_alike_nor_assembles(frame: bytes):
    chunks: Final = _message(_block_start(0, _TEXT_BLOCK), _text(0, "Hello"), _stop(0), frame)

    assert not anthropic_sse_reads_alike(chunks)
    assert assemble_anthropic_sse_stream(chunks) is None


@pytest.mark.parametrize("hidden_by", [b": note\x0c", b": note\xe2\x80\xa8"], ids=["form-feed", "line-separator"])
def test_an_object_python_cannot_decode_that_only_httpx_dispatches_does_not_read_alike(hidden_by: bytes):
    frame: Final = hidden_by + _OBJECTS_PYTHON_CANNOT_DECODE["number-too-long"]

    assert not anthropic_sse_reads_alike(_message(_block_start(0, _TEXT_BLOCK), _text(0, "Hello"), _stop(0), frame))


@pytest.mark.parametrize("line_break", [b"\x0c", " ".encode()], ids=["form-feed", "line-separator"])
def test_copies_of_a_delta_only_httpx_dispatches_do_not_read_alike(line_break: bytes):
    repeated: Final = _text(0, "1111 ")
    hidden: Final = b": note" + line_break + repeated.replace(b"\n", line_break, 1)

    assert anthropic_sse_reads_alike(
        _message(_block_start(0, _TEXT_BLOCK), _text(0, "4111 "), repeated, repeated, repeated, _stop(0))
    )
    assert not anthropic_sse_reads_alike(
        _message(_block_start(0, _TEXT_BLOCK), _text(0, "4111 "), repeated, hidden, hidden, _stop(0))
    )

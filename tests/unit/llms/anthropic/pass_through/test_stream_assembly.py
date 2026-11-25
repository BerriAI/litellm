import json
from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream,
    build_usage_only_response_from_chunks,
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

GOOGLE_STREAM: Final = (b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n',)
ANTHROPIC_ERROR: Final = _frame("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
CHAT_ERROR: Final = b'data: {"error": {"message": "blocked by guardrail"}}\n\n'
INVALID_UTF8: Final = (b"\xff\xfe data",)


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


NON_OBJECT_FRAMES: Final = (b"data: 5\n\n", b"event: ping\ndata: null\n\n", b'data: "x"\n\n', b"data: [1]\n\n")


@pytest.mark.parametrize("non_object_frame", NON_OBJECT_FRAMES)
def test_a_non_object_data_line_before_message_start_reads_like_the_stream_without_it(non_object_frame: bytes):
    chunks: Final = (non_object_frame, *ANTHROPIC_STREAM)

    assert is_anthropic_sse_stream(chunks) is True
    assert is_sse_error_stream(chunks) is False
    assembled: Final = assemble_anthropic_sse_stream(chunks)
    assert assembled is not None
    choice: Final = assembled.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "Hello there"


@pytest.mark.parametrize("non_object_frame", NON_OBJECT_FRAMES)
def test_usage_of_a_stream_that_ends_in_an_error_survives_a_non_object_data_line(non_object_frame: bytes):
    errored: Final = (*ANTHROPIC_STREAM[:3], ANTHROPIC_ERROR)
    expected: Final = build_usage_only_response_from_chunks(errored, model=MODEL)

    actual: Final = build_usage_only_response_from_chunks((*errored[:3], non_object_frame, ANTHROPIC_ERROR), model=MODEL)

    assert expected is not None
    assert actual is not None
    assert actual.usage == expected.usage


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
    ],
)
def test_is_sse_error_stream_is_true_only_for_frames_that_are_all_errors(chunks: tuple[object, ...], expected: bool):
    assert is_sse_error_stream(chunks) is expected

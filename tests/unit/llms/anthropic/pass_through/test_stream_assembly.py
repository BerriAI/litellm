import json
from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream,
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

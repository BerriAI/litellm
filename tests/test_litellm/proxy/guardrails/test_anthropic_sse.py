import json
from typing import Final

from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_frames,
    rewrite_text_blocks,
    text_block_texts,
)


def _frame(event_type: str, payload: dict, separator: str = "\n\n") -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}{separator}".encode()


def test_anthropic_sse_frames_round_trip_bytes() -> None:
    chunks: Final = (
        _frame(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "model": "claude", "content": [], "usage": {}}},
        ),
        _frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            separator="\r\n\r\n",
        ),
        _frame("ping", {"type": "ping"}),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            separator="\r\n\r\n",
        ),
        _frame("message_stop", {"type": "message_stop"}),
    )
    frames: Final = anthropic_sse_frames(chunks)
    assert frames is not None
    assert b"".join(frame.raw.encode() for frame in frames) == b"".join(chunks)
    assert frames[0].event is not None and frames[0].event["type"] == "message_start"
    assert dict(text_block_texts(frames)) == {0: "hi"}


def test_text_block_texts_joins_only_text_deltas() -> None:
    chunks: Final = (
        _frame(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "model": "claude", "content": [], "usage": {}}},
        ),
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think"},
            },
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame(
            "content_block_start",
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Contact "}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Jane"}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "citations_delta", "citation": {}}},
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 2}),
        _frame("message_stop", {"type": "message_stop"}),
    )
    frames: Final = anthropic_sse_frames(chunks)
    assert frames is not None
    assert dict(text_block_texts(frames)) == {2: "Contact Jane"}

    thinking_only: Final = anthropic_sse_frames(chunks[:5] + chunks[-1:])
    assert thinking_only is not None
    assert dict(text_block_texts(thinking_only)) == {}


def test_rewrite_text_blocks_keeps_crlf_line_endings_on_the_rewritten_frame() -> None:
    start: Final = _frame(
        "message_start",
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude", "content": [], "usage": {}}},
    )
    delta_frame: Final = (
        b"event: content_block_delta\r\n"
        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}\r\n'
        b"\r\n"
    )
    stop: Final = _frame("message_stop", {"type": "message_stop"})
    frames: Final = anthropic_sse_frames((start, delta_frame, stop))
    assert frames is not None
    emitted: Final = rewrite_text_blocks(frames, {0: "masked"})
    assert emitted == (
        start,
        b"event: content_block_delta\r\n"
        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "masked"}}\r\n'
        b"\r\n",
        stop,
    )

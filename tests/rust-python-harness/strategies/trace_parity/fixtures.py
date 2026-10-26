from __future__ import annotations

import base64
import binascii
import json
import struct
from collections.abc import Iterable, Mapping
from typing import Final

from ...shared.parity.recorded_http import (
    HttpHeader,
    RecordedHttpResponse,
    RecordedHttpStreamResponse,
    RecordedStreamChunk,
)

JSON_HEADERS: Final = (HttpHeader(name="content-type", value="application/json"),)
SSE_HEADERS: Final = (HttpHeader(name="content-type", value="text/event-stream"),)
AWS_EVENT_STREAM_HEADERS: Final = (HttpHeader(name="content-type", value="application/vnd.amazon.eventstream"),)


def json_response(body: Mapping[str, object] | bytes, *, status: int = 200) -> RecordedHttpResponse:
    encoded: Final = body if isinstance(body, bytes) else json.dumps(body).encode()
    return RecordedHttpResponse.from_bytes(status, JSON_HEADERS, encoded)


def sse_event(event: str, payload: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def sse_response(events: Iterable[tuple[str, Mapping[str, object]]]) -> RecordedHttpStreamResponse:
    return RecordedHttpStreamResponse(
        kind="http_stream",
        status_code=200,
        headers=SSE_HEADERS,
        chunks=tuple(RecordedStreamChunk.from_bytes(sse_event(event, payload)) for event, payload in events),
    )


def _aws_string_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", 7)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


def aws_event_stream_frame(payload: Mapping[str, object]) -> bytes:
    event_payload: Final = json.dumps(
        {"bytes": base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()},
        separators=(",", ":"),
    ).encode()
    headers: Final = (
        _aws_string_header(":event-type", "chunk")
        + _aws_string_header(":content-type", "application/json")
        + _aws_string_header(":message-type", "event")
    )
    total_length: Final = 12 + len(headers) + len(event_payload) + 4
    prelude: Final = struct.pack("!II", total_length, len(headers))
    prelude_crc: Final = binascii.crc32(prelude) & 0xFFFFFFFF
    prelude_crc_bytes: Final = struct.pack("!I", prelude_crc)
    message_crc: Final = binascii.crc32(prelude_crc_bytes + headers + event_payload, prelude_crc) & 0xFFFFFFFF
    return prelude + prelude_crc_bytes + headers + event_payload + struct.pack("!I", message_crc)


def aws_event_stream_response(
    events: Iterable[Mapping[str, object]], *, corrupt_last_frame: bool = False
) -> RecordedHttpStreamResponse:
    frames: Final = tuple(aws_event_stream_frame(event) for event in events)
    body: Final = (
        b"".join((*frames[:-1], frames[-1][:-1] + bytes((frames[-1][-1] ^ 0xFF,))))
        if corrupt_last_frame
        else b"".join(frames)
    )
    return RecordedHttpStreamResponse(
        kind="http_stream",
        status_code=200,
        headers=AWS_EVENT_STREAM_HEADERS,
        chunks=(RecordedStreamChunk.from_bytes(body),),
    )


def anthropic_response_body(*, model: str = "claude-sonnet-5") -> dict[str, object]:
    return {
        "id": "msg_trace",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }


def anthropic_stream_events(*, model: str = "claude-sonnet-5") -> tuple[tuple[str, Mapping[str, object]], ...]:
    return (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_trace",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def responses_body(*, model: str = "gpt-5", status: str = "completed") -> dict[str, object]:
    return {
        "id": "resp_trace",
        "object": "response",
        "created_at": 1_750_000_000,
        "status": status,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_trace",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }


def responses_stream_events(*, model: str = "gpt-5") -> tuple[tuple[str, Mapping[str, object]], ...]:
    response: Final = responses_body(model=model)
    return (
        (
            "response.created",
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_trace",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
        ),
        ("response.completed", {"type": "response.completed", "response": response}),
    )

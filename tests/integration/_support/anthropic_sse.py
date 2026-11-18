from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from integration._support.wire import Reply
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_EMPTY: Final[Mapping[str, JsonValue]] = MappingProxyType({})

ANTHROPIC_ERROR_TYPES: Final = MappingProxyType(
    {
        400: "invalid_request_error",
        401: "authentication_error",
        408: "api_error",
        409: "api_error",
        429: "rate_limit_error",
        500: "api_error",
        503: "api_error",
        529: "overloaded_error",
    }
)
LIFECYCLE: Final = (
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
)


def sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def message_start(message_id: str, model: str) -> bytes:
    return sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        },
    )


def text_delta(text: str) -> bytes:
    return sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


PING: Final = sse("ping", {"type": "ping"})
CONTENT_BLOCK_START: Final = sse(
    "content_block_start",
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
)
CONTENT_TAIL: Final = (
    sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    + sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
    )
    + sse("message_stop", {"type": "message_stop"})
)


def message_stream(message_id: str, model: str, text: str) -> tuple[bytes, bytes, bytes, bytes]:
    return (message_start(message_id, model), CONTENT_BLOCK_START, text_delta(text), CONTENT_TAIL)


def message_json(message_id: str, model: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    ).encode()


def error_frame(status: int, message: str) -> bytes:
    return sse("error", {"type": "error", "error": {"type": ANTHROPIC_ERROR_TYPES[status], "message": message}})


def error_body(status: int, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": ANTHROPIC_ERROR_TYPES[status], "message": message}}).encode()


DROP_PAUSE: Final = 0.2


def stream_reply(chunks: tuple[bytes, ...], *, abort_after: int | None = None, pause: float = 0) -> Reply:
    return Reply(content_type="text/event-stream", chunks=chunks, abort_after=abort_after, pause_between_chunks=pause)


def dropping_reply(chunks: tuple[bytes, ...], *, abort_after: int) -> Reply:
    return stream_reply(chunks, abort_after=abort_after, pause=DROP_PAUSE if abort_after else 0)


def status_reply(status: int) -> Reply:
    return Reply(status=status, body=error_body(status, f"scripted {status}"))


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str
    data: Mapping[str, JsonValue]


def _parse_block(block: str) -> SseEvent:
    lines: Final = block.splitlines()
    event: Final = next((line.removeprefix("event:").strip() for line in lines if line.startswith("event:")), "")
    data: Final = "".join(line.removeprefix("data:").strip() for line in lines if line.startswith("data:"))
    return SseEvent(event, _JSON_OBJECT.validate_json(data) if data else _EMPTY)


def _is_event_block(block: str) -> bool:
    return bool(block.strip()) and block.strip() != "data: [DONE]"


def parse_sse(text: str) -> tuple[SseEvent, ...]:
    return tuple(_parse_block(block) for block in text.replace("\r\n", "\n").split("\n\n") if _is_event_block(block))


def event_type(event: SseEvent) -> str:
    return event.event or str(event.data.get("type", ""))


def event_types(events: tuple[SseEvent, ...]) -> tuple[str, ...]:
    return tuple(event_type(event) for event in events)


def message_id(events: tuple[SseEvent, ...]) -> str:
    start: Final = next(event for event in events if event.event == "message_start")
    return str(_JSON_OBJECT.validate_python(start.data["message"])["id"])


def delta_text(events: tuple[SseEvent, ...]) -> str:
    deltas: Final = tuple(
        _JSON_OBJECT.validate_python(event.data["delta"]) for event in events if event.event == "content_block_delta"
    )
    return "".join(str(delta.get("text", "")) for delta in deltas)


def error_type(events: tuple[SseEvent, ...]) -> str | None:
    error: Final = next((event for event in events if event.event == "error"), None)
    if error is None:
        return None
    return str(_JSON_OBJECT.validate_python(error.data["error"])["type"])


def user_prompt(body: Mapping[str, JsonValue]) -> str:
    content: Final = _MESSAGES.validate_python(body["messages"])[0]["content"]
    assert isinstance(content, str), content
    return content


class Attempts:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._seen: Final = Counter[str]()

    def record(self, marker: str) -> int:
        with self._lock:
            self._seen[marker] += 1
            return self._seen[marker]

    def count(self, marker: str) -> int:
        with self._lock:
            return self._seen[marker]

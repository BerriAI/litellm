from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from multiprocessing.sharedctypes import Synchronized
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote

from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
REASONING_EFFORTS: Final = frozenset(("none", "minimal", "low", "medium", "high", "xhigh"))
NATIVE_CHAT: Final = "/openai/v1/chat/completions"
NATIVE_RESPONSES: Final = "/openai/v1/responses"
PNG_1X1: Final = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
)
USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "prompt_tokens": 9,
        "completion_tokens": 5,
        "total_tokens": 14,
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
)
_STATUS: Final = re.compile(r"status=(\d{3})")
_CONVERSE: Final = re.compile(r"^/model/(.+)/converse$")
_CONVERSE_STREAM: Final = re.compile(r"^/model/(.+)/converse-stream$")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NO_MARKER: Final = "0" * 32


def marker_of(request: Request) -> str:
    found: Final = MARKER.search(request.body.decode(errors="replace"))
    return _NO_MARKER if found is None else found.group(1)


def body_of(request: Request) -> Mapping[str, JsonValue]:
    try:
        return _JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


def target_of(request: Request) -> str:
    return unquote(request.target)


def answer(marker: str) -> str:
    return f"answer marker-{marker}"


def reasoning_answer(marker: str) -> str:
    return f"<reasoning>why marker-{marker}</reasoning> {answer(marker)}"


def _headers(marker: str) -> Mapping[str, str]:
    return MappingProxyType({"x-amzn-requestid": marker})


def _json_reply(status: int, payload: Mapping[str, JsonValue], marker: str) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode(), headers=_headers(marker))


def _error(status: int, message: str, marker: str) -> Reply:
    return _json_reply(status, {"message": message}, marker)


def _effort_of(target: str, body: Mapping[str, JsonValue]) -> JsonValue:
    if not _CONVERSE.match(target) and not _CONVERSE_STREAM.match(target):
        return body.get("reasoning_effort")
    fields: Final = body.get("additionalModelRequestFields")
    reasoning: Final = fields.get("reasoning") if isinstance(fields, Mapping) else None
    return reasoning.get("effort") if isinstance(reasoning, Mapping) else None


def forwarded_effort(request: Request) -> JsonValue:
    return _effort_of(target_of(request), body_of(request))


def _sse(frames: tuple[Mapping[str, JsonValue], ...], pause: float) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
        pause_between_chunks=pause,
    )


def _with_headers(reply: Reply, marker: str) -> Reply:
    return Reply(
        status=reply.status,
        body=reply.body,
        content_type=reply.content_type,
        chunks=reply.chunks,
        abort_after=reply.abort_after,
        gate_after_first=reply.gate_after_first,
        pause_between_chunks=reply.pause_between_chunks,
        headers=_headers(marker),
    )


def _content_deltas(model: str, marker: str) -> tuple[str, ...]:
    if "gpt-oss" in model:
        return ("<reason", "ing>why ", f"marker-{marker}", "</reas", "oning> answer ", f"marker-{marker}")
    return ("answer ", f"marker-{marker}")


def _chat_text(model: str, marker: str) -> str:
    return reasoning_answer(marker) if "gpt-oss" in model else answer(marker)


def _chat_reply(model: str, marker: str, stream: bool, pause: float) -> Reply:
    identity: Final = f"chatcmpl-{marker}"
    if not stream:
        return _json_reply(
            200,
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _chat_text(model, marker)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": dict(USAGE),
            },
            marker,
        )
    deltas: Final = _content_deltas(model, marker)
    frames: Final = tuple(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": delta}, "finish_reason": None}],
        }
        for delta in deltas
    )
    finish: Final[Mapping[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": dict(USAGE),
    }
    return _with_headers(_sse((*frames, finish), pause), marker)


def _responses_reply(model: str, marker: str, stream: bool, pause: float) -> Reply:
    identity: Final = f"resp_upstream_{marker}"
    item_id: Final = f"msg_{marker}"
    response: Final[Mapping[str, JsonValue]] = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": answer(marker), "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
    }
    if not stream:
        return _json_reply(200, response, marker)
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": answer(marker),
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
        pause_between_chunks=pause,
        headers=_headers(marker),
    )


def _converse_reply(marker: str) -> Reply:
    return _json_reply(
        200,
        {
            "output": {"message": {"role": "assistant", "content": [{"text": answer(marker)}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14},
            "metrics": {"latencyMs": 1},
        },
        marker,
    )


def _converse_stream_reply(marker: str, pause: float) -> Reply:
    events: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": "answer "}, "contentBlockIndex": 0}),
        ("contentBlockDelta", {"delta": {"text": f"marker-{marker}"}, "contentBlockIndex": 0}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14}, "metrics": {"latencyMs": 1}}),
    )
    return Reply(
        content_type=EVENT_STREAM,
        chunks=tuple(_aws_event_frame(kind, payload, "sc", marker) for kind, payload in events),
        pause_between_chunks=pause,
        headers=_headers(marker),
    )


def respond(request: Request, *, pause: float = 0.0) -> Reply:
    target: Final = target_of(request)
    marker: Final = marker_of(request)
    if request.method == "GET":
        if target == "/image.png":
            return Reply(body=PNG_1X1, content_type="image/png", headers=_headers(marker))
        return _error(404, f"no scripted object at {target}", marker)
    scripted_status: Final = _STATUS.search(request.body.decode(errors="replace"))
    if scripted_status is not None:
        status: Final = int(scripted_status.group(1))
        return _error(status, f"scripted {status}", marker)
    body: Final = body_of(request)
    effort: Final = _effort_of(target, body)
    if effort is not None and (not isinstance(effort, str) or effort not in REASONING_EFFORTS):
        return _error(400, f"Invalid reasoning effort: {json.dumps(effort)}", marker)
    model: Final = str(body.get("model", ""))
    stream: Final = body.get("stream") is True
    if request.method == "POST" and target == NATIVE_CHAT:
        return _chat_reply(model, marker, stream, pause)
    if request.method == "POST" and target == NATIVE_RESPONSES:
        return _responses_reply(model, marker, stream, pause)
    if request.method == "POST" and _CONVERSE.match(target):
        return _converse_reply(marker)
    if request.method == "POST" and _CONVERSE_STREAM.match(target):
        return _converse_stream_reply(marker, pause)
    return _error(404, f"unknown bedrock route {request.method} {target}", marker)


def serve_peer(port: int, received: Synchronized[int], answer_first: int) -> None:
    held: Final = threading.Event()

    def respond_or_hold(request: Request) -> Reply:
        with received.get_lock():
            received.value += 1
            ordinal: Final = received.value
        if ordinal > answer_first:
            held.wait()
        return respond(request)

    with wire_server(respond_or_hold, port=port):
        threading.Event().wait()

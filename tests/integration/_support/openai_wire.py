from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

from tests.integration._support.provider import MODEL_DISCOVERY

_USAGE: Final = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def answering_model_discovery(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def guarded(request: Request) -> Reply:
        if (request.method, request.target) == MODEL_DISCOVERY:
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        return respond(request)

    return guarded


def posted_targets(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target for request in wire.drain() if request.method == "POST")


def openai_error(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": f"scripted {status}", "type": "server_error", "code": None}}).encode(),
    )


def _data_frame(frame: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _typed_frame(event: dict[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def chat_stream(identity: str, model: str, text: str) -> tuple[bytes, bytes, bytes]:
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": model}
    role_only: Final = _data_frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
    content: Final = _data_frame({**chunk, "choices": [{"index": 0, "delta": {"content": text}}]})
    finish: Final = _data_frame(
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": _USAGE}
    )
    return (role_only, content, finish + b"data: [DONE]\n\n")


def chat_reply(identity: str, model: str, text: str, *, stream: bool) -> Reply:
    if stream:
        return Reply(content_type="text/event-stream", chunks=chat_stream(identity, model, text))
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": _USAGE,
            }
        ).encode()
    )


def _response_object(identity: str, model: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [_message_item(identity, text, "completed")],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }


def _message_item(identity: str, text: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}] if status == "completed" else [],
    }


def responses_stream(identity: str, model: str, text: str) -> tuple[bytes, bytes, bytes]:
    response: Final = _response_object(identity, model, text)
    item: Final = _message_item(identity, text, "in_progress")
    opened: Final = _typed_frame(
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        }
    ) + _typed_frame({"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": item})
    delta: Final = _typed_frame(
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
    )
    closed: Final = _typed_frame(
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 0,
            "item": _message_item(identity, text, "completed"),
        }
    ) + _typed_frame({"type": "response.completed", "sequence_number": 4, "response": response})
    return (opened, delta, closed)


def responses_reply(identity: str, model: str, text: str, *, stream: bool) -> Reply:
    if stream:
        return Reply(content_type="text/event-stream", chunks=responses_stream(identity, model, text))
    return Reply(body=json.dumps(_response_object(identity, model, text)).encode())

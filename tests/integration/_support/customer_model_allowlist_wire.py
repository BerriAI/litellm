import json
import uuid
from threading import Event
from typing import Final

from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import string_value
from tests.integration._support.wire import Reply, Request

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_RESPONSE_TEXT: Final = "scripted response"


def _response_reply(body: dict[str, JsonValue], gate_after_first: Event | None) -> Reply:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    item_id: Final = f"msg_{identity}"
    model: Final = string_value(body["model"])
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _RESPONSE_TEXT, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 2},
    }
    if body.get("stream") is not True:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
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
            "delta": _RESPONSE_TEXT,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
        gate_after_first=gate_after_first,
    )


def _chat_reply(body: dict[str, JsonValue], gate_after_first: Event | None) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    model: Final = string_value(body["model"])
    if body.get("stream") is True:
        head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": model}
        chunks: Final = (
            b"data: "
            + json.dumps(
                {
                    **head,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "scripted "}}],
                }
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {
                    **head,
                    "choices": [{"index": 0, "delta": {"content": "response"}, "finish_reason": "stop"}],
                }
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {
                    **head,
                    "choices": [],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                }
            ).encode()
            + b"\n\n",
            b"data: [DONE]\n\n",
        )
        return Reply(content_type="text/event-stream", chunks=chunks, gate_after_first=gate_after_first)
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _RESPONSE_TEXT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }
        ).encode()
    )


def customer_model_allowlist_reply(request: Request, gate_after_first: Event | None = None) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if request.target.split("?", maxsplit=1)[0].endswith("/responses"):
        return _response_reply(body, gate_after_first)
    if request.target.split("?", maxsplit=1)[0].endswith("/embeddings"):
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.125, 0.25]}],
                    "model": string_value(body["model"]),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
        )
    return _chat_reply(body, gate_after_first)

import json
from typing import Final

import litellm
from tests._support.provider import SharedProvider
from tests._support.wire import Reply
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.routing import call_public

RESPONSES_MODEL: Final = "openai/gpt-6.1-sol"
RESPONSES_TEXT: Final = "Hello from native Responses"
RESPONSES_RESPONSE: Final = {
    "id": "resp_native",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": RESPONSES_MODEL.removeprefix("openai/"),
    "output": [
        {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": RESPONSES_TEXT, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
}
RESPONSES_EVENTS: Final = (
    (
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**RESPONSES_RESPONSE, "status": "in_progress", "output": []},
        },
    ),
    (
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "delta": RESPONSES_TEXT,
        },
    ),
    ("response.completed", {"type": "response.completed", "sequence_number": 2, "response": RESPONSES_RESPONSE}),
)
RESPONSES_SSE: Final = tuple(
    f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in RESPONSES_EVENTS
)


def responses_reply(body: object = RESPONSES_RESPONSE, status: int = 200) -> Reply:
    return Reply(body=json.dumps(body).encode(), status=status)


def _error(kind: str) -> dict[str, object]:
    return {"error": {"type": kind, "message": kind, "param": None, "code": None}}


REQUEST: Final = {"model": RESPONSES_MODEL, "input": "Hello"}
BAD_REQUEST: Final = _error("invalid_request_error")
RATE_LIMITED: Final = _error("rate_limit_exceeded")


async def call(server: SharedProvider, *, asynchronous: bool = True, **kwargs: object) -> object:
    return await call_public(
        litellm.responses,
        litellm.aresponses,
        REQUEST,
        server,
        asynchronous=asynchronous,
        **kwargs,
    )


SYNC_DEPLOYMENT_HOOK: Final = gap("native sync calls skip the async pre-call deployment hook")
STREAMING: Final = gap("native Responses streaming declines to Python", raises=RuntimeError)
BODY_HOOKS: Final = gap("native Responses does not run the handler body hooks")

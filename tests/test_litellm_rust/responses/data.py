from typing import Final

import litellm
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.support.callbacks import Surface, gap

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


def _error(kind: str) -> dict[str, object]:
    return {"error": {"type": kind, "message": kind, "param": None, "code": None}}


RESPONSES_SURFACE: Final = Surface(
    route=Route.RESPONSES,
    request={"model": RESPONSES_MODEL, "input": "Hello"},
    response=RESPONSES_RESPONSE,
    events=RESPONSES_EVENTS,
    bad_request=_error("invalid_request_error"),
    rate_limited=_error("rate_limit_exceeded"),
    create=litellm.responses,
    acreate=litellm.aresponses,
)

SYNC_DEPLOYMENT_HOOK: Final = gap("native sync calls skip the async pre-call deployment hook")
STREAMING: Final = gap("native Responses streaming declines to Python", raises=RuntimeError)
BODY_HOOKS: Final = gap("native Responses does not run the handler body hooks")

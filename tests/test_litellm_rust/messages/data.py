from typing import Final

import litellm
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import call_natively

MESSAGES_MODEL: Final = "anthropic/claude-sonnet-5"
MESSAGES: Final = ({"role": "user", "content": "Hello"},)
MESSAGES_TEXT: Final = "Hello from native Messages"
MESSAGES_RESPONSE: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": MESSAGES_TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 4},
}
MESSAGES_EVENTS: Final = (
    ("message_start", {"type": "message_start", "message": {**MESSAGES_RESPONSE, "content": [], "stop_reason": None}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": MESSAGES_TEXT}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)

REQUEST: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 64}


def _error(kind: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": kind}}


BAD_REQUEST: Final = _error("invalid_request_error")
RATE_LIMITED: Final = _error("rate_limit_error")


async def call(server: RecordingServer, *, asynchronous: bool = True, **kwargs: object) -> object:
    return await call_natively(
        Route.MESSAGES,
        litellm.anthropic.messages.create,
        litellm.anthropic.messages.acreate,
        REQUEST,
        server,
        asynchronous=asynchronous,
        **kwargs,
    )


BODY_HOOKS: Final = gap("native Messages does not run the handler body hooks")

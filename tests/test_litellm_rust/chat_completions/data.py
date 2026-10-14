from typing import Final

import litellm
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_MODEL
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import call_natively


def _error(kind: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": kind}}


REQUEST: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32}
BAD_REQUEST: Final = _error("invalid_request_error")
RATE_LIMITED: Final = _error("rate_limit_error")


async def call(server: RecordingServer, *, asynchronous: bool = True, **kwargs: object) -> object:
    return await call_natively(
        Route.CHAT_COMPLETIONS,
        litellm.completion,
        litellm.acompletion,
        REQUEST,
        server,
        asynchronous=asynchronous,
        **kwargs,
    )

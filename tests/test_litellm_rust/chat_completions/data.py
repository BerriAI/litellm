from typing import Final

import litellm
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.support.callbacks import Surface


def _error(kind: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": kind}}


CHAT_SURFACE: Final = Surface(
    route=Route.CHAT_COMPLETIONS,
    request={"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32},
    response=MESSAGES_RESPONSE,
    events=MESSAGES_EVENTS,
    bad_request=_error("invalid_request_error"),
    rate_limited=_error("rate_limit_error"),
    create=litellm.completion,
    acreate=litellm.acompletion,
)

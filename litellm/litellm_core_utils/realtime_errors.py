"""Loud-failure helpers for the realtime WebSocket paths.

A realtime caller that only gets a bare close frame has nothing to act on, so
every failure surfaces as an OpenAI-style ``error`` event plus a close frame
whose reason names the failure. Close reasons are capped at
``WEBSOCKET_CLOSE_REASON_MAX_BYTES``: RFC 6455 control frames carry at most 125
bytes, two of which hold the status code, and a longer reason makes the close
frame itself fail, which is how a loud failure turns back into a silent one.
"""

import json
from typing import Final

from litellm.types.realtime import RealtimeErrorDetail, RealtimeErrorEvent

WEBSOCKET_CLOSE_REASON_MAX_BYTES: Final = 123


def realtime_error_event(message: str, error_type: str) -> str:
    detail: Final[RealtimeErrorDetail] = {"type": error_type, "message": message}
    event: Final[RealtimeErrorEvent] = {"type": "error", "error": detail}
    return json.dumps(event)


def websocket_close_reason(message: str, fallback: str) -> str:
    encoded: Final = message.encode("utf-8")
    if not encoded:
        return fallback
    if len(encoded) <= WEBSOCKET_CLOSE_REASON_MAX_BYTES:
        return message
    return encoded[:WEBSOCKET_CLOSE_REASON_MAX_BYTES].decode("utf-8", errors="ignore")

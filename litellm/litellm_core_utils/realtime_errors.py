"""Loud-failure helpers for the realtime WebSocket paths.

A realtime caller that only gets a bare close frame has nothing to act on, so
every failure surfaces as an OpenAI-style ``error`` event plus a close frame
whose reason names the failure. Close reasons are capped at
``WEBSOCKET_CLOSE_REASON_MAX_BYTES``: RFC 6455 control frames carry at most 125
bytes, two of which hold the status code, and a longer reason makes the close
frame itself fail, which is how a loud failure turns back into a silent one.
"""

import json
from types import MappingProxyType
from typing import Final, Protocol

from litellm.types.realtime import RealtimeErrorDetail, RealtimeErrorEvent

WEBSOCKET_CLOSE_REASON_MAX_BYTES: Final = 123


class _ClientWebSocket(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = ..., reason: str | None = ...) -> None: ...


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


def client_close_code(upstream_code: int) -> int:
    from websockets.frames import EXTERNAL_CLOSE_CODES, CloseCode

    if upstream_code in EXTERNAL_CLOSE_CODES or 3000 <= upstream_code < 5000:
        return upstream_code
    return int(CloseCode.INTERNAL_ERROR)


def upstream_handshake_close_code(status_code: int) -> int:
    from websockets.frames import CloseCode

    refusal_codes: Final = MappingProxyType(
        {
            401: int(CloseCode.POLICY_VIOLATION),
            403: int(CloseCode.POLICY_VIOLATION),
            429: int(CloseCode.TRY_AGAIN_LATER),
        }
    )
    return refusal_codes.get(status_code, int(CloseCode.INTERNAL_ERROR))


async def close_after_upstream_handshake_refusal(websocket: _ClientWebSocket, status_code: int) -> None:
    message: Final = f"Upstream realtime handshake rejected with HTTP {status_code}"
    try:
        await websocket.send_text(realtime_error_event(message, error_type="server_error"))
    except Exception:  # noqa: BLE001  # best-effort notice: a dead client socket must not skip the close below
        pass
    await websocket.close(
        code=upstream_handshake_close_code(status_code),
        reason=websocket_close_reason(message, fallback="Upstream handshake rejected"),
    )

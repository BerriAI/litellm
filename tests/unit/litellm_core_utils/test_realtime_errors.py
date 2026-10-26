import json
from typing import cast

import pytest

from litellm.litellm_core_utils.realtime_errors import (
    WEBSOCKET_CLOSE_REASON_MAX_BYTES,
    client_close_code,
    close_after_upstream_handshake_refusal,
    realtime_error_event,
    upstream_handshake_close_code,
    websocket_close_reason,
)
from litellm.types.realtime import RealtimeErrorEvent


def test_realtime_error_event_shape():
    event = json.loads(realtime_error_event("token refresh failed", error_type="server_error"))

    assert event == {
        "type": "error",
        "error": {"type": "server_error", "message": "token refresh failed"},
    }


def test_websocket_close_reason_keeps_short_messages_intact():
    assert websocket_close_reason("boom", fallback="Internal server error") == "boom"


def test_websocket_close_reason_falls_back_on_empty_message():
    assert websocket_close_reason("", fallback="Internal server error") == "Internal server error"


def test_websocket_close_reason_truncates_long_ascii_message():
    reason = websocket_close_reason("x" * 500, fallback="Internal server error")

    assert len(reason.encode("utf-8")) <= WEBSOCKET_CLOSE_REASON_MAX_BYTES
    assert reason == "x" * WEBSOCKET_CLOSE_REASON_MAX_BYTES


def test_websocket_close_reason_truncates_multibyte_message_by_bytes():
    """A close frame carries at most 123 bytes of reason, not 123 characters:
    truncating by characters lets a multibyte message overflow the control
    frame, which makes the close itself fail and leaves the caller with a bare
    abnormal closure and no reason at all."""
    reason = websocket_close_reason("あ" * 200, fallback="Internal server error")

    assert len(reason.encode("utf-8")) <= WEBSOCKET_CLOSE_REASON_MAX_BYTES
    assert reason == "あ" * (WEBSOCKET_CLOSE_REASON_MAX_BYTES // 3)
    assert "�" not in reason


@pytest.mark.parametrize(
    ("upstream_code", "expected"),
    [(1000, 1000), (1008, 1008), (1011, 1011), (4001, 4001), (1005, 1011), (1006, 1011), (1015, 1011), (2999, 1011)],
)
def test_client_close_code_only_forwards_codes_a_server_may_send(upstream_code, expected):
    assert client_close_code(upstream_code) == expected


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, 1008), (403, 1008), (429, 1013), (500, 1011)],
)
def test_upstream_handshake_close_code_maps_http_status_to_close_code(status_code: int, expected: int):
    assert upstream_handshake_close_code(status_code) == expected


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str | None] | None = None

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason)


@pytest.mark.asyncio
async def test_close_after_upstream_handshake_refusal_sends_error_event_then_policy_close():
    websocket = _RecordingWebSocket()

    await close_after_upstream_handshake_refusal(websocket, 401)

    assert len(websocket.sent) == 1
    event = cast(RealtimeErrorEvent, json.loads(websocket.sent[0]))
    assert event["type"] == "error"
    assert event["error"]["type"] == "server_error"
    assert "401" in event["error"]["message"]
    assert websocket.closed is not None
    assert websocket.closed[0] == 1008
    assert websocket.closed[1]


@pytest.mark.asyncio
async def test_close_after_upstream_handshake_refusal_still_closes_when_send_fails():
    class _DeadWebSocket(_RecordingWebSocket):
        async def send_text(self, data: str) -> None:
            raise RuntimeError("socket gone")

    websocket = _DeadWebSocket()

    await close_after_upstream_handshake_refusal(websocket, 500)

    assert websocket.closed == (1011, "Upstream realtime handshake rejected with HTTP 500")

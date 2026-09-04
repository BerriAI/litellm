import json


from litellm.litellm_core_utils.realtime_errors import (
    WEBSOCKET_CLOSE_REASON_MAX_BYTES,
    realtime_error_event,
    websocket_close_reason,
)


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

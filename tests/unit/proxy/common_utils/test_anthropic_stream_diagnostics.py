import json
import logging
from typing import Final

import pytest

from litellm.proxy.common_utils.anthropic_stream_diagnostics import AnthropicStreamDiagnostics

_START: Final = b'data:{"type":"message_start","message":{"id":"msg_synthetic"}}\r\n\r\n'
_TEXT: Final = (
    b'event:content_block_delta\r\ndata:{"type":"content_block_delta",'
    b'"delta":{"type":"text_delta","text":"PRIVATE_PROMPT sk-synthetic-secret"}}\r\n\r\n'
)
_STOP: Final = b'data:{"type":"message_stop"}\r\n\r\n'


@pytest.mark.parametrize("mode", ("off", "", "invalid"))
def test_diagnostics_disabled_creates_no_observer(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", mode)
    assert AnthropicStreamDiagnostics.from_env("provider_read", "call_synthetic") is None


def test_diagnostics_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", raising=False)
    assert AnthropicStreamDiagnostics.from_env("provider_read", "call_synthetic") is None


def test_only_allowlisted_request_id_header_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", "all")
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics.from_env(
        "provider_read",
        "call_synthetic",
        {"x-request-id": "upstream_request_1", "authorization": "sk-private", "x-api-key": "secret-value"},
    )
    assert observer is not None
    observer.finish("eof")
    assert "upstream_request_1" in caplog.text
    assert "sk-private" not in caplog.text and "secret-value" not in caplog.text


@pytest.mark.parametrize("mode,complete,expected", (("errors", True, 0), ("errors", False, 1), ("all", True, 1)))
def test_summary_is_bounded_and_excludes_content(
    mode: str, complete: bool, expected: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics(mode, "provider_read", "call_synthetic", clock=lambda: 1.0)
    wire: Final = _START + _TEXT + (_STOP if complete else b"")
    for byte in wire:
        observer.observe(bytes((byte,)))
    assert not caplog.records
    observer.finish("eof" if complete else "closed")
    assert len(caplog.records) == expected
    assert "PRIVATE_PROMPT" not in caplog.text and "sk-synthetic-secret" not in caplog.text
    if not expected:
        return
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["stage"], summary["call_id"], summary["message_id"]) == (
        "provider_read",
        "call_synthetic",
        "msg_synthetic",
    )
    assert (summary["starts"], summary["text_deltas"], summary["stops"]) == (1, 1, int(complete))
    assert (summary["chunks"], summary["bytes"], summary["pending_bytes"]) == (len(wire), len(wire), 0)
    assert summary["complete"] is complete
    assert summary["first_text_ms"] == 0
    assert len(caplog.records[0].getMessage()) < 1024


def test_oversize_unfinished_frame_is_bounded_and_next_frame_is_observed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("all", "provider_read", "call_synthetic", clock=lambda: 1.0)
    observer.observe(_START)
    for _ in range(100):
        observer.observe(b"x" * 65536)
        assert len(observer.pending) <= 65536
    observer.observe(b"\r")
    observer.observe(b"\n\r")
    observer.observe(b"\n" + _TEXT + _STOP)
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["unparsed_frames"], summary["text_deltas"], summary["stops"]) == (1, 1, 1)
    assert len(caplog.records) == 1


def test_empty_text_delta_does_not_count_as_body_received(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("all", "provider_read", "call_synthetic", clock=lambda: 1.0)
    observer.observe(
        _START + b'data:{"type":"content_block_delta","delta":{"type":"text_delta","text":""}}\n\n' + _STOP
    )
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["text_deltas"], summary["text_chars"], summary["first_text_ms"]) == (1, 0, None)

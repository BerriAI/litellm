import json
import logging
from collections.abc import AsyncGenerator
from typing import Final

import pytest

from litellm.proxy.common_utils.anthropic_stream_diagnostics import AnthropicStreamDiagnostics, StreamStage

_START: Final = b'data:{"type":"message_start","message":{"id":"msg_synthetic"}}\r\n\r\n'
_TEXT: Final = (
    b'event:content_block_delta\r\ndata:{"type":"content_block_delta",'
    b'"delta":{"type":"text_delta","text":"PRIVATE_PROMPT sk-synthetic-secret"}}\r\n\r\n'
)
_STOP: Final = b'data:{"type":"message_stop"}\r\n\r\n'


class _Clock:
    def __init__(self) -> None:
        self.seconds = 100.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


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


@pytest.mark.parametrize(
    "stage,header", (("provider_read", "X-TC-RequestId"), ("http_send", "LLM_Provider-X-TC-RequestId"))
)
def test_tencent_request_id_survives_header_casing_and_proxy_prefix(
    stage: StreamStage, header: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", "slow")
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics.from_env(
        stage, "call_synthetic", {header: "tc-request-synthetic", "authorization": "PRIVATE_KEY"}
    )
    assert observer is not None
    observer.observe(_START)
    observer.finish("closed")
    observer.finish("closed")
    assert len(caplog.records) == 1
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert summary["upstream_request_id"] == "tc-request-synthetic"
    assert "PRIVATE_KEY" not in caplog.text


def test_event_timestamps_distinguish_reasoning_from_text_without_logging_either(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    clock: Final = _Clock()
    observer: Final = AnthropicStreamDiagnostics(
        "all", "provider_read", "call_synthetic", clock=clock, wall_clock=lambda: 1_000.0
    )
    clock.advance(2)
    observer.observe(_START)
    clock.advance(3)
    observer.observe(
        b'data:{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"PRIVATE_REASONING"}}\n\n'
    )
    clock.advance(7)
    observer.observe(_TEXT[:10])
    clock.advance(1)
    observer.observe(_TEXT[10:])
    observer.observe(_STOP)
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["started_at_unix_ms"], summary["started_monotonic_ms"]) == (1_000_000, 100_000)
    assert (
        summary["first_event_ms"],
        summary["first_content_ms"],
        summary["first_text_ms"],
        summary["last_content_ms"],
    ) == (2_000, 5_000, 13_000, 13_000)
    assert (summary["first_event_type"], summary["last_event_type"], summary["max_content_gap_ms"]) == (
        "message_start",
        "message_stop",
        8_000,
    )
    assert "PRIVATE_REASONING" not in caplog.text and "PRIVATE_PROMPT" not in caplog.text


@pytest.mark.parametrize("gap,expected", ((2, 0), (31, 1)))
def test_slow_mode_logs_content_gaps_not_total_generation_length(
    gap: int, expected: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    clock: Final = _Clock()
    observer: Final = AnthropicStreamDiagnostics("slow", "provider_read", "call_synthetic", clock=clock)
    observer.observe(_START)
    for _ in range(50):
        clock.advance(gap)
        observer.observe(_TEXT)
    observer.observe(_STOP)
    observer.finish("eof")
    observer.finish("closed")
    assert len(caplog.records) == expected


def test_unknown_events_and_invalid_ids_cannot_leak_payload_into_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics(
        "all", "provider_read", "sk-secret", upstream_request_id="PRIVATE\nPROMPT"
    )
    observer.observe(
        b'data:{"type":"PRIVATE_EVENT","message":{"id":"sk-private"},"error":{"message":"PRIVATE_ERROR"}}\n\n'
    )
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["call_id"], summary["upstream_request_id"], summary["message_id"]) == (None, None, None)
    assert (summary["first_event_type"], summary["last_event_type"]) == ("other", "other")
    assert "PRIVATE" not in caplog.text and "sk-secret" not in caplog.text
    assert len(caplog.records[0].getMessage()) < 2048


@pytest.mark.parametrize("frame", (b"data:\n\n", b"data: \r\n\r\n", b"data\r\r", b"data: \ndata:\n\n"))
def test_empty_data_is_reported_even_when_the_stream_completes(frame: bytes, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("errors", "provider_read", "call_synthetic", clock=lambda: 1.0)
    for byte in _START + frame + _STOP:
        observer.observe(bytes((byte,)))
    observer.finish("eof")
    assert len(caplog.records) == 1
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["empty_data_frames"], summary["unparsed_frames"], summary["complete"]) == (1, 0, False)
    assert (summary["starts"], summary["stops"], summary["pending_bytes"]) == (1, 1, 0)


def test_comment_heartbeats_do_not_make_a_completed_stream_fail(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("errors", "provider_read", "call_synthetic", clock=lambda: 1.0)
    observer.observe(_START + b": ping\n\n" + _TEXT + b": heartbeat\r\n\r\n" + _STOP)
    observer.finish("eof")
    assert not caplog.records


@pytest.mark.parametrize("delimiter", (b"\n\n", b"\r\n\r\n", b"\r\r"))
def test_large_fragmented_frames_and_split_delimiters_preserve_event_counts(
    delimiter: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("all", "provider_read", "call_synthetic", clock=lambda: 1.0)
    prefix: Final = b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"'
    text: Final = b"x" * 32_768
    observer.observe(_START)
    for byte in prefix + text + b'"}}' + delimiter:
        observer.observe(bytes((byte,)))
    observer.observe(_STOP)
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["events"], summary["text_chars"], summary["pending_bytes"], summary["complete"]) == (
        3,
        len(text),
        0,
        True,
    )


@pytest.mark.parametrize("frame", (b"data: broken JSON\n\n", b"data: " + b"x" * 65_536 + b"\n\n"))
def test_unparseable_complete_event_is_reported_without_blocking_later_events(
    frame: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("errors", "provider_read", "call_synthetic", clock=lambda: 1.0)
    observer.observe(_START + frame + _STOP)
    observer.finish("eof")
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert (summary["unparsed_frames"], summary["stops"], summary["complete"]) == (1, 1, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_error", (False, True))
async def test_relay_preserves_bytes_and_reports_eof_or_transport_failure_once(
    transport_error: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    observer: Final = AnthropicStreamDiagnostics("all", "router_emit", "call_synthetic", clock=lambda: 1.0)

    async def source() -> AsyncGenerator[bytes, None]:
        yield _START
        if transport_error:
            raise OSError("PRIVATE_ERROR")
        yield _STOP

    relay: Final = observer.relay(source())
    assert await anext(relay) == _START
    if transport_error:
        with pytest.raises(OSError, match="PRIVATE_ERROR"):
            await anext(relay)
    else:
        assert await anext(relay) == _STOP
        with pytest.raises(StopAsyncIteration):
            await anext(relay)
    observer.finish("closed")
    observer.observe(b"ignored after finish")
    assert len(caplog.records) == 1
    summary: Final = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert summary["end"] == ("transport_error" if transport_error else "eof")
    assert summary["complete"] is not transport_error
    assert "PRIVATE_ERROR" not in caplog.text

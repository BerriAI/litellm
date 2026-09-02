import asyncio
import base64
import json
from collections.abc import Callable
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.meta.realtime.handler import (
    DEFAULT_MUSE_REALTIME_URL,
    MetaRealtime,
    MuseAdapterError,
    MuseRealtimeAdapter,
    build_muse_realtime_url,
    normalize_access_token,
    safe_close_reason,
    sanitize_close_code,
)
from litellm.llms.meta.realtime.transformation import MUSE_MODEL


class FakeProviderWebSocket:
    def __init__(self, session_id: str = "provider-session") -> None:
        self.sent: list[str | bytes] = []
        self.close_calls: list[tuple[int, str]] = []
        self._session_id: Final = session_id
        self._recv_count = 0
        self._closed = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        self._recv_count += 1
        if self._recv_count == 1:
            return json.dumps({"sessionId": self._session_id})
        await self._closed.wait()
        raise MuseAdapterError("closed", close_code=1000)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_calls.append((code, reason))
        self._closed.set()


class DelayedAckWebSocket(FakeProviderWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.ack_release = asyncio.Event()

    async def recv(self, decode: bool | None = None) -> str | bytes:
        self._recv_count += 1
        if self._recv_count == 1:
            await self.ack_release.wait()
            return json.dumps({"sessionId": self._session_id})
        await self._closed.wait()
        raise MuseAdapterError("closed", close_code=1000)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def _session_update(*, rate: int = 24_000, mode: str = "ENDPOINTING") -> str:
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "mode": mode,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate, "channels": 1},
                        "transcription": {"model": MUSE_MODEL},
                    }
                },
            },
        }
    )


async def _configured_adapter(
    *,
    rate: int = 24_000,
    mode: str = "ENDPOINTING",
    provider_ws: FakeProviderWebSocket | None = None,
    monotonic: Callable[[], float] = lambda: 10.0,
    sleep: Callable[[float], object] | None = None,
) -> tuple[MuseRealtimeAdapter, FakeProviderWebSocket, dict[str, object]]:
    ws: Final = provider_ws or FakeProviderWebSocket()
    connect_call: Final[dict[str, object]] = {}

    async def connect(url: str, **kwargs: object) -> FakeProviderWebSocket:
        connect_call.update({"url": url, **kwargs})
        return ws

    async def no_sleep(_: float) -> None:
        return None

    adapter: Final = MuseRealtimeAdapter(
        model=f"meta/{MUSE_MODEL}",
        api_key=" raw-token ",
        websocket_connect=connect,
        monotonic=monotonic,
        sleep=sleep or no_sleep,
    )
    created: Final = json.loads(await adapter.recv())
    assert created["type"] == "session.created"
    await adapter.send(_session_update(rate=rate, mode=mode))
    updated: Final = json.loads(await adapter.recv())
    assert updated["type"] == "session.updated"
    return adapter, ws, connect_call


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("token", "Bearer token"),
        (" Bearer token ", "Bearer token"),
        ("bearer   token", "Bearer token"),
    ],
)
def test_normalize_access_token_emits_exactly_one_bearer_prefix(api_key: str, expected: str):
    assert normalize_access_token(api_key) == expected


@pytest.mark.parametrize("api_key", ["", " ", "Bearer", " bearer "])
def test_normalize_access_token_rejects_missing_token(api_key: str):
    with pytest.raises(ValueError, match=r"token|key is required"):
        normalize_access_token(api_key)


def test_build_muse_realtime_url_uses_fixed_secure_path():
    assert build_muse_realtime_url(None) == DEFAULT_MUSE_REALTIME_URL
    assert build_muse_realtime_url("https://example.test/custom/path?ignored=yes") == (
        "wss://example.test/v1/asr/realtime"
    )
    assert build_muse_realtime_url("wss://example.test:8443/other") == ("wss://example.test:8443/v1/asr/realtime")


@pytest.mark.parametrize(
    "api_base",
    [
        "http://example.test",
        "ws://example.test",
        "wss://user:pass@example.test",
        "wss://example.test/path#fragment",
        "not-a-url",
    ],
)
def test_build_muse_realtime_url_rejects_insecure_or_ambiguous_overrides(api_base: str):
    with pytest.raises(ValueError, match="absolute wss:// or https://"):
        build_muse_realtime_url(api_base)


@pytest.mark.asyncio
async def test_handshake_contains_bearer_only_in_json_body_and_waits_for_ack():
    provider_ws: Final = DelayedAckWebSocket()
    connect_call: Final[dict[str, object]] = {}

    async def connect(url: str, **kwargs: object) -> DelayedAckWebSocket:
        connect_call.update({"url": url, **kwargs})
        return provider_ws

    adapter: Final = MuseRealtimeAdapter(
        model=MUSE_MODEL,
        api_key="Bearer private-token",
        websocket_connect=connect,
    )
    await adapter.recv()
    update_task: Final = asyncio.create_task(adapter.send(_session_update()))
    await _wait_until(lambda: len(provider_ws.sent) == 1)

    assert connect_call["url"] == DEFAULT_MUSE_REALTIME_URL
    assert "additional_headers" not in connect_call
    handshake: Final = json.loads(provider_ws.sent[0])
    assert handshake["authorization"] == {"accessToken": "Bearer private-token"}
    assert handshake["audioEncoding"] == "PCM_24KHZ"
    assert not update_task.done()
    assert not any(isinstance(frame, bytes) for frame in provider_ws.sent)

    provider_ws.ack_release.set()
    await update_task
    updated: Final = json.loads(await adapter.recv())
    assert updated["type"] == "session.updated"
    assert updated["session"]["id"] == "provider-session"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("rate", "packet_bytes"), [(16_000, 2_560), (24_000, 3_840)])
async def test_audio_is_strictly_decoded_and_packetized_as_raw_pcm(rate: int, packet_bytes: int):
    adapter, provider_ws, _ = await _configured_adapter(rate=rate)
    pcm: Final = (b"\xff\xfe\x00\x80" * (packet_bytes // 2))[: packet_bytes * 2]

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
    await _wait_until(lambda: sum(isinstance(frame, bytes) for frame in provider_ws.sent) == 2)

    binary_frames: Final = tuple(frame for frame in provider_ws.sent if isinstance(frame, bytes))
    assert binary_frames == (pcm[:packet_bytes], pcm[packet_bytes:])
    assert b"\xff\xfe" in pcm
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio", "expected_message"),
    [
        ("not base64!", "valid base64"),
        (base64.b64encode(b"\x00").decode(), "complete samples"),
    ],
)
async def test_invalid_base64_or_odd_pcm_is_rejected(audio: str, expected_message: str):
    adapter, provider_ws, _ = await _configured_adapter()

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
    error: Final = json.loads(await adapter.recv())

    assert error["type"] == "error"
    assert error["error"]["code"] == "invalid_audio"
    assert expected_message in error["error"]["message"]
    assert adapter.close_code == 1008
    assert not any(isinstance(frame, bytes) for frame in provider_ws.sent)
    await adapter.close()


@pytest.mark.asyncio
async def test_absolute_pacing_delays_only_audio_ahead_of_wall_time():
    sleeps: Final[list[float]] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    adapter, provider_ws, _ = await _configured_adapter(monotonic=lambda: 10.0, sleep=record_sleep)
    pcm: Final = b"\x01\x02" * 3_840

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
    await _wait_until(lambda: sum(isinstance(frame, bytes) for frame in provider_ws.sent) == 2)

    assert sleeps == pytest.approx([0.08])
    await adapter.close()


@pytest.mark.asyncio
async def test_append_larger_than_four_seconds_is_rejected_without_dropping_prefix():
    adapter, provider_ws, _ = await _configured_adapter(rate=16_000)
    oversized_pcm: Final = b"\x00\x00" * (16_000 * 4 + 1)

    await adapter.send(
        json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(oversized_pcm).decode()})
    )
    error: Final = json.loads(await adapter.recv())

    assert error["error"]["code"] == "audio_backlog_exceeded"
    assert adapter.close_code == 1008
    assert not any(isinstance(frame, bytes) for frame in provider_ws.sent)
    await adapter.close()


@pytest.mark.asyncio
async def test_clear_discards_only_unsent_audio():
    adapter, provider_ws, _ = await _configured_adapter()
    old_pcm: Final = b"\x01\x02" * 100
    new_pcm: Final = b"\x03\x04" * 100

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(old_pcm).decode()}))
    await adapter.send(_event("input_audio_buffer.clear"))
    cleared: Final = json.loads(await adapter.recv())
    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(new_pcm).decode()}))
    await adapter.send(_event("input_audio_buffer.commit"))
    await _wait_until(lambda: any(isinstance(frame, bytes) for frame in provider_ws.sent))

    assert cleared["type"] == "input_audio_buffer.cleared"
    assert tuple(frame for frame in provider_ws.sent if isinstance(frame, bytes)) == (new_pcm,)
    assert '{"type":"endStream"}' not in provider_ws.sent
    await adapter.close()


@pytest.mark.asyncio
async def test_endpointing_commit_flushes_partial_packet_without_ending_stream():
    adapter, provider_ws, _ = await _configured_adapter(mode="ENDPOINTING")
    pcm: Final = b"\x01\x02" * 100

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
    await adapter.send(_event("input_audio_buffer.commit"))
    committed: Final = json.loads(await adapter.recv())
    await _wait_until(lambda: any(isinstance(frame, bytes) for frame in provider_ws.sent))

    assert committed["type"] == "input_audio_buffer.committed"
    assert committed["item_id"].startswith("item_")
    assert committed["previous_item_id"] is None
    assert tuple(frame for frame in provider_ws.sent if isinstance(frame, bytes)) == (pcm,)
    assert '{"type":"endStream"}' not in provider_ws.sent
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "terminal_event"),
    [("PUSH_TO_TALK", "input_audio_buffer.commit"), ("ENDPOINTING", "input_audio_buffer.end")],
)
async def test_commit_or_end_sends_end_stream_exactly_once(mode: str, terminal_event: str):
    adapter, provider_ws, _ = await _configured_adapter(mode=mode)
    pcm: Final = b"\x01\x02" * 100

    await adapter.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
    await adapter.send(_event(terminal_event))
    await adapter.send(_event("input_audio_buffer.end"))
    await _wait_until(lambda: '{"type":"endStream"}' in provider_ws.sent)

    assert tuple(frame for frame in provider_ws.sent if isinstance(frame, bytes)) == (pcm,)
    assert provider_ws.sent.count('{"type":"endStream"}') == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_response_create_is_returned_as_error_and_never_sent_upstream():
    adapter, provider_ws, _ = await _configured_adapter()

    await adapter.send(_event("response.create"))
    error: Final = json.loads(await adapter.recv())

    assert error["type"] == "error"
    assert error["error"]["code"] == "unsupported_event"
    assert not any(isinstance(frame, str) and "response.create" in frame for frame in provider_ws.sent)
    await adapter.close()


@pytest.mark.asyncio
async def test_close_codes_and_reasons_are_sanitized_without_secret_leakage():
    adapter, provider_ws, _ = await _configured_adapter()
    secret: Final = "Bearer private-token"

    await adapter.close(code=4001, reason=f"provider rejected {secret}")

    assert adapter.close_code == 1011
    assert adapter.close_reason == "Realtime transcription service error"
    assert provider_ws.close_calls == [(1011, "Realtime transcription service error")]
    assert secret not in json.dumps(provider_ws.close_calls)
    assert sanitize_close_code(1013) == 1013
    assert safe_close_reason(1008) == "Invalid realtime transcription request"


@pytest.mark.asyncio
async def test_handshake_failure_reports_only_exception_type():
    secret: Final = "private-token"

    async def failing_connect(url: str, **kwargs: object) -> FakeProviderWebSocket:
        raise RuntimeError(f"failed with {secret}")

    adapter: Final = MuseRealtimeAdapter(
        model=MUSE_MODEL,
        api_key=secret,
        websocket_connect=failing_connect,
    )
    await adapter.recv()

    await adapter.send(_session_update())
    error = json.loads(await adapter.recv())

    assert error["type"] == "error"
    assert error["error"]["message"] == "Meta Muse realtime handshake failed"
    assert secret not in json.dumps(error)
    with pytest.raises(MuseAdapterError) as exc_info:
        await adapter.recv()
    assert exc_info.value.close_code == 1011
    assert secret not in str(exc_info.value)


@pytest.mark.asyncio
async def test_meta_realtime_missing_credentials_closes_client_with_policy_code():
    client_ws: Final = MagicMock()
    client_ws.send_text = AsyncMock()
    client_ws.close = AsyncMock()

    await MetaRealtime().async_realtime(
        model=MUSE_MODEL,
        websocket=client_ws,
        logging_obj=MagicMock(),
        api_key=None,
    )

    client_ws.close.assert_awaited_once_with(
        code=1008,
        reason="Invalid realtime transcription request",
    )
    sent_error: Final = json.loads(client_ws.send_text.await_args.args[0])
    assert sent_error["type"] == "error"
    assert sent_error["error"]["code"] == "invalid_configuration"


@pytest.mark.asyncio
async def test_meta_realtime_invalid_constructor_input_sends_error_before_close():
    client_ws: Final = MagicMock()
    client_ws.send_text = AsyncMock()
    client_ws.close = AsyncMock()

    await MetaRealtime().async_realtime(
        model=MUSE_MODEL,
        websocket=client_ws,
        logging_obj=MagicMock(),
        api_key="Bearer",
    )

    sent_error: Final = json.loads(client_ws.send_text.await_args.args[0])
    assert sent_error["error"]["message"] == "Invalid Meta Muse realtime configuration"
    client_ws.close.assert_awaited_once_with(
        code=1008,
        reason="Invalid realtime transcription request",
    )


@pytest.mark.asyncio
async def test_meta_realtime_enables_private_logging_usage_and_model_enforcement(monkeypatch: pytest.MonkeyPatch):
    captured: Final[dict[str, object]] = {}

    class CapturingStreaming:
        def __init__(self, websocket, backend_ws, logging_obj, **kwargs):
            captured.update({"websocket": websocket, "backend_ws": backend_ws, "logging_obj": logging_obj, **kwargs})

        async def bidirectional_forward(self) -> None:
            return None

    client_ws: Final = MagicMock()
    client_ws.send_text = AsyncMock()
    client_ws.close = AsyncMock()
    monkeypatch.setattr("litellm.llms.meta.realtime.handler.RealTimeStreaming", CapturingStreaming)

    await MetaRealtime().async_realtime(
        model=MUSE_MODEL,
        websocket=client_ws,
        logging_obj=MagicMock(),
        api_key="private-token",
    )

    adapter: Final = captured["backend_ws"]
    assert isinstance(adapter, MuseRealtimeAdapter)
    assert captured["force_transcription_model"] == MUSE_MODEL
    assert captured["usage_provider"] is adapter
    assert captured["exclude_private_content_from_logs"] is True
    client_ws.close.assert_awaited_once_with(code=1000, reason="Session closed")


def _event(event_type: str) -> str:
    return json.dumps({"type": event_type})

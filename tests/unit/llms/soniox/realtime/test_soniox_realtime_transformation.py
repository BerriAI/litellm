import asyncio
import base64
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.soniox.realtime.transformation import (
    CONNECTED_FRAME,
    DEFAULT_SONIOX_REALTIME_URL,
    SESSION_STARTED_FRAME,
    SonioxEventTransformer,
    SonioxRealtimeBackend,
    SonioxProtocolError,
    SonioxRealtimeConfig,
    SonioxRealtimeOptions,
)
from litellm.types.llms.openai import OpenAIRealtimeEvents
from litellm.types.realtime import RealtimeResponseTransformInput
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL: Final = "stt-rt-v5"
EMPTY_TRANSFORM_INPUT: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}
PCM: Final = b"\x01\x02" * 160


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


def _session_update(audio_input: dict[str, object]) -> str:
    return _event("session.update", session={"type": "transcription", "audio": {"input": audio_input}})


def _append() -> str:
    return _event("input_audio_buffer.append", audio=base64.b64encode(PCM).decode())


def _start_request(frames: tuple[str | bytes, ...]) -> dict[str, object]:
    assert isinstance(frames[0], str)
    return json.loads(frames[0])


def _response(tokens: list[dict[str, object]], total_audio_proc_ms: int | None = None) -> str:
    return json.dumps({"tokens": tokens, "total_audio_proc_ms": total_audio_proc_ms})


def _token(text: str, is_final: bool = True, **fields: object) -> dict[str, object]:
    return {"text": text, "is_final": is_final, **fields}


def _types(events: tuple[OpenAIRealtimeEvents, ...]) -> list[object]:
    return [event["type"] for event in events]


def test_session_update_starts_the_stream_with_client_audio_and_deployment_options():
    config = SonioxRealtimeConfig(
        options=SonioxRealtimeOptions(language_hints=("en", "fr"), enable_speaker_diarization=True)
    )

    frames = config.transform_realtime_request(
        _session_update(
            {
                "format": {"type": "audio/pcm", "rate": 16_000},
                "transcription": {"model": "client-chosen-model", "language": "fr"},
                "turn_detection": None,
            }
        ),
        MODEL,
    )

    assert _start_request(frames) == {
        "model": MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": 16_000,
        "num_channels": 1,
        "enable_endpoint_detection": False,
        "language_hints": ["fr", "en"],
        "enable_speaker_diarization": True,
    }


def test_audio_before_session_update_starts_a_default_stream_once():
    config = SonioxRealtimeConfig()

    first = config.transform_realtime_request(_append(), MODEL)
    second = config.transform_realtime_request(_append(), MODEL)
    late_update = config.transform_realtime_request(_session_update({"turn_detection": None}), MODEL)

    assert _start_request(first) == {
        "model": MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": 24_000,
        "num_channels": 1,
        "enable_endpoint_detection": True,
    }
    assert first[1:] == (PCM,)
    assert second == (PCM,)
    assert late_update == ()


@pytest.mark.parametrize(
    ("event_type", "control_frame"),
    [("input_audio_buffer.commit", '{"type":"finalize"}'), ("input_audio_buffer.end", "")],
)
def test_buffer_control_events_map_to_soniox_control_frames(event_type: str, control_frame: str):
    config = SonioxRealtimeConfig()
    config.transform_realtime_request(_append(), MODEL)

    assert config.transform_realtime_request(_event(event_type), MODEL) == (control_frame,)


def test_unsupported_client_events_are_dropped():
    config = SonioxRealtimeConfig()

    assert config.transform_realtime_request(_event("input_audio_buffer.clear"), MODEL) == ()


@pytest.mark.parametrize(
    "audio_input",
    [
        {"format": {"type": "audio/pcmu"}},
        {"format": {"type": "audio/pcm", "rate": 24_000, "channels": 2}},
        {"format": {"type": "audio/pcm", "rate": 0}},
    ],
)
def test_unsupported_audio_formats_are_rejected_without_starting_the_stream(audio_input: dict[str, object]):
    config = SonioxRealtimeConfig()

    with pytest.raises(SonioxProtocolError):
        config.transform_realtime_request(_session_update(audio_input), MODEL)
    assert _start_request(config.transform_realtime_request(_append(), MODEL))["sample_rate"] == 24_000


@pytest.mark.parametrize(
    ("audio_input", "sample_rate"),
    [({"format": {"type": "audio/pcm"}}, 24_000), ({"input_audio_format": "pcm16"}, 24_000)],
)
def test_pcm16_without_a_rate_streams_at_the_openai_default_rate(audio_input: dict[str, object], sample_rate: int):
    config = SonioxRealtimeConfig()

    frames = config.transform_realtime_request(_session_update(audio_input), MODEL)

    assert _start_request(frames)["sample_rate"] == sample_rate


def test_validate_environment_sends_the_key_as_a_bearer_header(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "env-key")
    config = SonioxRealtimeConfig()

    assert config.validate_environment({"x": "y"}, MODEL, "deployment-key") == {
        "x": "y",
        "Authorization": "Bearer deployment-key",
    }
    assert config.validate_environment({}, MODEL) == {"Authorization": "Bearer env-key"}


def test_validate_environment_rejects_a_missing_key(monkeypatch):
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SONIOX_API_KEY"):
        SonioxRealtimeConfig().validate_environment({}, MODEL)


def test_get_complete_url_defaults_to_soniox_and_only_accepts_websocket_bases():
    config = SonioxRealtimeConfig()

    assert config.get_complete_url(None, MODEL) == DEFAULT_SONIOX_REALTIME_URL
    assert config.get_complete_url("wss://proxy.example/ws", MODEL) == "wss://proxy.example/ws"
    with pytest.raises(ValueError, match="ws://"):
        config.get_complete_url("https://api.soniox.com", MODEL)


def test_endpoint_detection_emits_one_openai_turn_billed_by_processed_audio():
    transformer = SonioxEventTransformer(translated_only=False)

    provisional = transformer.transform(_response([_token("Hel", is_final=False)], total_audio_proc_ms=500))
    finals = transformer.transform(
        _response([_token("Hello"), _token(" world"), _token(" next", is_final=False)], total_audio_proc_ms=1500)
    )
    endpoint = transformer.transform(_response([_token("<end>")], total_audio_proc_ms=2000))

    assert _types(provisional) == ["input_audio_buffer.speech_started"]
    item_id = provisional[0]["item_id"]
    assert [(event["type"], event.get("delta")) for event in finals] == [
        ("conversation.item.input_audio_transcription.delta", "Hello"),
        ("conversation.item.input_audio_transcription.delta", " world"),
    ]
    assert _types(endpoint) == [
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert {event["item_id"] for event in (*finals, *endpoint)} == {item_id}
    assert endpoint[1]["transcript"] == "Hello world"
    assert endpoint[1]["usage"] == {"type": "duration", "seconds": 2.0}
    assert transformer.take_unbilled_usage() is None


def test_manual_finalization_completes_the_turn_without_speech_stopped_and_starts_a_new_item():
    transformer = SonioxEventTransformer(translated_only=False)

    first = transformer.transform(_response([_token("One"), _token("<fin>")]))
    second = transformer.transform(_response([_token("Two"), _token("<fin>")]))

    assert _types(first) == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert first[-1]["transcript"] == "One"
    assert "usage" not in first[-1]
    assert second[-1]["transcript"] == "Two"
    assert second[0]["item_id"] != first[0]["item_id"]


def test_finalization_without_speech_emits_nothing():
    transformer = SonioxEventTransformer(translated_only=False)

    assert transformer.transform(_response([_token("<fin>")])) == ()


def test_translation_sessions_transcribe_only_the_translated_tokens():
    transformer = SonioxEventTransformer(translated_only=True)

    events = transformer.transform(
        _response(
            [
                _token("Hola", translation_status="original"),
                _token("Hello", translation_status="translation"),
                _token("<end>"),
            ]
        )
    )

    assert [event.get("delta") for event in events if "delta" in event] == ["Hello"]
    assert events[-1]["transcript"] == "Hello"


def test_provider_errors_surface_as_openai_error_events():
    transformer = SonioxEventTransformer(translated_only=False)

    events = transformer.transform(json.dumps({"error_code": 401, "error_message": "Invalid API key."}))

    assert events == (
        {"type": "error", "error": {"type": "server_error", "message": "Soniox realtime error 401: Invalid API key."}},
    )


def test_malformed_provider_frames_are_rejected():
    with pytest.raises(SonioxProtocolError):
        SonioxEventTransformer(translated_only=False).transform('{"tokens": "nope"}')


def test_audio_processed_after_the_last_turn_is_billed_on_session_close():
    config = SonioxRealtimeConfig()
    config.transform_realtime_response(
        _response([_token("Hi"), _token("<end>")], total_audio_proc_ms=1000), MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT
    )
    config.transform_realtime_response(
        _response([], total_audio_proc_ms=3500), MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT
    )

    assert config.unbilled_usage_on_session_close(MODEL) == {"type": "duration", "seconds": 2.5}
    assert config.unbilled_usage_on_session_close(MODEL) is None


def test_translation_option_switches_the_config_to_translated_transcripts():
    config = SonioxRealtimeConfig(
        options=SonioxRealtimeOptions(translation={"type": "one_way", "target_language": "en"})
    )

    result = config.transform_realtime_response(
        _response([_token("Hola", translation_status="original"), _token("Hello", translation_status="translation")]),
        MODEL,
        MagicMock(),
        EMPTY_TRANSFORM_INPUT,
    )

    assert [event.get("delta") for event in result["response"] if "delta" in event] == ["Hello"]


def test_stream_open_time_is_billed_on_close_because_soniox_charges_the_whole_stream():
    now = [100.0]
    config = SonioxRealtimeConfig(clock=lambda: now[0])
    config.wrap_backend(_RecordingBackend())
    config.transform_realtime_response(
        _response([_token("Hi"), _token("<end>")], total_audio_proc_ms=5000), MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT
    )
    now[0] = 130.5

    assert config.unbilled_usage_on_session_close(MODEL) == {"type": "duration", "seconds": 25.5}
    assert config.unbilled_usage_on_session_close(MODEL) is None


def test_connection_is_announced_with_the_deployment_session_for_cost_tracking():
    config = SonioxRealtimeConfig(options=SonioxRealtimeOptions(language_hints=("de",)))

    result = config.transform_realtime_response(CONNECTED_FRAME, MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)

    assert _types(tuple(result["response"])) == ["session.created"]
    assert result["response"][0]["session"]["audio"]["input"]["transcription"] == {"model": MODEL, "language": "de"}


def test_backend_session_start_is_announced_with_the_negotiated_session():
    config = SonioxRealtimeConfig()
    connected = config.transform_realtime_response(CONNECTED_FRAME, MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)
    config.transform_realtime_request(
        _session_update(
            {
                "format": {"type": "audio/pcm", "rate": 16_000},
                "transcription": {"language": "it"},
                "turn_detection": None,
            }
        ),
        MODEL,
    )

    result = config.transform_realtime_response(SESSION_STARTED_FRAME, MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)

    assert _types(tuple(result["response"])) == ["session.created"]
    assert result["response"][0]["session"]["id"] == connected["response"][0]["session"]["id"]
    assert result["response"][0]["session"]["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 16_000},
        "transcription": {"model": MODEL, "language": "it"},
        "turn_detection": None,
    }


def test_provider_config_manager_passes_only_soniox_options_from_deployment_params():
    config = ProviderConfigManager.get_provider_realtime_config(
        MODEL,
        LlmProviders.SONIOX,
        {
            "api_key": "secret",
            "model": "soniox/stt-rt-v5",
            "audio_format": "mulaw",
            "language_hints": ["pl"],
            "context": {"terms": ["LiteLLM"]},
            "max_endpoint_delay_ms": 800,
        },
    )

    assert isinstance(config, SonioxRealtimeConfig)
    assert _start_request(config.transform_realtime_request(_append(), MODEL)) == {
        "model": MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": 24_000,
        "num_channels": 1,
        "enable_endpoint_detection": True,
        "language_hints": ["pl"],
        "context": {"terms": ["LiteLLM"]},
        "max_endpoint_delay_ms": 800,
    }


class _RecordingBackend:
    def __init__(self) -> None:
        self.sent: Final[list[str | bytes]] = []
        self.upstream: Final[asyncio.Queue[str | Exception]] = asyncio.Queue()
        self.entered = False
        self.exited = False
        self.closed = False

    async def __aenter__(self) -> "_RecordingBackend":
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.exited = True

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        frame = await self.upstream.get()
        if isinstance(frame, Exception):
            raise frame
        return frame

    async def close(self) -> None:
        self.closed = True


class _BrokenBackend(_RecordingBackend):
    async def send(self, message: str | bytes) -> None:
        raise ConnectionError("upstream gone")


@pytest.mark.asyncio
async def test_keepalive_is_sent_only_while_the_client_is_idle_and_stops_after_end_of_stream():
    now = [0.0]
    ticks: asyncio.Queue[float] = asyncio.Queue()

    async def sleep(_: float) -> None:
        now[0] = await ticks.get()

    async def tick(at: float) -> None:
        await ticks.put(at)
        for _ in range(5):
            await asyncio.sleep(0)

    inner = _RecordingBackend()
    backend = SonioxRealtimeBackend(inner, keepalive_interval=5.0, clock=lambda: now[0], sleep=sleep)

    async with backend:
        await tick(5.0)
        assert inner.sent == ['{"type":"keepalive"}']
        now[0] = 7.0
        await backend.send(PCM)
        await tick(10.0)
        assert inner.sent == ['{"type":"keepalive"}', PCM]
        await backend.send("")
        await tick(30.0)
        assert inner.sent == ['{"type":"keepalive"}', PCM, ""]

    assert inner.entered and inner.exited


@pytest.mark.asyncio
async def test_backend_announces_the_session_once_the_start_request_is_sent_while_relaying_upstream_frames():
    inner = _RecordingBackend()
    backend = SonioxRealtimeBackend(inner)

    async with backend:
        assert await backend.recv() == CONNECTED_FRAME
        pending = asyncio.create_task(backend.recv())
        await backend.send('{"type":"keepalive"}')
        await asyncio.sleep(0)
        assert not pending.done()
        await backend.send('{"model":"stt-rt-v5"}')
        assert await pending == SESSION_STARTED_FRAME
        await backend.send('{"model":"ignored-second-config"}')
        await inner.upstream.put('{"tokens":[]}')
        assert await backend.recv() == '{"tokens":[]}'


@pytest.mark.asyncio
async def test_backend_relays_upstream_frames_that_arrive_before_the_session_starts():
    inner = _RecordingBackend()

    async with SonioxRealtimeBackend(inner) as backend:
        await inner.upstream.put('{"error_code":401,"error_message":"Invalid API key."}')

        assert await backend.recv() == CONNECTED_FRAME
        assert await backend.recv() == '{"error_code":401,"error_message":"Invalid API key."}'


@pytest.mark.asyncio
async def test_a_cancelled_read_loses_no_frames_and_upstream_errors_reach_the_reader():
    inner = _RecordingBackend()

    async with SonioxRealtimeBackend(inner) as backend:
        assert await backend.recv() == CONNECTED_FRAME
        cancelled = asyncio.create_task(backend.recv())
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        await inner.upstream.put('{"tokens":[]}')
        await inner.upstream.put(ConnectionError("upstream closed"))

        assert await backend.recv() == '{"tokens":[]}'
        with pytest.raises(ConnectionError, match="upstream closed"):
            await backend.recv()


@pytest.mark.asyncio
async def test_closing_the_backend_cancels_pending_reads_and_closes_upstream():
    inner = _RecordingBackend()
    backend = SonioxRealtimeBackend(inner)
    await backend.__aenter__()
    assert await backend.recv() == CONNECTED_FRAME
    pending = asyncio.create_task(backend.recv())
    await backend.send('{"model":"stt-rt-v5"}')
    assert await pending == SESSION_STARTED_FRAME

    await backend.close()
    await inner.upstream.put('{"tokens":[]}')
    await asyncio.sleep(0)

    assert inner.closed
    assert inner.upstream.qsize() == 1


@pytest.mark.asyncio
async def test_keepalive_stops_after_the_upstream_rejects_a_send():
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    async with SonioxRealtimeBackend(_BrokenBackend(), keepalive_interval=0.0, sleep=sleep):
        for _ in range(5):
            await asyncio.sleep(0)

    assert sleeps == [0.0]

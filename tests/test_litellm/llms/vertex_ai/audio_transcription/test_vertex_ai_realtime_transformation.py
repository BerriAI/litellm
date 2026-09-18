import base64
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.base_llm.realtime.transformation import RealtimeBackend
from litellm.llms.vertex_ai.audio_transcription.realtime_transformation import (
    MAX_AUDIO_MESSAGE_BYTES,
    ChirpProtocolError,
    ChirpSessionConfig,
    SpeechStreamingTarget,
    VertexChirpRealtimeConfig,
    is_vertex_speech_to_text_model,
    new_words,
    parse_chirp_session_update,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.llms.vertex_ai_speech_to_text import (
    VertexSpeechStreamingConfigured,
    VertexSpeechStreamingResponse,
    VertexSpeechStreamingResult,
    VertexSpeechStreamingTurnDiscarded,
    VertexSpeechStreamingTurnFinished,
)
from litellm.types.realtime import RealtimeResponseTransformInput

MODEL: Final = "chirp_3"
EMPTY_TRANSFORM_INPUT: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}
DELTA: Final = "conversation.item.input_audio_transcription.delta"
COMPLETED: Final = "conversation.item.input_audio_transcription.completed"


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


def _ga_session_update(
    rate: int = 24_000, turn_detection: str | None = "server_vad", language: str | None = "en", model: str = MODEL
) -> str:
    transcription = {"model": model} if language is None else {"model": model, "language": language}
    return _event(
        "session.update",
        session={
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "turn_detection": None if turn_detection is None else {"type": turn_detection},
                    "transcription": transcription,
                }
            },
        },
    )


def _config(location: str | None = "us") -> VertexChirpRealtimeConfig:
    return VertexChirpRealtimeConfig(access_token="token", project="proj-1", location=location)


def _configured(
    rate: int = 24_000, turn_detection: str | None = "server_vad", language: str | None = "en"
) -> VertexChirpRealtimeConfig:
    config = _config()
    config.transform_session_created_event(MODEL, "sess_1")
    config.transform_realtime_request(_ga_session_update(rate, turn_detection, language), MODEL)
    return config


def _backend_events(config: VertexChirpRealtimeConfig, frame: object) -> list[dict[str, object]]:
    assert hasattr(frame, "model_dump_json")
    response = config.transform_realtime_response(frame.model_dump_json(), MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)[
        "response"
    ]
    assert isinstance(response, list)
    return response


def _response(
    *results: tuple[str, bool], speech_event: str = "none", billed_seconds: float = 0.0
) -> VertexSpeechStreamingResponse:
    return VertexSpeechStreamingResponse(
        speech_event=speech_event,
        results=tuple(VertexSpeechStreamingResult(transcript=text, is_final=final) for text, final in results),
        billed_seconds=billed_seconds,
    )


def _types(events: list[dict[str, object]]) -> list[object]:
    return [event["type"] for event in events]


def _commands(config: VertexChirpRealtimeConfig, payload: str) -> list[object]:
    return [json.loads(command) if isinstance(command, str) else command for command in config.transform_realtime_request(payload, MODEL)]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("vertex_ai/chirp_3", True),
        ("chirp_3", True),
        ("chirp_2", True),
        ("gemini-live-2.5-flash", False),
        ("vertex_ai/gemini-2.0-flash-live-preview-04-09", False),
    ],
)
def test_is_vertex_speech_to_text_model(model: str, expected: bool):
    assert is_vertex_speech_to_text_model(model) is expected


def test_ga_session_update_maps_to_a_speech_config():
    config = parse_chirp_session_update(_ga_session_update(16_000, "server_vad", "pt"), "vertex_ai/chirp_3")
    assert config == ChirpSessionConfig(model=MODEL, language="pt-BR", sample_rate=16_000, server_vad=True)
    assert json.loads(config.configure_command()) == {
        "kind": "configure",
        "model": MODEL,
        "language_codes": ["pt-BR"],
        "sample_rate_hertz": 16_000,
    }


def test_beta_session_update_defaults_the_rate_and_auto_detects_the_language():
    config = parse_chirp_session_update(
        _event(
            "transcription_session.update",
            session={"input_audio_format": "pcm16", "input_audio_transcription": {"model": MODEL}, "turn_detection": None},
        ),
        MODEL,
    )
    assert config == ChirpSessionConfig(model=MODEL, language=None, sample_rate=24_000, server_vad=False)
    assert json.loads(config.configure_command())["language_codes"] == ["auto"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_event("session.update", session={"type": "realtime_voice"}), "transcription sessions only"),
        (_ga_session_update(model="gemini-live-2.5-flash"), "cannot be changed"),
        (_event("session.update", session={"audio": {"input": {"format": {"type": "audio/pcmu"}}}}), "pcm16"),
        (_event("session.update", session={"audio": {"input": {"format": {"type": "audio/pcm", "channels": 2}}}}), "mono"),
        (_ga_session_update(rate=4_000), "sample rates"),
        (_ga_session_update(rate=96_000), "sample rates"),
        (_ga_session_update(turn_detection="semantic_vad"), "server_vad"),
    ],
)
def test_unsupported_session_settings_are_rejected(payload: str, message: str):
    with pytest.raises(ChirpProtocolError, match=message):
        parse_chirp_session_update(payload, MODEL)


def test_session_update_configures_once_and_later_updates_are_ignored():
    config = _config()
    config.transform_session_created_event(MODEL, "sess_1")
    first = _commands(config, _ga_session_update(16_000))
    assert first == [{"kind": "configure", "model": MODEL, "language_codes": ["en-US"], "sample_rate_hertz": 16_000}]
    assert config.is_setup_message(first[0])
    assert _commands(config, _ga_session_update(8_000)) == []


def test_audio_and_commits_before_session_update_are_rejected():
    config = _config()
    with pytest.raises(ChirpProtocolError, match=r"session\.update must configure"):
        config.transform_realtime_request(_event("input_audio_buffer.append", audio=base64.b64encode(b"\x00\x00").decode()), MODEL)
    with pytest.raises(ChirpProtocolError, match=r"session\.update must configure"):
        config.transform_realtime_request(_event("input_audio_buffer.commit"), MODEL)


def test_append_is_split_into_google_sized_chunks():
    config = _configured()
    audio = bytes(range(256)) * 250
    chunks = config.transform_realtime_request(_event("input_audio_buffer.append", audio=base64.b64encode(audio).decode()), MODEL)
    assert [len(chunk) for chunk in chunks] == [MAX_AUDIO_MESSAGE_BYTES, MAX_AUDIO_MESSAGE_BYTES, 64_000 - 2 * MAX_AUDIO_MESSAGE_BYTES]
    assert b"".join(chunk for chunk in chunks if isinstance(chunk, bytes)) == audio


def test_commit_end_and_clear_map_to_turn_commands():
    config = _configured()
    assert _commands(config, _event("input_audio_buffer.commit")) == [{"kind": "finish_turn"}]
    assert _commands(config, _event("input_audio_buffer.end")) == [{"kind": "finish_turn"}]
    assert _commands(config, _event("input_audio_buffer.clear")) == [{"kind": "discard_turn"}]


def test_unsupported_client_events_are_dropped():
    assert _commands(_configured(), _event("response.create")) == []


def test_connect_announces_a_session_with_chirp_defaults():
    event = _config().transform_session_created_event(MODEL, "sess_1")
    assert event["type"] == "session.created"
    assert event["session"]["id"] == "sess_1"
    assert event["session"]["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 24_000},
        "transcription": {"model": MODEL},
        "turn_detection": {"type": "server_vad"},
    }


def test_configured_backend_reports_the_negotiated_session():
    config = _configured(rate=16_000, turn_detection=None, language="pt-BR")
    events = _backend_events(config, VertexSpeechStreamingConfigured())
    assert _types(events) == ["session.created"]
    session = events[0]["session"]
    assert isinstance(session, dict)
    assert session["id"] == "sess_1"
    assert session["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 16_000},
        "transcription": {"model": MODEL, "language": "pt-BR"},
        "turn_detection": None,
    }


def test_backend_frames_before_session_update_are_an_error():
    config = _config()
    config.transform_session_created_event(MODEL, "sess_1")
    with pytest.raises(ChirpProtocolError, match=r"session\.update must configure"):
        _backend_events(config, VertexSpeechStreamingConfigured())


def test_server_vad_turn_streams_new_words_then_completes_with_usage():
    config = _configured()
    assert _types(_backend_events(config, _response(speech_event="begin"))) == ["input_audio_buffer.speech_started"]
    first = _backend_events(config, _response(("four score", False)))
    assert [(event["type"], event["delta"]) for event in first] == [(DELTA, "four score")]
    second = _backend_events(config, _response(("four score and seven", False)))
    assert [event["delta"] for event in second] == [" and seven"]
    final = _backend_events(config, _response(("Four score and seven years ago.", True), billed_seconds=3.5))
    assert _types(final) == [DELTA, "input_audio_buffer.speech_stopped", COMPLETED]
    assert final[0]["delta"] == " years ago."
    assert final[2]["transcript"] == "Four score and seven years ago."
    assert final[2]["usage"] == {"type": "duration", "seconds": 3.5}
    assert {event["item_id"] for event in (*first, *second, *final)} == {first[0]["item_id"]}
    assert _backend_events(config, _response(speech_event="end")) == []


def test_manual_turns_complete_on_commit_without_speech_events():
    config = _configured(turn_detection=None)
    assert _backend_events(config, _response(speech_event="begin")) == []
    first = _backend_events(config, _response(("hello there", True), billed_seconds=1.25))
    assert [(event["type"], event["delta"]) for event in first] == [(DELTA, "hello there")]
    second = _backend_events(config, _response(("world", True)))
    assert [event["delta"] for event in second] == [" world"]
    completed = _backend_events(config, VertexSpeechStreamingTurnFinished())
    assert _types(completed) == [COMPLETED]
    assert completed[0]["transcript"] == "hello there world"
    assert completed[0]["usage"] == {"type": "duration", "seconds": 1.25}
    assert _backend_events(config, VertexSpeechStreamingTurnFinished()) == []


def test_clear_discards_the_open_turn():
    config = _configured(turn_detection=None)
    draft = _backend_events(config, _response(("draft", False)))
    assert _backend_events(config, VertexSpeechStreamingTurnDiscarded()) == []
    assert _backend_events(config, VertexSpeechStreamingTurnFinished()) == []
    fresh = _backend_events(config, _response(("again", False)))
    assert fresh[0]["delta"] == "again"
    assert fresh[0]["item_id"] != draft[0]["item_id"]


def test_usage_is_billed_once_across_turns_and_flushed_on_close():
    config = _configured()
    first = _backend_events(config, _response(("one", True), billed_seconds=2.0))
    second = _backend_events(config, _response(("two", True), billed_seconds=5.0))
    assert first[-1]["usage"] == {"type": "duration", "seconds": 2.0}
    assert second[-1]["usage"] == {"type": "duration", "seconds": 3.0}
    assert config.unbilled_usage_on_session_close(MODEL) is None
    assert _backend_events(config, _response(billed_seconds=6.5)) == []
    assert config.unbilled_usage_on_session_close(MODEL) == {"type": "duration", "seconds": 1.5}
    assert config.unbilled_usage_on_session_close(MODEL) is None


class _NullBackend:
    async def __aenter__(self) -> "_NullBackend":
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def send(self, message: str | bytes) -> None:
        return None

    async def recv(self, decode: bool | None = None) -> str | bytes:
        return ""

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_open_backend_targets_the_regional_speech_endpoint():
    targets: list[SpeechStreamingTarget] = []

    def factory(target: SpeechStreamingTarget) -> RealtimeBackend:
        targets.append(target)
        return _NullBackend()

    config = VertexChirpRealtimeConfig(access_token="token", project="proj-1", location=None, backend_factory=factory)
    url = config.get_complete_url(None, "vertex_ai/chirp_3")
    assert url == "us-speech.googleapis.com"
    assert config.validate_environment({}, MODEL, "https://" + url) == {}
    backend = await config.open_backend(url, {})
    assert isinstance(backend, _NullBackend)
    assert targets == [
        SpeechStreamingTarget(
            api_endpoint="us-speech.googleapis.com",
            recognizer="projects/proj-1/locations/us/recognizers/_",
            access_token="token",
        )
    ]


@pytest.mark.parametrize(
    ("location", "api_base", "endpoint"),
    [
        ("global", None, "speech.googleapis.com"),
        ("europe-west4", None, "europe-west4-speech.googleapis.com"),
        ("us", "https://speech-proxy.internal:8443/v2", "speech-proxy.internal:8443"),
    ],
)
def test_get_complete_url_honors_location_and_api_base(location: str, api_base: str | None, endpoint: str):
    assert _config(location).get_complete_url(api_base, MODEL) == endpoint


def test_get_complete_url_rejects_non_speech_models():
    with pytest.raises(ValueError, match="Unsupported Speech-to-Text streaming model"):
        _config().get_complete_url(None, "gemini-live-2.5-flash")


@pytest.mark.parametrize("location", ["bad loc", "../us"])
def test_invalid_locations_are_rejected_up_front(location: str):
    with pytest.raises(VertexAIError):
        _config(location)


@pytest.mark.parametrize(
    ("previous", "current", "delta"),
    [
        ("", "hello", "hello"),
        ("hello", "hello world", " world"),
        ("hello", "Hello, world", " world"),
        ("hello world", "hello world", ""),
        ("hello there", "hello world", " world"),
        ("hello world", "hello", ""),
    ],
)
def test_new_words(previous: str, current: str, delta: str):
    assert new_words(previous, current) == delta

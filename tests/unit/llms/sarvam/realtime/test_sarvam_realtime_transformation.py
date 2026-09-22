import base64
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.sarvam.realtime.transformation import (
    DEFAULT_SARVAM_REALTIME_MODEL,
    SUPPORTED_LANGUAGE_CODES,
    SarvamProtocolError,
    SarvamRealtimeConfig,
    build_sarvam_connection,
    normalize_language_code,
)
from litellm.types.realtime import RealtimeResponseTransformInput
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

EMPTY_TRANSFORM_INPUT: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}
MODEL: Final = f"sarvam/{DEFAULT_SARVAM_REALTIME_MODEL}"

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")
ONE_SECOND_OF_SILENCE: Final = b"\x00\x00" * 16_000


def _session_update(
    rate: int | None = 16_000,
    turn_detection: object = "server_vad",
    language: str | None = None,
    model: str | None = None,
) -> str:
    """``turn_detection="omit"`` leaves the key out entirely, the way a client that only changes its language
    sends a partial update; ``None`` sends an explicit null, which disables turn detection."""
    transcription: Final[dict[str, str]] = {
        **({"model": model} if model is not None else {}),
        **({"language": language} if language is not None else {}),
    }
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        **({"format": {"type": "audio/pcm", "rate": rate}} if rate is not None else {}),
                        **(
                            {}
                            if turn_detection == "omit"
                            else {"turn_detection": None if turn_detection is None else {"type": turn_detection}}
                        ),
                        "transcription": transcription,
                    }
                },
            },
        }
    )


def _append(audio: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")})


def _client_event(event_type: str) -> str:
    return json.dumps({"type": event_type})


def _configured(
    *, api_base: str | None = None, turn_detection: object = "server_vad", rate: int = 16_000
) -> SarvamRealtimeConfig:
    config: Final = SarvamRealtimeConfig()
    config.get_complete_url(api_base, MODEL)
    config.validate_environment({}, MODEL, api_key="secret-key")
    config.transform_realtime_request(_session_update(rate=rate, turn_detection=turn_detection), MODEL)
    return config


def _backend(stt: SarvamRealtimeConfig, **frame: object) -> list[dict[str, object]]:
    response: Final = stt.transform_realtime_response(json.dumps(frame), MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)[
        "response"
    ]
    assert isinstance(response, list)
    return response


def _sent(stt: SarvamRealtimeConfig, message: str) -> list[dict[str, object]]:
    return [json.loads(frame) for frame in stt.transform_realtime_request(message, MODEL)]


def test_connection_url_pins_the_fields_sarvam_fixes_at_connect_time():
    connection = build_sarvam_connection(None, MODEL)

    assert connection.url.startswith("wss://api.sarvam.ai/speech-to-text-realtime/ws?")
    assert connection.sample_rate == 16_000
    query = dict(part.split("=", 1) for part in connection.url.split("?", 1)[1].split("&"))
    assert query["language_code"] == "auto"
    assert query["encoding"] == "linear16"
    assert query["sample_rate"] == "16000"
    assert query["model"] == "saaras%3Av3-realtime"


@pytest.mark.parametrize(
    ("api_base", "expected_rate"),
    [
        ("https://api.sarvam.ai", 16_000),
        ("wss://api.sarvam.ai?sample_rate=8000", 8_000),
        ("ws://localhost:8123?sample_rate=16000", 16_000),
    ],
)
def test_api_base_supplies_the_connection_only_sample_rate(api_base: str, expected_rate: int):
    connection = build_sarvam_connection(api_base, MODEL)

    assert connection.sample_rate == expected_rate
    assert f"sample_rate={expected_rate}" in connection.url


@pytest.mark.parametrize(
    "api_base",
    [
        "wss://api.sarvam.ai?sample_rate=24000",
        "wss://api.sarvam.ai?sample_rate=16000&sample_rate=8000",
        "wss://api.sarvam.ai?sample_rate=sixteen",
    ],
)
def test_unsupported_connection_sample_rate_is_rejected(api_base: str):
    with pytest.raises(ValueError, match="sample_rate"):
        build_sarvam_connection(api_base, MODEL)


@pytest.mark.parametrize("api_base", ["ftp://api.sarvam.ai", "wss://user:pass@api.sarvam.ai", "not-a-url"])
def test_unusable_api_base_is_rejected(api_base: str):
    with pytest.raises(ValueError, match="api_base"):
        build_sarvam_connection(api_base, MODEL)


@pytest.mark.parametrize("api_base", ["ws://stt.internal.example", "http://198.51.100.7:8123"])
def test_cleartext_to_a_remote_host_is_rejected_because_the_key_is_a_connection_header(api_base: str):
    with pytest.raises(ValueError, match="loopback"):
        build_sarvam_connection(api_base, MODEL)


@pytest.mark.parametrize(
    ("api_base", "expected_netloc"),
    [
        ("ws://localhost:8123", "localhost:8123"),
        ("ws://127.0.0.1:8123", "127.0.0.1:8123"),
        ("ws://[::1]:8123", "[::1]:8123"),
    ],
)
def test_a_loopback_test_double_stays_reachable_over_cleartext(api_base: str, expected_netloc: str):
    url = build_sarvam_connection(api_base, MODEL).url

    assert url.startswith(f"ws://{expected_netloc}/speech-to-text-realtime/ws?")


def test_a_model_the_registry_does_not_serve_on_the_realtime_endpoint_is_rejected():
    with pytest.raises(SarvamProtocolError, match="model"):
        build_sarvam_connection(None, "sarvam/whisper-large")


def test_a_model_registered_for_the_realtime_endpoint_is_accepted_without_a_code_change():
    """Support reads the model registry, so a new Sarvam realtime model ships as a cost map entry."""
    import litellm

    litellm.register_model(
        {
            "sarvam/saaras:v9-realtime": {
                "litellm_provider": "sarvam",
                "mode": "audio_transcription",
                "supported_endpoints": ["/v1/realtime"],
            }
        }
    )

    assert build_sarvam_connection(None, "sarvam/saaras:v9-realtime").model == "saaras:v9-realtime"


def test_a_sarvam_chat_model_is_not_served_on_the_realtime_endpoint():
    """The registry entry has to say audio_transcription on /v1/realtime, not merely exist under sarvam/."""
    with pytest.raises(SarvamProtocolError, match="model"):
        build_sarvam_connection(None, "sarvam/sarvam-m")


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("hi", "hi-IN"), ("hi-IN", "hi-IN"), ("HI-in", "hi-IN"), ("en_IN", "en-IN"), ("auto", "auto")],
)
def test_language_normalization_maps_onto_the_documented_codes(requested: str, expected: str):
    assert normalize_language_code(requested) == expected
    assert expected == "auto" or expected in SUPPORTED_LANGUAGE_CODES


@pytest.mark.parametrize("requested", ["", "  ", "klingon", "zz-ZZ"])
def test_unsupported_language_is_rejected(requested: str):
    with pytest.raises(SarvamProtocolError, match="language"):
        normalize_language_code(requested)


def test_api_key_travels_in_sarvams_subscription_header():
    headers = SarvamRealtimeConfig().validate_environment({"x-existing": "kept"}, MODEL, api_key=" secret-key ")

    assert headers == {"x-existing": "kept", "api-subscription-key": "secret-key"}


@pytest.mark.parametrize("api_key", [None, "   "])
def test_missing_api_key_is_rejected(api_key: str | None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key"):
        SarvamRealtimeConfig().validate_environment({}, MODEL, api_key=api_key)


def test_session_language_is_applied_with_a_config_update_rather_than_at_connect_time():
    config = _configured()

    assert _sent(config, _session_update(language="hi")) == [{"event": "config.update", "language_code": "hi-IN"}]


def test_repeating_the_same_session_settings_sends_nothing():
    config = _configured()
    config.transform_realtime_request(_session_update(language="hi"), MODEL)

    assert config.transform_realtime_request(_session_update(language="hi-IN"), MODEL) == ()


def test_disabling_turn_detection_switches_sarvam_to_manual_endpointing():
    config = _configured()

    assert _sent(config, _session_update(turn_detection=None)) == [{"event": "config.update", "endpointing": "manual"}]


@pytest.mark.parametrize(
    ("update", "expected_message"),
    [
        (_session_update(rate=24_000), "16000 Hz"),
        (_session_update(turn_detection="semantic_vad"), "turn detection"),
        (_session_update(model="sarvam/saaras:v4"), "cannot be changed"),
    ],
)
def test_client_session_settings_sarvam_cannot_honor_are_rejected(update: str, expected_message: str):
    config = _configured()

    with pytest.raises(SarvamProtocolError, match=expected_message):
        config.transform_realtime_request(update, MODEL)


def test_audio_is_forwarded_as_a_sarvam_audio_input_frame():
    config = _configured()

    frames = _sent(config, _append(ONE_SECOND_OF_SILENCE))

    assert len(frames) == 1
    assert frames[0]["event"] == "audio_input"
    assert base64.b64decode(str(frames[0]["audio"])) == ONE_SECOND_OF_SILENCE


def test_commit_is_a_no_op_under_vad_endpointing():
    config = _configured()
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE), MODEL)

    assert config.transform_realtime_request(_client_event("input_audio_buffer.commit"), MODEL) == ()


def test_manual_endpointing_brackets_each_utterance_with_speech_events():
    config = _configured(turn_detection=None)

    first = _sent(config, _append(ONE_SECOND_OF_SILENCE))
    second = _sent(config, _append(ONE_SECOND_OF_SILENCE))
    committed = _sent(config, _client_event("input_audio_buffer.commit"))
    next_utterance = _sent(config, _append(ONE_SECOND_OF_SILENCE))

    assert [frame["event"] for frame in first] == ["speech_start", "audio_input"]
    assert [frame["event"] for frame in second] == ["audio_input"]
    assert [frame["event"] for frame in committed] == ["speech_end", "flush"]
    assert [frame["event"] for frame in next_utterance] == ["speech_start", "audio_input"]


def test_committing_without_audio_sends_nothing_under_manual_endpointing():
    config = _configured(turn_detection=None)

    assert config.transform_realtime_request(_client_event("input_audio_buffer.commit"), MODEL) == ()


def test_buffer_end_closes_the_sarvam_session():
    config = _configured(turn_detection=None)
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE), MODEL)

    assert [frame["event"] for frame in _sent(config, _client_event("input_audio_buffer.end"))] == [
        "speech_end",
        "flush",
        "end",
    ]


def test_buffer_clear_is_dropped_because_sarvam_cannot_discard_sent_audio():
    config = _configured()
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE), MODEL)

    assert config.transform_realtime_request(_client_event("input_audio_buffer.clear"), MODEL) == ()


def test_session_begin_becomes_an_openai_transcription_session():
    config = _configured(api_base="wss://api.sarvam.ai?sample_rate=8000", rate=8_000)
    config.transform_realtime_request(_session_update(rate=8_000, language="ta"), MODEL)

    events = _backend(config, event="session.begin", request_id="req-42", config={})

    assert len(events) == 1
    session = events[0]["session"]
    assert events[0]["type"] == "session.created"
    assert session["id"] == "req-42"
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 8_000}
    assert session["audio"]["input"]["transcription"] == {"model": "saaras:v3-realtime", "language": "ta-IN"}
    assert session["audio"]["input"]["turn_detection"] == {"type": "server_vad"}


def test_a_later_update_that_only_changes_the_language_keeps_manual_endpointing():
    """Turn detection left out of a partial update means unchanged. Reverting to VAD here would silently turn
    the client's audio buffer commits into no-ops."""
    config = _configured(turn_detection=None)

    language_only = _sent(config, _session_update(turn_detection="omit", language="ta"))
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE), MODEL)
    committed = _sent(config, _client_event("input_audio_buffer.commit"))

    assert language_only == [{"event": "config.update", "language_code": "ta-IN"}]
    assert [frame["event"] for frame in committed] == ["speech_end", "flush"]


def test_applied_settings_reach_the_client_as_a_session_updated_event():
    """Sarvam opens on its own defaults, so a client that configures the session after session.created only
    learns the language and turn detection that took effect from the config.updated acknowledgement."""
    config = _configured()
    _backend(config, event="session.begin", request_id="req-42", config={})
    config.transform_realtime_request(_session_update(language="hi", turn_detection=None), MODEL)

    events = _backend(config, event="config.updated", applied=["language_code", "endpointing"])

    assert [event["type"] for event in events] == ["session.updated"]
    session = events[0]["session"]
    assert session["id"] == "req-42"
    assert session["audio"]["input"]["transcription"]["language"] == "hi-IN"
    assert session["audio"]["input"]["turn_detection"] is None


def test_an_acknowledgement_before_the_session_exists_is_dropped():
    config = _configured()

    assert _backend(config, event="config.updated", applied=["language_code"]) == []


def test_session_begin_reports_manual_endpointing_as_disabled_turn_detection():
    config = _configured(turn_detection=None)

    events = _backend(config, event="session.begin", request_id="req-42", config={})

    assert events[0]["session"]["audio"]["input"]["turn_detection"] is None


def test_session_begin_without_a_request_id_is_rejected():
    config = _configured()

    with pytest.raises(SarvamProtocolError, match="request_id"):
        _backend(config, event="session.begin", config={})


def test_vad_events_become_openai_speech_events_once_per_utterance():
    config = _configured()

    started = _backend(config, event="vad.speech_start", utterance_idx=0, confidence=0.92)
    repeated = _backend(config, event="vad.speech_start", utterance_idx=0, confidence=0.93)
    stopped = _backend(config, event="vad.speech_end", utterance_idx=0, confidence=0.88)

    assert [event["type"] for event in started] == ["input_audio_buffer.speech_started"]
    assert repeated == []
    assert [event["type"] for event in stopped] == ["input_audio_buffer.speech_stopped"]
    assert stopped[0]["item_id"] == started[0]["item_id"]


def test_each_utterance_gets_its_own_item_id():
    config = _configured()

    first = _backend(config, event="vad.speech_start", utterance_idx=0, confidence=0.9)
    second = _backend(config, event="vad.speech_start", utterance_idx=1, confidence=0.9)

    assert first[0]["item_id"] != second[0]["item_id"]


def test_growing_partials_become_incremental_deltas():
    config = _configured()

    first = _backend(config, event="transcript.partial", utterance_idx=0, text="Four score")
    second = _backend(config, event="transcript.partial", utterance_idx=0, text="Four score and seven")

    assert [event["type"] for event in first] == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
    ]
    assert first[-1]["delta"] == "Four score"
    assert [event["delta"] for event in second] == [" and seven"]


def test_a_revised_partial_never_replays_words_the_client_already_has():
    """Sarvam revises partials mid-utterance: a shorter revision must not make the next growth re-send a
    prefix the client already rendered. Regression for text observed on a live `saaras:v3-realtime` session."""
    config = _configured()

    _backend(config, event="transcript.partial", utterance_idx=0, text="Four score and seven years ago")
    revised = _backend(config, event="transcript.partial", utterance_idx=0, text="Four score")
    regrown = _backend(config, event="transcript.partial", utterance_idx=0, text="Four score and seven years ago, our")

    assert revised == []
    assert [event["delta"] for event in regrown] == [", our"]


def test_unchanged_partials_emit_nothing():
    config = _configured()
    _backend(config, event="transcript.partial", utterance_idx=0, text="Four score")

    assert _backend(config, event="transcript.partial", utterance_idx=0, text="Four score") == []


def test_final_transcript_completes_the_item_with_the_authoritative_text():
    config = _configured()
    _backend(config, event="transcript.partial", utterance_idx=0, text="Four score and seven")

    events = _backend(config, event="transcript.final", utterance_idx=0, text="Four score and seven years ago.")

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert events[-1]["transcript"] == "Four score and seven years ago."


def test_a_completed_utterance_ignores_later_frames_for_the_same_index():
    config = _configured()
    _backend(config, event="transcript.final", utterance_idx=0, text="done")

    assert _backend(config, event="transcript.final", utterance_idx=0, text="again") == []
    assert _backend(config, event="transcript.partial", utterance_idx=0, text="done and more") == []


@pytest.mark.parametrize("frame", [{"utterance_idx": -1}, {"utterance_idx": "0"}, {}])
def test_backend_events_without_a_usable_utterance_index_are_rejected(frame: dict[str, object]):
    config = _configured()

    with pytest.raises(SarvamProtocolError, match="utterance_idx"):
        _backend(config, event="transcript.partial", text="hello", **frame)


def test_session_end_bills_the_duration_sarvam_reports():
    config = _configured()
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE), MODEL)

    assert _backend(config, event="session.end", request_id="req-42", audio_duration_s=17.58) == []
    assert config.unbilled_usage_on_session_close(MODEL) == {"type": "duration", "seconds": 17.58}


def test_a_session_that_ends_before_sarvam_reports_bills_the_audio_that_was_forwarded():
    config = _configured()
    config.transform_realtime_request(_append(ONE_SECOND_OF_SILENCE * 3), MODEL)

    assert config.unbilled_usage_on_session_close(MODEL) == {"type": "duration", "seconds": 3.0}


def test_a_session_with_no_audio_bills_nothing():
    config = _configured()

    assert config.unbilled_usage_on_session_close(MODEL) is None


def test_usage_is_only_reported_once():
    config = _configured()
    _backend(config, event="session.end", request_id="req-42", audio_duration_s=17.58)
    config.unbilled_usage_on_session_close(MODEL)

    assert config.unbilled_usage_on_session_close(MODEL) is None


def test_provider_errors_reach_the_client_as_openai_error_events():
    config = _configured()

    events = _backend(config, event="error", code="invalid_config", is_fatal=False, message="language not supported")

    assert events[0]["type"] == "error"
    assert "invalid_config" in events[0]["error"]["message"]
    assert "language not supported" in events[0]["error"]["message"]


@pytest.mark.parametrize("event", ["pong", "something.new"])
def test_backend_events_without_an_openai_equivalent_are_dropped(event: str):
    config = _configured()

    assert _backend(config, event=event) == []


def test_realtime_requests_before_the_socket_url_is_built_are_rejected():
    config = SarvamRealtimeConfig()

    with pytest.raises(SarvamProtocolError, match="url"):
        config.transform_realtime_request(_session_update(), MODEL)


def test_the_provider_registry_serves_the_sarvam_realtime_config():
    config = ProviderConfigManager.get_provider_realtime_config(
        model=DEFAULT_SARVAM_REALTIME_MODEL, provider=LlmProviders.SARVAM
    )

    assert isinstance(config, SarvamRealtimeConfig)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", ["saaras:v3-realtime", "saaras:v4"])
def test_the_realtime_models_are_registered_as_transcription_models(model: str):
    """Sarvam prices in INR, which this registry cannot hold, so the entry carries no rate. It still has to
    describe the model, or the endpoint cannot resolve it and sessions log neither duration nor cost."""
    import litellm

    registry = litellm.get_model_info(model=f"sarvam/{model}", custom_llm_provider="sarvam")

    assert registry["mode"] == "audio_transcription"
    assert "/v1/realtime" in (registry.get("supported_endpoints") or ())


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_priced_deployment_bills_the_seconds_the_session_reported():
    """An operator who converts Sarvam's INR rate themselves gets that rate applied to the reported audio."""
    import litellm
    from litellm.cost_calculator import handle_realtime_transcription_cost_calculation

    litellm.register_model(
        {"sarvam/saaras:v3-realtime": {"litellm_provider": "sarvam", "input_cost_per_second": 0.00009}}
    )
    completed = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"type": "duration", "seconds": 18.3},
        }
    ]

    cost = handle_realtime_transcription_cost_calculation(
        results=completed, custom_llm_provider="sarvam", litellm_model_name="saaras:v3-realtime"
    )

    assert cost == pytest.approx(18.3 * 0.00009)

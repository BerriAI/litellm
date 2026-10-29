import json
from typing import Final
from urllib.parse import parse_qs

import pytest

from litellm.llms.elevenlabs.realtime.transformation import (
    SCRIBE_REALTIME_MODEL,
    SUPPORTED_SAMPLE_RATES,
    ElevenLabsProtocolError,
    ElevenLabsSessionConfig,
    normalize_language,
    parse_session_update,
    realtime_options_from_params,
    session_created_event,
)
from litellm.types.llms.elevenlabs import ElevenLabsRealtimeOptions

NO_OPTIONS: Final = ElevenLabsRealtimeOptions()


def _session_update(
    rate: int = 24_000,
    turn_detection: str | None = "server_vad",
    language: str | None = None,
    model: str = SCRIBE_REALTIME_MODEL,
    audio_type: str = "audio/pcm",
    channels: int | None = None,
) -> str:
    audio_format: Final = {"type": audio_type, "rate": rate} | ({} if channels is None else {"channels": channels})
    transcription: Final = {"model": model} | ({} if language is None else {"language": language})
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": audio_format,
                        "turn_detection": None if turn_detection is None else {"type": turn_detection},
                        "transcription": transcription,
                    }
                },
            },
        }
    )


def _query(config: ElevenLabsSessionConfig) -> dict[str, list[str]]:
    return parse_qs(config.query())


@pytest.mark.parametrize("rate", sorted(SUPPORTED_SAMPLE_RATES))
def test_session_rate_selects_the_matching_pcm_audio_format(rate: int):
    config: Final = parse_session_update(_session_update(rate=rate), NO_OPTIONS)

    assert _query(config)["audio_format"] == [f"pcm_{rate}"]
    assert config.bytes_per_second == rate * 2


def test_server_vad_selects_vad_and_null_turn_detection_selects_manual_commit():
    vad: Final = parse_session_update(_session_update(turn_detection="server_vad"), NO_OPTIONS)
    manual: Final = parse_session_update(_session_update(turn_detection=None), NO_OPTIONS)

    assert _query(vad)["commit_strategy"] == ["vad"]
    assert _query(manual)["commit_strategy"] == ["manual"]


def test_semantic_vad_falls_back_to_vad_only_under_drop_params():
    with pytest.raises(ElevenLabsProtocolError, match="drop_params"):
        parse_session_update(_session_update(turn_detection="semantic_vad"), NO_OPTIONS)

    fallback: Final = parse_session_update(_session_update(turn_detection="semantic_vad"), NO_OPTIONS, drop_params=True)
    assert _query(fallback)["commit_strategy"] == ["vad"]


@pytest.mark.parametrize("model", [SCRIBE_REALTIME_MODEL, f"elevenlabs/{SCRIBE_REALTIME_MODEL}"])
def test_query_pins_the_scribe_realtime_model(model: str):
    assert _query(parse_session_update(_session_update(model=model), NO_OPTIONS))["model_id"] == [SCRIBE_REALTIME_MODEL]


def test_beta_pcm16_session_uses_the_default_rate_and_vad():
    beta: Final = json.dumps(
        {
            "type": "session.update",
            "session": {"input_audio_format": "pcm16", "input_audio_transcription": {"model": SCRIBE_REALTIME_MODEL}},
        }
    )
    query: Final = _query(parse_session_update(beta, NO_OPTIONS))

    assert query["audio_format"] == ["pcm_24000"]
    assert query["commit_strategy"] == ["vad"]


@pytest.mark.parametrize(("source", "expected"), [("en", "en"), ("EN-us", "en"), ("pt_BR", "pt"), ("fil", "fil")])
def test_language_becomes_an_iso_language_code(source: str, expected: str):
    assert normalize_language(source) == expected
    assert _query(parse_session_update(_session_update(language=source), NO_OPTIONS))["language_code"] == [expected]


@pytest.mark.parametrize("language", ["", " ", "english", "e1"])
def test_invalid_language_is_rejected(language: str):
    with pytest.raises(ElevenLabsProtocolError, match="ISO 639"):
        normalize_language(language)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (_session_update(audio_type="audio/pcmu"), "audio/pcm"),
        (_session_update(channels=2), "mono"),
        (_session_update(rate=11_025), "8000, 16000"),
        (_session_update(model="gpt-4o-transcribe"), "cannot be changed"),
        (json.dumps({"type": "session.update", "session": {"type": "conversation"}}), "transcription sessions only"),
    ],
)
def test_session_rejects_unsupported_settings(update: str, message: str):
    with pytest.raises(ElevenLabsProtocolError, match=message):
        parse_session_update(update, NO_OPTIONS)


def test_options_are_appended_to_the_query():
    options: Final = ElevenLabsRealtimeOptions(
        keyterms=("GBS", "Portainer"),
        secondary_languages=("es",),
        no_verbatim=True,
        filter_background_audio=False,
        vad_threshold=0.4,
        vad_silence_threshold_secs=1.2,
        min_speech_duration_ms=100,
        min_silence_duration_ms=250,
        enable_logging=False,
    )
    query: Final = _query(parse_session_update(_session_update(), options))

    assert query["keyterms"] == ["GBS", "Portainer"]
    assert query["secondary_languages"] == ["es"]
    assert query["no_verbatim"] == ["true"]
    assert query["filter_background_audio"] == ["false"]
    assert query["vad_threshold"] == ["0.4"]
    assert query["vad_silence_threshold_secs"] == ["1.2"]
    assert query["min_speech_duration_ms"] == ["100"]
    assert query["min_silence_duration_ms"] == ["250"]
    assert query["enable_logging"] == ["false"]


def test_unset_options_are_not_sent():
    assert set(_query(parse_session_update(_session_update(), NO_OPTIONS))) == {
        "model_id",
        "audio_format",
        "commit_strategy",
    }


def test_options_from_params_ignore_unrelated_litellm_params():
    options: Final = realtime_options_from_params(
        {"api_key": "secret", "rpm": 10, "keyterms": ["GBS"], "no_verbatim": True, "vad_threshold": None}
    )

    assert options == ElevenLabsRealtimeOptions(keyterms=("GBS",), no_verbatim=True)


@pytest.mark.parametrize(
    "params",
    [{"keyterms": "GBS"}, {"no_verbatim": "yes"}, {"min_speech_duration_ms": -1}, {"vad_threshold": -0.1}],
)
def test_options_from_params_reject_invalid_values(params: dict[str, object]):
    with pytest.raises(ValueError, match="Invalid ElevenLabs realtime options"):
        realtime_options_from_params(params)


def test_session_created_event_reports_the_effective_transcription_session():
    config: Final = parse_session_update(_session_update(rate=16_000, turn_detection=None, language="es"), NO_OPTIONS)
    event: Final = session_created_event(config, "sess_1")

    assert event["type"] == "session.created"
    assert event["session"]["id"] == "sess_1"
    audio_input: Final = event["session"]["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 16_000}
    assert audio_input["transcription"] == {"model": SCRIBE_REALTIME_MODEL, "language": "es"}
    assert audio_input["turn_detection"] is None

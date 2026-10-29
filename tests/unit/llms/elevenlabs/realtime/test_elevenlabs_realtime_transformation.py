import itertools
import json
from collections.abc import Callable
from typing import Final, get_args
from urllib.parse import parse_qs

import pytest

from litellm.llms.elevenlabs.realtime.transformation import (
    SCRIBE_REALTIME_MODEL,
    SUPPORTED_SAMPLE_RATES,
    ElevenLabsEventTransformer,
    ElevenLabsProtocolError,
    ElevenLabsSessionConfig,
    default_session_config,
    normalize_language,
    parse_server_message,
    parse_session_update,
    realtime_options_from_params,
    session_created_event,
)
from litellm.types.llms.elevenlabs import ElevenLabsErrorType, ElevenLabsRealtimeOptions

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


DEFAULT_CONFIG: Final = default_session_config(NO_OPTIONS)


def _item_ids() -> Callable[[], str]:
    counter: Final = itertools.count()
    return lambda: f"item_{next(counter)}"


def _frame(message_type: str, **fields: object) -> str:
    return json.dumps({"message_type": message_type, **fields})


def _transform(transformer: ElevenLabsEventTransformer, frame: str) -> tuple[object, ...]:
    message: Final = parse_server_message(frame)
    return () if message is None else transformer.transform(message, DEFAULT_CONFIG)


def _events(transformer: ElevenLabsEventTransformer, *frames: str) -> list[dict[str, object]]:
    transformed: Final = itertools.chain.from_iterable(_transform(transformer, frame) for frame in frames)
    return [dict(event) for event in transformed]


def _summary(events: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return [(event["type"], event.get("item_id"), event.get("delta", event.get("transcript"))) for event in events]


def test_cumulative_partials_start_the_turn_once_and_emit_only_new_text():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="Hello"),
        _frame("partial_transcript", text="Hello wor"),
        _frame("partial_transcript", text="Hello world"),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hello"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " wor"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "ld"),
    ]


def test_revised_partial_emits_nothing_and_the_commit_carries_the_final_text():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="Hello wor"),
        _frame("partial_transcript", text="Hello war"),
        _frame("partial_transcript", text="Hello world"),
        _frame("committed_transcript", text="Hello world."),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hello wor"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "ld"),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "Hello world."),
    ]


def test_each_commit_closes_its_turn_and_the_next_partial_opens_a_new_item():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="One"),
        _frame("committed_transcript", text="One."),
        _frame("partial_transcript", text="Two"),
        _frame("committed_transcript", text="Two."),
    )

    completed: Final = [
        event for event in events if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert [(event["item_id"], event["transcript"]) for event in completed] == [("item_0", "One."), ("item_1", "Two.")]


def test_commit_without_partials_still_reports_a_whole_turn():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()), _frame("committed_transcript", text="Hi")
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "Hi"),
    ]


def test_empty_partials_and_empty_commits_outside_a_turn_are_ignored():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text=""),
        _frame("committed_transcript", text=""),
    )

    assert events == []


def test_each_completion_bills_only_the_audio_recorded_since_the_previous_one():
    transformer: Final = ElevenLabsEventTransformer(new_item_id=_item_ids())
    transformer.record_audio(1.5)
    first: Final = _events(transformer, _frame("committed_transcript", text="One."))
    transformer.record_audio(0.25)
    second: Final = _events(transformer, _frame("committed_transcript", text="Two."))

    assert first[-1]["usage"] == {"type": "duration", "seconds": 1.5}
    assert second[-1]["usage"] == {"type": "duration", "seconds": 0.25}
    assert transformer.take_unbilled_usage() is None


def test_trailing_audio_is_billed_once_at_close():
    transformer: Final = ElevenLabsEventTransformer()
    transformer.record_audio(2.0)

    assert transformer.take_unbilled_usage() == {"type": "duration", "seconds": 2.0}
    assert transformer.take_unbilled_usage() is None


def test_session_started_reports_session_created_with_the_upstream_session_id():
    events: Final = _events(ElevenLabsEventTransformer(), _frame("session_started", session_id="el_sess_1", config={}))

    assert [event["type"] for event in events] == ["session.created"]
    session: Final = events[0]["session"]
    assert isinstance(session, dict)
    assert session["id"] == "el_sess_1"


@pytest.mark.parametrize("error_type", get_args(ElevenLabsErrorType))
def test_provider_errors_name_the_error_type_without_upstream_detail(error_type: str):
    events: Final = _events(ElevenLabsEventTransformer(), _frame(error_type, error="private upstream detail"))

    assert len(events) == 1
    assert events[0]["type"] == "error"
    message: Final = json.dumps(events[0])
    assert error_type in message
    assert "private upstream detail" not in message


def test_warnings_and_unknown_messages_are_not_forwarded():
    assert _events(ElevenLabsEventTransformer(), _frame("warning", warning="slow down")) == []
    assert parse_server_message(_frame("committed_transcript_with_timestamps", text="x")) is None
    assert parse_server_message(json.dumps({"message_type": ["not", "a", "string"]})) is None


@pytest.mark.parametrize("payload", [_frame("partial_transcript"), "[]", "not json"])
def test_malformed_server_messages_raise_the_protocol_error(payload: str):
    with pytest.raises(ElevenLabsProtocolError):
        parse_server_message(payload)


def test_bytes_payloads_are_decoded():
    assert parse_server_message(_frame("partial_transcript", text="hi").encode()) is not None

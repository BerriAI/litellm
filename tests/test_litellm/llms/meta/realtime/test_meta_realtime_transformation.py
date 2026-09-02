import json

import pytest

from litellm.llms.meta.realtime.transformation import (
    MUSE_MODEL,
    MuseEventTransformer,
    MuseProtocolError,
    encode_event,
    normalize_language,
    parse_session_update,
    session_created_event,
    session_updated_event,
)


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


def test_beta_session_builds_authenticated_24khz_handshake_with_hints():
    config = parse_session_update(
        _event(
            "session.update",
            session={
                "type": "transcription",
                "input_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
                "input_audio_transcription": {
                    "model": "meta/muse-voice-transcribe-1.0",
                    "language": "en-US",
                    "language_bias": ["Spanish", "english", "French"],
                    "keywords": [" Muse ", "LiteLLM", "Muse"],
                    "prompt": "must not become a keyword",
                },
            },
        ),
        "meta/muse-voice-transcribe-1.0",
    )

    assert config.sample_rate == 24_000
    assert config.packet_bytes == 3_840
    assert config.mode == "ENDPOINTING"
    assert config.language_bias == ("English", "Spanish", "French")
    assert config.keywords == ("Muse", "LiteLLM")
    assert config.handshake("Bearer token") == {
        "mode": "ENDPOINTING",
        "authorization": {"accessToken": "Bearer token"},
        "audioEncoding": "PCM_24KHZ",
        "model": MUSE_MODEL,
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": True,
        "keywords": ["Muse", "LiteLLM"],
        "languageBias": ["English", "Spanish", "French"],
    }
    assert "must not become a keyword" not in json.dumps(config.handshake("Bearer token"))


def test_ga_session_accepts_16khz_mono_push_to_talk():
    config = parse_session_update(
        _event(
            "session.update",
            session={
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000, "channels": 1},
                        "turn_detection": None,
                        "transcription": {"model": MUSE_MODEL, "language": "zh-Hans"},
                    }
                },
            },
        ),
        MUSE_MODEL,
    )

    assert config.sample_rate == 16_000
    assert config.packet_bytes == 2_560
    assert config.mode == "PUSH_TO_TALK"
    assert config.language_bias == ("Mandarin Chinese",)
    assert config.handshake("Bearer token")["audioEncoding"] == "PCM_16KHZ"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("EN_us", "English"),
        ("mandarin chinese", "Mandarin Chinese"),
        ("fil-PH", "Tagalog"),
        ("iw-IL", "Hebrew"),
        ("pt-BR", "Portuguese"),
    ],
)
def test_language_normalization_uses_official_muse_names(source: str, expected: str):
    assert normalize_language(source) == expected


@pytest.mark.parametrize(
    ("session", "message"),
    [
        ({"input_audio_format": "g711_ulaw"}, "requires pcm16"),
        ({"audio": {"input": {"format": {"type": "audio/pcm", "rate": 8000}}}}, "16000 Hz or 24000 Hz"),
        (
            {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000, "channels": 2}}}},
            "requires mono",
        ),
        (
            {"input_audio_format": "pcm16", "audio": {"input": {"format": {"type": "audio/pcm"}}}},
            "either beta or GA layout",
        ),
        ({"input_audio_transcription": {"model": "other-model"}}, "cannot be changed"),
        ({"input_audio_transcription": {"keywords": ["valid", ""]}}, "non-empty strings"),
        ({"input_audio_transcription": {"language": "xx"}}, "unsupported Muse Voice language"),
    ],
)
def test_session_rejects_unsupported_audio_model_and_hints(session: dict[str, object], message: str):
    with pytest.raises(MuseProtocolError, match=message):
        parse_session_update(_event("session.update", session={"type": "transcription", **session}), MUSE_MODEL)


def test_session_events_expose_openai_transcription_shapes():
    config = parse_session_update(
        _event(
            "session.update",
            session={
                "mode": "DIARIZATION",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": MUSE_MODEL, "language": "ja", "keywords": ["Meta"]},
                    }
                },
            },
        ),
        MUSE_MODEL,
    )

    created = session_created_event(MUSE_MODEL, "session-before-handshake")
    updated = session_updated_event(config, "provider-session")

    assert created["type"] == "session.created"
    assert created["session"]["type"] == "transcription"
    assert updated["type"] == "session.updated"
    assert updated["session"]["id"] == "provider-session"
    assert updated["session"]["audio"]["input"]["transcription"] == {
        "model": MUSE_MODEL,
        "language": "Japanese",
        "keywords": ["Meta"],
        "language_bias": ["Japanese"],
    }


def test_turnless_empty_silence_transcript_is_ignored():
    transformer = MuseEventTransformer()

    assert transformer.transform(_event("transcript", transcript="", final=True)) == ()


def test_transcript_without_speech_start_synthesizes_start_before_delta():
    transformer = MuseEventTransformer()

    events = transformer.transform(_event("transcript", turnId="turn-1", transcript="hello", final=False))

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
    ]


def test_cumulative_partials_emit_only_extensions_and_final_is_authoritative():
    transformer = MuseEventTransformer()

    started = transformer.transform(_event("speechStart", turnId="turn-1"))
    first = transformer.transform(_event("transcript", turnId="turn-1", transcript="hello", final=False))
    extension = transformer.transform(_event("transcript", turnId="turn-1", transcript="hello world", final=False))
    rewrite = transformer.transform(_event("transcript", turnId="turn-1", transcript="hullo world", final=False))
    assert transformer.transform(_event("speechComplete", turnId="turn-1", transcript="hullo world")) == ()
    completed = transformer.transform(_event("speechEnd", turnId="turn-1"))

    assert [event["type"] for event in started] == ["input_audio_buffer.speech_started"]
    assert first[0]["delta"] == "hello"
    assert extension[0]["delta"] == " world"
    assert rewrite == ()
    assert completed[0]["type"] == "input_audio_buffer.speech_stopped"
    assert completed[1]["type"] == "conversation.item.input_audio_transcription.completed"
    assert completed[1]["item_id"] == "turn-1"
    assert completed[1]["transcript"] == "hullo world"


def test_completed_transcript_waits_for_speech_stopped():
    transformer = MuseEventTransformer()

    transformer.transform(_event("speechStart", turnId="turn-1"))
    assert transformer.transform(_event("speechComplete", turnId="turn-1", transcript="done")) == ()

    released = transformer.transform(_event("speechEnd", turnId="turn-1"))
    assert [event["type"] for event in released] == [
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]


def test_overlapping_turns_are_emitted_in_provider_turn_order():
    transformer = MuseEventTransformer()

    transformer.transform(_event("speechStart", turnId="turn-a"))
    transformer.transform(_event("speechStart", turnId="turn-b"))
    assert transformer.transform(_event("transcript", turnId="turn-b", transcript="second", final=False)) == ()
    assert transformer.transform(_event("speechComplete", turnId="turn-a", transcript="first")) == ()
    released = transformer.transform(_event("speechEnd", turnId="turn-a"))

    assert [(event["type"], event["item_id"]) for event in released] == [
        ("input_audio_buffer.speech_stopped", "turn-a"),
        ("conversation.item.input_audio_transcription.completed", "turn-a"),
        ("input_audio_buffer.speech_started", "turn-b"),
        ("conversation.item.input_audio_transcription.delta", "turn-b"),
    ]
    assert transformer.transform(_event("speechComplete", turnId="turn-b", transcript="second final")) == ()
    final_b = transformer.transform(_event("speechEnd", turnId="turn-b"))
    assert final_b[0]["type"] == "input_audio_buffer.speech_stopped"
    assert final_b[1]["item_id"] == "turn-b"
    assert final_b[1]["transcript"] == "second final"


def test_committed_item_id_is_used_for_next_provider_turn():
    transformer = MuseEventTransformer()

    previous_item_id, item_id = transformer.commit_item()
    started = transformer.transform(_event("speechStart", turnId="provider-turn"))
    transformer.transform(_event("speechComplete", turnId="provider-turn", transcript="hello"))
    completed = transformer.transform(_event("speechEnd", turnId="provider-turn"))

    assert previous_item_id is None
    assert started[0]["item_id"] == item_id
    assert completed[-1]["item_id"] == item_id


def test_commit_after_speech_start_reuses_active_item_id():
    transformer = MuseEventTransformer()

    started = transformer.transform(_event("speechStart", turnId="provider-turn"))
    previous_item_id, item_id = transformer.commit_item()
    transformer.transform(_event("speechComplete", turnId="provider-turn", transcript="hello"))
    completed = transformer.transform(_event("speechEnd", turnId="provider-turn"))

    assert previous_item_id is None
    assert item_id == "provider-turn"
    assert started[0]["item_id"] == item_id
    assert completed[-1]["item_id"] == item_id


def test_speaker_and_positive_audio_progress_deltas_attach_to_next_completion():
    transformer = MuseEventTransformer()

    transformer.transform(_event("audioProgress", audioProcessedMs=1000))
    transformer.transform(_event("audioProgress", audioProcessedMs=750))
    transformer.transform(_event("audioProgress", audioProcessedMs=1600))
    transformer.transform(_event("speaker", turnId=42, label=" Speaker 2 "))
    transformer.transform(_event("speechComplete", turnId=42, transcript="hello"))
    completed = transformer.transform(_event("speechEnd", turnId=42))

    assert completed[-1]["speaker"] == "Speaker 2"
    assert completed[-1]["usage"] == {"type": "duration", "seconds": 1.6}
    assert transformer.take_unbilled_usage() is None


def test_trailing_audio_progress_is_returned_once():
    transformer = MuseEventTransformer()

    transformer.transform(_event("audioProgress", audioProcessedMs=250))

    assert transformer.take_unbilled_usage() == {"type": "duration", "seconds": 0.25}
    assert transformer.take_unbilled_usage() is None


def test_completed_turn_tombstone_suppresses_late_duplicates():
    transformer = MuseEventTransformer()

    transformer.transform(_event("speechComplete", turnId="turn-1", transcript="done"))

    assert transformer.transform(_event("speechComplete", turnId="turn-1", transcript="duplicate")) == ()
    assert transformer.transform(_event("speaker", turnId="turn-1", label="late")) == ()


def test_provider_error_is_sanitized_and_encodable():
    token = "private-token"
    provider_body = f"authorization failed for Bearer {token}"
    transformed = MuseEventTransformer().transform(
        _event("error", code="AUTH", message=provider_body, request={"accessToken": token})
    )

    encoded = encode_event(transformed[0])
    assert json.loads(encoded)["error"] == {
        "type": "server_error",
        "code": "provider_error",
        "message": "Meta Muse realtime transcription failed",
    }
    assert token not in encoded
    assert provider_body not in encoded

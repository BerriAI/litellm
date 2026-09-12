import base64
import itertools
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.meta.realtime.transformation import (
    DEFAULT_MUSE_REALTIME_URL,
    MUSE_MODEL,
    MetaRealtimeConfig,
    MuseEventTransformer,
    MuseProtocolError,
    MuseSessionConfig,
    build_muse_realtime_url,
    normalize_access_token,
    normalize_language,
    parse_session_update,
    session_created_event,
)
from litellm.types.llms.meta import MuseMode
from litellm.types.realtime import RealtimeResponseTransformInput

EMPTY_TRANSFORM_INPUT: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


def _ga_session_update(rate: int = 24_000, turn_detection: object = "server_vad") -> str:
    return _event(
        "session.update",
        session={
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "turn_detection": None if turn_detection is None else {"type": turn_detection},
                    "transcription": {"model": f"meta/{MUSE_MODEL}"},
                }
            },
        },
    )


def _configured(rate: int = 24_000, turn_detection: object = "server_vad", **kwargs: object) -> MetaRealtimeConfig:
    config = MetaRealtimeConfig(**kwargs)
    config.validate_environment({}, MUSE_MODEL, api_key="secret-token")
    config.transform_realtime_request(_ga_session_update(rate, turn_detection), MUSE_MODEL)
    return config


def _backend_events(config: MetaRealtimeConfig, payload: str) -> list[dict[str, object]]:
    response = config.transform_realtime_response(payload, MUSE_MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)["response"]
    assert isinstance(response, list)
    return response


def test_beta_session_translates_language_and_drops_non_openai_hints():
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
                    "prompt": "must not become a keyword",
                },
            },
        ),
        "meta/muse-voice-transcribe-1.0",
    )

    assert config.sample_rate == 24_000
    assert config.packet_bytes == 3_840
    assert config.mode == "ENDPOINTING"
    assert config.language_bias == ("English",)
    assert config.handshake("Bearer token") == {
        "mode": "ENDPOINTING",
        "authorization": {"accessToken": "Bearer token"},
        "audioEncoding": "PCM_24KHZ",
        "model": MUSE_MODEL,
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": True,
        "languageBias": ("English",),
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
    assert "languageBias" not in MuseSessionConfig(MUSE_MODEL, "ENDPOINTING", 24_000, ()).handshake("Bearer token")


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
        ({"input_audio_transcription": {"language": "xx"}}, "unsupported Muse Voice language"),
        ({"turn_detection": {"type": "semantic_vad"}}, "server_vad turn detection or null"),
        ({"type": "realtime", "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}, "server_vad"),
    ],
)
def test_session_rejects_unsupported_audio_model_and_hints(session: dict[str, object], message: str):
    with pytest.raises(MuseProtocolError, match=message):
        parse_session_update(_event("session.update", session={"type": "transcription", **session}), MUSE_MODEL)


def test_session_created_event_exposes_openai_transcription_shape():
    config = parse_session_update(
        _event(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": MUSE_MODEL, "language": "ja"},
                    }
                },
            },
        ),
        MUSE_MODEL,
    )

    created = session_created_event(config, "provider-session")

    assert created["type"] == "session.created"
    assert created["session"]["id"] == "provider-session"
    assert created["session"]["type"] == "transcription"
    assert created["session"]["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    assert created["session"]["audio"]["input"]["transcription"] == {"model": MUSE_MODEL, "language": "Japanese"}


def test_turnless_empty_silence_transcript_is_ignored():
    transformer = MuseEventTransformer()

    assert transformer.transform(json.loads(_event("transcript", transcript="", final=True))) == ()


def test_transcript_without_speech_start_synthesizes_start_before_delta():
    transformer = MuseEventTransformer()

    events = transformer.transform(json.loads(_event("transcript", turnId="turn-1", transcript="hello", final=False)))

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
    ]


def test_cumulative_partials_emit_only_extensions_and_final_is_authoritative():
    transformer = MuseEventTransformer()

    def send(payload: str) -> tuple[dict[str, object], ...]:
        return transformer.transform(json.loads(payload))

    started = send(_event("speechStart", turnId="turn-1"))
    first = send(_event("transcript", turnId="turn-1", transcript="hello", final=False))
    extension = send(_event("transcript", turnId="turn-1", transcript="hello world", final=False))
    rewrite = send(_event("transcript", turnId="turn-1", transcript="hullo world", final=False))
    completed = send(_event("speechComplete", turnId="turn-1", transcript="hullo world"))

    assert [event["type"] for event in started] == ["input_audio_buffer.speech_started"]
    assert first[0]["delta"] == "hello"
    assert extension[0]["delta"] == " world"
    assert rewrite == ()
    assert completed[0]["type"] == "input_audio_buffer.speech_stopped"
    assert completed[1]["type"] == "conversation.item.input_audio_transcription.completed"
    assert completed[1]["item_id"] == "turn-1"
    assert completed[1]["transcript"] == "hullo world"
    assert send(_event("speechEnd", turnId="turn-1")) == ()


def test_speech_end_then_speech_complete_emits_stopped_then_completed():
    transformer = MuseEventTransformer()

    transformer.transform(json.loads(_event("speechStart", turnId="turn-1")))
    stopped = transformer.transform(json.loads(_event("speechEnd", turnId="turn-1")))
    completed = transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript="done")))

    assert [event["type"] for event in stopped] == ["input_audio_buffer.speech_stopped"]
    assert [event["type"] for event in completed] == ["conversation.item.input_audio_transcription.completed"]
    assert completed[0]["transcript"] == "done"


def _typed(events: tuple[dict[str, object], ...]) -> list[tuple[object, object]]:
    return [(event["type"], event["item_id"]) for event in events]


def test_overlapping_turns_emit_independently_and_correlate_by_item_id():
    transformer = MuseEventTransformer()

    def send(payload: str) -> list[tuple[object, object]]:
        return _typed(transformer.transform(json.loads(payload)))

    assert send(_event("speechStart", turnId="turn-a")) == [("input_audio_buffer.speech_started", "turn-a")]
    assert send(_event("speechStart", turnId="turn-b")) == [("input_audio_buffer.speech_started", "turn-b")]
    assert send(_event("transcript", turnId="turn-b", transcript="second", final=False)) == [
        ("conversation.item.input_audio_transcription.delta", "turn-b")
    ]
    assert send(_event("speechComplete", turnId="turn-a", transcript="first")) == [
        ("input_audio_buffer.speech_stopped", "turn-a"),
        ("conversation.item.input_audio_transcription.completed", "turn-a"),
    ]
    assert send(_event("speechEnd", turnId="turn-a")) == []
    assert send(_event("speechEnd", turnId="turn-b")) == [("input_audio_buffer.speech_stopped", "turn-b")]
    assert send(_event("speechComplete", turnId="turn-b", transcript="second final")) == [
        ("conversation.item.input_audio_transcription.completed", "turn-b")
    ]


def test_empty_vad_turn_is_closed_and_does_not_block_the_next_turn():
    transformer = MuseEventTransformer()

    def send(payload: str) -> list[tuple[object, object]]:
        return _typed(transformer.transform(json.loads(payload)))

    assert send(_event("speechStart", turnId="noise")) == [("input_audio_buffer.speech_started", "noise")]
    assert send(_event("speechEnd", turnId="noise")) == [("input_audio_buffer.speech_stopped", "noise")]
    assert send(_event("speechStart", turnId="speech")) == [("input_audio_buffer.speech_started", "speech")]
    assert send(_event("transcript", turnId="speech", transcript="hello", final=False)) == [
        ("conversation.item.input_audio_transcription.delta", "speech")
    ]
    assert send(_event("speechEnd", turnId="speech")) == [("input_audio_buffer.speech_stopped", "speech")]
    assert send(_event("speechComplete", turnId="speech", transcript="hello world")) == [
        ("conversation.item.input_audio_transcription.completed", "speech")
    ]


@pytest.mark.parametrize("transcript", ["", "late words"])
def test_late_speech_complete_after_an_empty_speech_end_completes_that_item(transcript: str):
    transformer = MuseEventTransformer()
    transformer.transform(json.loads(_event("speechStart", turnId="turn-1")))
    transformer.transform(json.loads(_event("speechEnd", turnId="turn-1")))
    transformer.transform(json.loads(_event("speechStart", turnId="turn-2")))

    (completed,) = transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript=transcript)))

    assert completed["type"] == "conversation.item.input_audio_transcription.completed"
    assert completed["item_id"] == "turn-1"
    assert completed["transcript"] == transcript


def test_push_to_talk_speech_complete_closes_the_turn_without_speech_end():
    transformer = MuseEventTransformer()
    transformer.configure(MuseSessionConfig(MUSE_MODEL, "PUSH_TO_TALK", 24_000, ()))

    transformer.transform(json.loads(_event("speechStart", turnId="turn-1")))
    transformer.transform(json.loads(_event("transcript", turnId="turn-1", transcript="hel", final=False)))
    events = transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript="hello")))

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert events[1]["transcript"] == "hello"


_TERMINAL_SIGNALS: Final = {
    "speechEnd": _event("speechEnd", turnId="turn-1"),
    "speechComplete": _event("speechComplete", turnId="turn-1", transcript="final words"),
    "final": _event("transcript", turnId="turn-1", transcript="final words", final=True),
}
_TERMINAL_ORDERINGS: Final = tuple(
    ordering for size in (1, 2, 3) for ordering in itertools.permutations(_TERMINAL_SIGNALS, size)
)


@pytest.mark.parametrize("mode", ["ENDPOINTING", "PUSH_TO_TALK"])
@pytest.mark.parametrize("ordering", _TERMINAL_ORDERINGS, ids="-".join)
def test_every_terminal_signal_order_closes_the_turn_exactly_once(mode: MuseMode, ordering: tuple[str, ...]):
    transformer = MuseEventTransformer()
    transformer.configure(MuseSessionConfig(MUSE_MODEL, mode, 24_000, ()))
    transformer.transform(json.loads(_event("speechStart", turnId="turn-1")))
    transformer.transform(json.loads(_event("transcript", turnId="turn-1", transcript="fin", final=False)))

    emitted = [
        event["type"] for signal in ordering for event in transformer.transform(json.loads(_TERMINAL_SIGNALS[signal]))
    ]
    replayed = [
        event["type"] for signal in ordering for event in transformer.transform(json.loads(_TERMINAL_SIGNALS[signal]))
    ]

    has_text = bool(set(ordering) & {"speechComplete", "final"})
    assert emitted == [
        "input_audio_buffer.speech_stopped",
        *(["conversation.item.input_audio_transcription.completed"] if has_text else []),
    ]
    assert replayed == []


def test_push_to_talk_final_transcript_completes_without_speech_end():
    transformer = MuseEventTransformer()
    transformer.configure(MuseSessionConfig(MUSE_MODEL, "PUSH_TO_TALK", 24_000, ()))

    events = transformer.transform(json.loads(_event("transcript", transcript="hello there", final=True)))

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert events[2]["transcript"] == "hello there"
    assert str(events[0]["item_id"]).startswith("item_")


def test_positive_audio_progress_deltas_attach_to_next_completion_and_speaker_is_ignored():
    transformer = MuseEventTransformer()

    def send(payload: str) -> tuple[dict[str, object], ...]:
        return transformer.transform(json.loads(payload))

    send(_event("audioProgress", audioProcessedMs=1000))
    send(_event("audioProgress", audioProcessedMs=750))
    send(_event("audioProgress", audioProcessedMs=1600))
    assert send(_event("speaker", turnId=42, label=" Speaker 2 ")) == ()
    completed = send(_event("speechComplete", turnId=42, transcript="hello"))

    assert "speaker" not in completed[-1]
    assert completed[-1]["usage"] == {"type": "duration", "seconds": 1.6}
    assert transformer.take_unbilled_usage() is None
    assert send(_event("speechEnd", turnId=42)) == ()


def test_trailing_audio_progress_is_returned_once():
    transformer = MuseEventTransformer()

    transformer.transform(json.loads(_event("audioProgress", audioProcessedMs=250)))

    assert transformer.take_unbilled_usage() == {"type": "duration", "seconds": 0.25}
    assert transformer.take_unbilled_usage() is None


def test_finished_turn_ignores_late_duplicates():
    transformer = MuseEventTransformer()

    released = transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript="done")))

    assert [event["type"] for event in released] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript="duplicate"))) == ()
    assert transformer.transform(json.loads(_event("speechEnd", turnId="turn-1"))) == ()
    assert transformer.transform(json.loads(_event("speechStart", turnId="turn-1"))) == ()
    assert (
        transformer.transform(json.loads(_event("transcript", turnId="turn-1", transcript="late", final=False))) == ()
    )


def test_turn_memory_is_bounded_by_turn_limit():
    transformer = MuseEventTransformer(turn_limit=2)

    transformer.transform(json.loads(_event("speechComplete", turnId="turn-1", transcript="one")))
    transformer.transform(json.loads(_event("speechComplete", turnId="turn-2", transcript="two")))
    assert transformer.transform(json.loads(_event("speechEnd", turnId="turn-1"))) == ()
    transformer.transform(json.loads(_event("speechComplete", turnId="turn-3", transcript="three")))

    forgotten = transformer.transform(json.loads(_event("speechEnd", turnId="turn-1")))

    assert [event["type"] for event in forgotten] == ["input_audio_buffer.speech_stopped"]


def test_provider_error_is_sanitized_and_encodable():
    token = "private-token"
    provider_body = f"authorization failed for Bearer {token}"
    transformed = MuseEventTransformer().transform(
        json.loads(_event("error", code="AUTH", message=provider_body, request={"accessToken": token}))
    )

    encoded = json.dumps(transformed[0])
    assert json.loads(encoded)["error"] == {
        "type": "server_error",
        "message": "Meta Muse realtime transcription failed",
    }
    assert token not in encoded
    assert provider_body not in encoded


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("token", "Bearer token"), (" Bearer token ", "Bearer token"), ("bearer   token", "Bearer token")],
)
def test_access_token_normalization_adds_single_bearer_prefix(raw: str, expected: str):
    assert normalize_access_token(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "Bearer", " bearer "])
def test_access_token_normalization_rejects_empty_tokens(raw: str):
    with pytest.raises(ValueError, match=r"token|key is required"):
        normalize_access_token(raw)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, DEFAULT_MUSE_REALTIME_URL),
        ("https://example.test/custom/path?ignored=yes", "wss://example.test/v1/asr/realtime"),
        ("wss://example.test:8443/other", "wss://example.test:8443/v1/asr/realtime"),
    ],
)
def test_realtime_url_pins_muse_path(api_base: str | None, expected: str):
    assert build_muse_realtime_url(api_base) == expected
    assert MetaRealtimeConfig().get_complete_url(api_base, f"meta/{MUSE_MODEL}") == expected


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
def test_realtime_url_rejects_insecure_or_ambiguous_bases(api_base: str):
    with pytest.raises(ValueError, match="absolute wss:// or https://"):
        build_muse_realtime_url(api_base)


def test_unsupported_model_is_rejected_before_connecting():
    with pytest.raises(ValueError, match="Unsupported Meta realtime model: meta/other-model"):
        MetaRealtimeConfig().get_complete_url(None, "meta/other-model")


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("META_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key is required for Meta API calls"):
        MetaRealtimeConfig().validate_environment({}, MUSE_MODEL)


def test_bearer_token_travels_only_in_the_json_handshake():
    config = MetaRealtimeConfig()
    headers = {"x-existing": "kept"}

    assert config.validate_environment(headers, MUSE_MODEL, api_key="secret-token") == {"x-existing": "kept"}
    (handshake,) = config.transform_realtime_request(_ga_session_update(), MUSE_MODEL)

    assert isinstance(handshake, str)
    assert json.loads(handshake)["authorization"] == {"accessToken": "Bearer secret-token"}
    assert config.is_setup_message(json.loads(handshake)) is True
    assert config.is_setup_message({"type": "input_audio_buffer.append"}) is False
    assert config.transform_realtime_request(_ga_session_update(), MUSE_MODEL) == ()


def test_synthetic_session_created_uses_default_transcription_shape():
    created = MetaRealtimeConfig().transform_session_created_event(f"meta/{MUSE_MODEL}", "trace-1")

    assert created["type"] == "session.created"
    assert created["session"]["id"] == "trace-1"
    assert created["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert created["session"]["audio"]["input"]["transcription"] == {"model": MUSE_MODEL}


def test_audio_before_session_update_is_rejected():
    config = MetaRealtimeConfig()
    config.validate_environment({}, MUSE_MODEL, api_key="secret-token")

    with pytest.raises(MuseProtocolError, match=r"session\.update must configure"):
        config.transform_realtime_request(_event("input_audio_buffer.append", audio="AAAA"), MUSE_MODEL)


@pytest.mark.parametrize(("rate", "packet_bytes"), [(16_000, 2_560), (24_000, 3_840)])
def test_pcm_is_packetized_into_raw_binary_frames(rate: int, packet_bytes: int):
    config = _configured(rate=rate)
    pcm = b"\xff\xfe\x00\x80" * (packet_bytes // 2) + b"\x01\x02\x03\x04"

    frames = config.transform_realtime_request(
        _event("input_audio_buffer.append", audio=base64.b64encode(pcm).decode()), MUSE_MODEL
    )
    remainder = config.transform_realtime_request(_event("input_audio_buffer.commit"), MUSE_MODEL)

    assert frames == (pcm[:packet_bytes], pcm[packet_bytes : packet_bytes * 2])
    assert remainder == (pcm[packet_bytes * 2 :],)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        ("not base64!", "valid base64"),
        (base64.b64encode(b"\x00").decode(), "complete samples"),
        (12, "base64 string"),
        ("A" * (4 * ((24_000 * 2 * 4 + 2) // 3) + 4), "four-second backlog"),
    ],
)
def test_invalid_audio_appends_are_rejected(audio: object, message: str):
    config = _configured()

    with pytest.raises(MuseProtocolError, match=message):
        config.transform_realtime_request(_event("input_audio_buffer.append", audio=audio), MUSE_MODEL)


@pytest.mark.asyncio
async def test_backend_sends_are_paced_to_real_time():
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    config = _configured(monotonic=lambda: 10.0, sleep=record_sleep)
    packet = b"\x01\x02" * 1_920

    await config.pace_backend_send(packet)
    await config.pace_backend_send(packet)
    await config.pace_backend_send(packet)

    assert sleeps == pytest.approx([0.08, 0.16])


def test_endpointing_commit_flushes_without_end_stream_but_end_sends_it_once():
    config = _configured(turn_detection="server_vad")

    assert config.transform_realtime_request(_event("input_audio_buffer.commit"), MUSE_MODEL) == ()
    assert config.transform_realtime_request(_event("input_audio_buffer.end"), MUSE_MODEL) == ('{"type":"endStream"}',)
    assert config.transform_realtime_request(_event("input_audio_buffer.end"), MUSE_MODEL) == ()


def test_push_to_talk_commit_ends_the_stream_once():
    config = _configured(turn_detection=None)
    config.transform_realtime_request(
        _event("input_audio_buffer.append", audio=base64.b64encode(b"\x01\x02").decode()), MUSE_MODEL
    )

    assert config.transform_realtime_request(_event("input_audio_buffer.commit"), MUSE_MODEL) == (
        b"\x01\x02",
        '{"type":"endStream"}',
    )
    assert config.transform_realtime_request(_event("input_audio_buffer.end"), MUSE_MODEL) == ()


def test_clear_drops_buffered_remainder_and_unknown_events_are_ignored():
    config = _configured()
    config.transform_realtime_request(
        _event("input_audio_buffer.append", audio=base64.b64encode(b"\x01\x02").decode()), MUSE_MODEL
    )

    assert config.transform_realtime_request(_event("input_audio_buffer.clear"), MUSE_MODEL) == ()
    assert config.transform_realtime_request(_event("response.create"), MUSE_MODEL) == ()
    assert config.transform_realtime_request(_event("input_audio_buffer.commit"), MUSE_MODEL) == ()


def test_provider_ack_becomes_session_created_with_provider_id():
    config = _configured(rate=16_000, turn_detection=None)

    (created,) = _backend_events(config, json.dumps({"sessionId": " provider-session "}))

    assert created["type"] == "session.created"
    assert created["session"]["id"] == "provider-session"
    assert created["session"]["audio"]["input"]["format"]["rate"] == 16000
    assert created["session"]["audio"]["input"]["turn_detection"] is None


def test_provider_turn_events_and_close_usage_flow_through_config():
    config = _configured()

    assert _backend_events(config, json.dumps({"type": "audioProgress", "audioProcessedMs": 1349})) == []
    assert _backend_events(config, _event("speechStart", turnId="t1"))[0]["type"] == "input_audio_buffer.speech_started"
    assert _backend_events(config, _event("speechEnd", turnId="t1"))[0]["type"] == "input_audio_buffer.speech_stopped"
    completed = _backend_events(config, _event("speechComplete", turnId="t1", transcript="what is the weather"))

    assert [event["type"] for event in completed] == ["conversation.item.input_audio_transcription.completed"]
    assert completed[0]["usage"] == {"type": "duration", "seconds": 1.349}
    assert config.unbilled_usage_on_session_close(MUSE_MODEL) is None

    assert _backend_events(config, json.dumps({"type": "audioProgress", "audioProcessedMs": 2349})) == []
    assert config.unbilled_usage_on_session_close(MUSE_MODEL) == {"type": "duration", "seconds": 1.0}


def test_provider_error_frame_becomes_openai_error_without_leaking_token():
    config = _configured()

    (error,) = _backend_events(config, _event("error", message="bad token secret-token"))

    assert error == {
        "type": "error",
        "error": {"type": "server_error", "message": "Meta Muse realtime transcription failed"},
    }
    assert "secret-token" not in json.dumps(error)


def test_invalid_provider_ack_is_rejected():
    config = _configured()

    with pytest.raises(MuseProtocolError, match="invalid handshake response"):
        _backend_events(config, json.dumps({"sessionId": ""}))

import base64
import json

import pytest

from litellm.llms.base_llm.realtime.transcription_protocol import (
    RealtimeTranscriptionProtocolError,
    completed_event,
    decode_pcm16_append,
    parse_transcription_session_update,
    transcription_session,
)


def _session_update(session: dict[str, object]) -> str:
    return json.dumps({"type": "session.update", "session": session})


def test_ga_layout_parses_format_language_and_turn_detection():
    update = parse_transcription_session_update(
        _session_update(
            {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16_000, "channels": 1},
                        "transcription": {"model": "chirp_3", "language": "pt-BR", "prompt": "names"},
                        "turn_detection": {"type": "server_vad", "threshold": 0.5},
                    }
                },
            }
        )
    )
    assert update.session_type == "transcription"
    assert update.audio_format is not None
    assert (update.audio_format.layout, update.audio_format.rate, update.audio_format.channels) == ("ga", 16_000, 1)
    assert update.audio_format.is_pcm16
    assert (update.model, update.language) == ("chirp_3", "pt-BR")
    assert update.unsupported_transcription_keys == ("prompt",)
    assert update.turn_detection_type == "server_vad"
    assert not update.turn_detection_disabled


def test_beta_layout_parses_flat_fields():
    update = parse_transcription_session_update(
        json.dumps(
            {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": None,
                },
            }
        )
    )
    assert update.audio_format is not None
    assert (update.audio_format.layout, update.audio_format.encoding) == ("beta", "pcm16")
    assert update.audio_format.is_pcm16
    assert update.model == "whisper-1"
    assert update.turn_detection_disabled


def test_absent_turn_detection_is_not_disabled():
    update = parse_transcription_session_update(_session_update({"audio": {"input": {"transcription": {}}}}))
    assert update.turn_detection is None
    assert not update.turn_detection_disabled


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "invalid JSON object"),
        ("[]", "must be a JSON object"),
        (json.dumps({"type": "response.create"}), "expected session.update"),
        (_session_update({}), "requires a session object"),
        (_session_update({"input_audio_format": "pcm16", "audio": {"input": {"format": "pcm16"}}}), "either beta or GA"),
        (_session_update({"input_audio_transcription": {}, "audio": {"input": {"transcription": {}}}}), "either beta or GA"),
        (_session_update({"audio": {"input": {"format": {"rate": "fast"}}}}), "must be an integer"),
        (_session_update({"audio": {"input": {"format": {"rate": True}}}}), "must be an integer"),
        (_session_update({"audio": {"input": {"transcription": {"language": 7}}}}), "must be a string"),
        (_session_update({"audio": {"input": {"transcription": []}}}), "must be an object"),
    ],
)
def test_malformed_session_updates_are_rejected(payload: str, message: str):
    with pytest.raises(RealtimeTranscriptionProtocolError, match=message):
        parse_transcription_session_update(payload)


def test_decode_pcm16_append_returns_the_raw_samples():
    assert decode_pcm16_append(base64.b64encode(b"\x01\x02\x03\x04").decode()) == b"\x01\x02\x03\x04"


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (None, "must be a base64 string"),
        ("@@@", "must be valid base64"),
        (base64.b64encode(b"\x01\x02\x03").decode(), "complete samples"),
    ],
)
def test_decode_pcm16_append_rejects_bad_audio(audio: object, message: str):
    with pytest.raises(RealtimeTranscriptionProtocolError, match=message):
        decode_pcm16_append(audio)


def test_decode_pcm16_append_enforces_the_backlog_limit():
    with pytest.raises(RealtimeTranscriptionProtocolError, match="backlog limit"):
        decode_pcm16_append(base64.b64encode(b"\x00" * 8).decode(), max_encoded_bytes=4)


def test_transcription_session_reflects_negotiated_settings():
    manual = transcription_session(session_id="sess_1", model="chirp_3", sample_rate=16_000, language=None, server_vad=False)
    assert manual["id"] == "sess_1"
    assert manual["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 16_000},
        "transcription": {"model": "chirp_3"},
        "turn_detection": None,
    }
    vad = transcription_session(session_id="sess_1", model="chirp_3", sample_rate=24_000, language="en-US", server_vad=True)
    assert vad["audio"]["input"]["transcription"] == {"model": "chirp_3", "language": "en-US"}
    assert vad["audio"]["input"]["turn_detection"] == {"type": "server_vad"}


def test_completed_event_carries_usage_only_when_billed():
    assert "usage" not in completed_event("item_1", "hello", None)
    billed = completed_event("item_1", "hello", {"type": "duration", "seconds": 2.5})
    assert (billed["item_id"], billed["transcript"], billed["usage"]) == ("item_1", "hello", {"type": "duration", "seconds": 2.5})

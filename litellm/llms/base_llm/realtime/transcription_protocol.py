import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._uuid import uuid
from litellm.types.llms.openai import (
    OpenAIRealtimeErrorEvent,
    OpenAIRealtimeInputAudioBufferSpeechEvent,
    OpenAIRealtimeInputAudioTranscriptionCompleted,
    OpenAIRealtimeInputAudioTranscriptionDelta,
    OpenAIRealtimeServerVadTurnDetection,
    OpenAIRealtimeTranscriptionSession,
    OpenAIRealtimeTranscriptionSessionCreated,
    OpenAIRealtimeTranscriptionSessionUpdated,
    OpenAIRealtimeTranscriptionSettings,
)
from litellm.types.realtime import RealtimeInputAudioTranscriptionDurationUsage, RealtimeInputAudioTranscriptionUsage

SESSION_UPDATE_EVENT_TYPES: Final = frozenset(("session.update", "transcription_session.update"))
PCM16_ENCODINGS: Final = frozenset(("pcm16", "audio/pcm"))
SERVER_VAD_TURN_DETECTION: Final[OpenAIRealtimeServerVadTurnDetection] = {"type": "server_vad"}
EMPTY_JSON_OBJECT: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_SUPPORTED_TRANSCRIPTION_KEYS: Final = frozenset(("model", "language"))
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class RealtimeTranscriptionProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TranscriptionAudioFormat:
    layout: Literal["beta", "ga"]
    encoding: str | None
    rate: int | None
    channels: int | None

    @property
    def is_pcm16(self) -> bool:
        return self.encoding in PCM16_ENCODINGS


@dataclass(frozen=True, slots=True)
class TranscriptionSessionUpdate:
    session_type: str | None
    audio_format: TranscriptionAudioFormat | None
    model: str | None
    language: str | None
    unsupported_transcription_keys: tuple[str, ...]
    turn_detection: Mapping[str, JsonValue] | None
    turn_detection_disabled: bool

    @property
    def turn_detection_type(self) -> JsonValue | None:
        return None if self.turn_detection is None else self.turn_detection.get("type")


ProtocolErrorType = type[RealtimeTranscriptionProtocolError]


def json_object(payload: str, error: ProtocolErrorType = RealtimeTranscriptionProtocolError) -> Mapping[str, JsonValue]:
    try:
        value: Final = _JSON_ADAPTER.validate_json(payload)
    except ValidationError:
        raise error("invalid JSON object") from None
    if not isinstance(value, dict):
        raise error("message must be a JSON object")
    return value


def json_mapping(
    value: JsonValue | None, name: str, error: ProtocolErrorType = RealtimeTranscriptionProtocolError
) -> Mapping[str, JsonValue]:
    if value is None:
        return EMPTY_JSON_OBJECT
    if not isinstance(value, dict):
        raise error(f"{name} must be an object")
    return value


def json_string(
    value: JsonValue | None, name: str, error: ProtocolErrorType = RealtimeTranscriptionProtocolError
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise error(f"{name} must be a string")
    return value


def json_integer(
    value: JsonValue | None, name: str, error: ProtocolErrorType = RealtimeTranscriptionProtocolError
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{name} must be an integer")
    return value


def new_event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def parse_transcription_session_update(
    payload: str, error: ProtocolErrorType = RealtimeTranscriptionProtocolError
) -> TranscriptionSessionUpdate:
    message: Final = json_object(payload, error)
    if message.get("type") not in SESSION_UPDATE_EVENT_TYPES:
        raise error("expected session.update")
    session: Final = json_mapping(message.get("session"), "session", error)
    if not session:
        raise error("session.update requires a session object")
    audio: Final = json_mapping(session.get("audio"), "session.audio", error)
    audio_input: Final = json_mapping(audio.get("input"), "session.audio.input", error)
    beta_transcription: Final = session.get("input_audio_transcription")
    ga_transcription: Final = audio_input.get("transcription")
    if beta_transcription is not None and ga_transcription is not None:
        raise error("input transcription must use either beta or GA layout")
    transcription: Final = json_mapping(
        beta_transcription if beta_transcription is not None else ga_transcription,
        "input audio transcription",
        error,
    )
    turn_detection_present: Final = "turn_detection" in session or "turn_detection" in audio_input
    turn_detection: Final = session.get("turn_detection", audio_input.get("turn_detection"))
    return TranscriptionSessionUpdate(
        session_type=json_string(session.get("type"), "session.type", error),
        audio_format=_parse_audio_format(session, audio_input, error),
        model=json_string(transcription.get("model"), "transcription model", error),
        language=json_string(transcription.get("language"), "language", error),
        unsupported_transcription_keys=tuple(
            sorted(key for key in transcription if key not in _SUPPORTED_TRANSCRIPTION_KEYS)
        ),
        turn_detection=None if turn_detection is None else json_mapping(turn_detection, "turn_detection", error),
        turn_detection_disabled=turn_detection_present and turn_detection is None,
    )


def _parse_audio_format(
    session: Mapping[str, JsonValue], audio_input: Mapping[str, JsonValue], error: ProtocolErrorType
) -> TranscriptionAudioFormat | None:
    beta_format: Final = session.get("input_audio_format")
    ga_format: Final = audio_input.get("format")
    if beta_format is not None and ga_format is not None:
        raise error("input audio format must use either beta or GA layout")
    if beta_format is not None:
        return TranscriptionAudioFormat(
            layout="beta",
            encoding=json_string(beta_format, "session.input_audio_format", error),
            rate=None,
            channels=None,
        )
    if ga_format is None:
        return None
    if isinstance(ga_format, str):
        return TranscriptionAudioFormat(layout="ga", encoding=ga_format, rate=None, channels=None)
    format_mapping: Final = json_mapping(ga_format, "session.audio.input.format", error)
    return TranscriptionAudioFormat(
        layout="ga",
        encoding=json_string(format_mapping.get("type"), "session.audio.input.format.type", error),
        rate=json_integer(format_mapping.get("rate"), "session.audio.input.format.rate", error),
        channels=json_integer(format_mapping.get("channels"), "session.audio.input.format.channels", error),
    )


def decode_pcm16_append(
    audio: JsonValue | None,
    max_encoded_bytes: int | None = None,
    error: ProtocolErrorType = RealtimeTranscriptionProtocolError,
) -> bytes:
    if not isinstance(audio, str):
        raise error("Audio must be a base64 string")
    if max_encoded_bytes is not None and len(audio) > max_encoded_bytes:
        raise error("Audio append exceeds the four-second backlog limit")
    try:
        decoded: Final = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError):
        raise error("Audio must be valid base64") from None
    if len(decoded) % 2:
        raise error("PCM16 audio must contain complete samples")
    return decoded


def _transcription_settings(model: str, language: str | None) -> OpenAIRealtimeTranscriptionSettings:
    if language is None:
        model_only: Final[OpenAIRealtimeTranscriptionSettings] = {"model": model}
        return model_only
    with_language: Final[OpenAIRealtimeTranscriptionSettings] = {"model": model, "language": language}
    return with_language


def transcription_session(
    *, session_id: str, model: str, sample_rate: int, language: str | None, server_vad: bool
) -> OpenAIRealtimeTranscriptionSession:
    settings: Final = _transcription_settings(model, language)
    session: Final[OpenAIRealtimeTranscriptionSession] = {
        "id": session_id,
        "object": "realtime.transcription_session",
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": sample_rate},
                "transcription": settings,
                "turn_detection": SERVER_VAD_TURN_DETECTION if server_vad else None,
            }
        },
    }
    return session


def transcription_session_created_event(
    session: OpenAIRealtimeTranscriptionSession,
) -> OpenAIRealtimeTranscriptionSessionCreated:
    event: Final[OpenAIRealtimeTranscriptionSessionCreated] = {
        "type": "session.created",
        "event_id": new_event_id(),
        "session": session,
    }
    return event


def transcription_session_updated_event(
    session: OpenAIRealtimeTranscriptionSession,
) -> OpenAIRealtimeTranscriptionSessionUpdated:
    event: Final[OpenAIRealtimeTranscriptionSessionUpdated] = {
        "type": "session.updated",
        "event_id": new_event_id(),
        "session": session,
    }
    return event


def error_event(message: str) -> OpenAIRealtimeErrorEvent:
    event: Final[OpenAIRealtimeErrorEvent] = {
        "type": "error",
        "error": {"type": "server_error", "message": message},
    }
    return event


def speech_event(
    event_type: Literal["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"], item_id: str
) -> OpenAIRealtimeInputAudioBufferSpeechEvent:
    event: Final[OpenAIRealtimeInputAudioBufferSpeechEvent] = {
        "type": event_type,
        "event_id": new_event_id(),
        "item_id": item_id,
    }
    return event


def delta_event(item_id: str, delta: str) -> OpenAIRealtimeInputAudioTranscriptionDelta:
    event: Final[OpenAIRealtimeInputAudioTranscriptionDelta] = {
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": new_event_id(),
        "item_id": item_id,
        "content_index": 0,
        "delta": delta,
    }
    return event


def completed_event(
    item_id: str, transcript: str, usage: RealtimeInputAudioTranscriptionUsage | None
) -> OpenAIRealtimeInputAudioTranscriptionCompleted:
    event: Final[OpenAIRealtimeInputAudioTranscriptionCompleted] = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": new_event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }
    if usage is None:
        return event
    billed: Final[OpenAIRealtimeInputAudioTranscriptionCompleted] = {**event, "usage": usage}
    return billed


def duration_usage(seconds: float) -> RealtimeInputAudioTranscriptionUsage:
    usage: Final[RealtimeInputAudioTranscriptionDurationUsage] = {"type": "duration", "seconds": seconds}
    return usage

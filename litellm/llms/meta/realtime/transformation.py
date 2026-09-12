import asyncio
import base64
import binascii
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.meta import MuseAudioEncoding, MuseHandshake, MuseMode, MuseSampleRate
from litellm.types.llms.openai import (
    OpenAIRealtimeErrorEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeInputAudioBufferSpeechEvent,
    OpenAIRealtimeInputAudioTranscriptionCompleted,
    OpenAIRealtimeInputAudioTranscriptionDelta,
    OpenAIRealtimeServerVadTurnDetection,
    OpenAIRealtimeTranscriptionSession,
    OpenAIRealtimeTranscriptionSessionCreated,
    OpenAIRealtimeTranscriptionSettings,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionDurationUsage,
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

MUSE_MODEL: Final = "muse-voice-transcribe-1.0"
DEFAULT_MUSE_REALTIME_URL: Final = "wss://api.meta.ai/v1/asr/realtime"
SUPPORTED_SAMPLE_RATES: Final = frozenset((16_000, 24_000))
SUPPORTED_LANGUAGES: Final = (
    "Arabic",
    "Bengali",
    "Dutch",
    "English",
    "French",
    "German",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Malay",
    "Mandarin Chinese",
    "Marathi",
    "Polish",
    "Portuguese",
    "Spanish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Vietnamese",
)
_LANGUAGE_NAMES: Final = MappingProxyType({language.casefold(): language for language in SUPPORTED_LANGUAGES})
_LANGUAGE_CODES: Final = MappingProxyType(
    {
        "ar": "Arabic",
        "bn": "Bengali",
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fil": "Tagalog",
        "fr": "French",
        "he": "Hebrew",
        "hi": "Hindi",
        "id": "Indonesian",
        "it": "Italian",
        "iw": "Hebrew",
        "ja": "Japanese",
        "kn": "Kannada",
        "ko": "Korean",
        "ms": "Malay",
        "mr": "Marathi",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "zh": "Mandarin Chinese",
    }
)
_SUPPORTED_TRANSCRIPTION_KEYS: Final = frozenset(("model", "language"))
_MAX_AUDIO_BACKLOG_SECONDS: Final = 4
_PACKET_MS: Final = 80
_END_STREAM: Final = '{"type":"endStream"}'
_PROVIDER_ERROR_MESSAGE: Final = "Meta Muse realtime transcription failed"
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_EMPTY_OBJECT: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_SERVER_VAD: Final[OpenAIRealtimeServerVadTurnDetection] = {"type": "server_vad"}


class MuseProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MuseSessionConfig:
    model: str
    mode: MuseMode
    sample_rate: MuseSampleRate
    language_bias: tuple[str, ...]

    @property
    def audio_encoding(self) -> MuseAudioEncoding:
        return "PCM_16KHZ" if self.sample_rate == 16_000 else "PCM_24KHZ"

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * 2

    @property
    def packet_bytes(self) -> int:
        return self.bytes_per_second * _PACKET_MS // 1000

    @property
    def max_encoded_append_bytes(self) -> int:
        return 4 * ((self.bytes_per_second * _MAX_AUDIO_BACKLOG_SECONDS + 2) // 3)

    def handshake(self, access_token: str) -> MuseHandshake:
        base: Final[MuseHandshake] = {
            "authorization": {"accessToken": access_token},
            "audioEncoding": self.audio_encoding,
            "model": self.model,
            "mode": self.mode,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,
        }
        if not self.language_bias:
            return base
        biased: Final[MuseHandshake] = {**base, "languageBias": self.language_bias}
        return biased

    def openai_session(self, session_id: str) -> OpenAIRealtimeTranscriptionSession:
        session: Final[OpenAIRealtimeTranscriptionSession] = {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.sample_rate},
                    "transcription": self._transcription_settings(),
                    "turn_detection": None if self.mode == "PUSH_TO_TALK" else _SERVER_VAD,
                }
            },
        }
        return session

    def _transcription_settings(self) -> OpenAIRealtimeTranscriptionSettings:
        base: Final[OpenAIRealtimeTranscriptionSettings] = {"model": self.model}
        if not self.language_bias:
            return base
        localized: Final[OpenAIRealtimeTranscriptionSettings] = {**base, "language": self.language_bias[0]}
        return localized


_DEFAULT_SESSION_CONFIG: Final = MuseSessionConfig(
    model=MUSE_MODEL, mode="ENDPOINTING", sample_rate=24_000, language_bias=()
)


def _json_object(payload: str) -> Mapping[str, JsonValue]:
    try:
        value: Final = _JSON_ADAPTER.validate_json(payload)
    except ValidationError:
        raise MuseProtocolError("invalid JSON object") from None
    if not isinstance(value, dict):
        raise MuseProtocolError("message must be a JSON object")
    return value


def _mapping(value: JsonValue | None, name: str) -> Mapping[str, JsonValue]:
    if value is None:
        return _EMPTY_OBJECT
    if not isinstance(value, dict):
        raise MuseProtocolError(f"{name} must be an object")
    return value


def _string(value: JsonValue | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MuseProtocolError(f"{name} must be a string")
    return value


def _normalize_model(model: str) -> str:
    return model.removeprefix("meta/").strip()


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def normalize_language(language: str) -> str:
    value: Final = language.strip()
    if not value:
        raise MuseProtocolError("language must be non-empty")
    documented_name: Final = _LANGUAGE_NAMES.get(value.casefold())
    if documented_name is not None:
        return documented_name
    primary: Final = value.replace("_", "-").split("-", 1)[0].casefold()
    mapped_name: Final = _LANGUAGE_CODES.get(primary)
    if mapped_name is None:
        raise MuseProtocolError("unsupported Muse Voice language")
    return mapped_name


def normalize_access_token(api_key: str) -> str:
    stripped: Final = api_key.strip()
    if not stripped:
        raise ValueError("Meta API key is required")
    parts: Final = stripped.split(None, 1)
    if parts[0].casefold() != "bearer":
        return f"Bearer {stripped}"
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError("Meta API key must include a token after Bearer")
    return f"Bearer {parts[1].strip()}"


def build_muse_realtime_url(api_base: str | None) -> str:
    if api_base is None:
        return DEFAULT_MUSE_REALTIME_URL
    parsed: Final = urlparse(api_base.strip())
    scheme: Final = "wss" if parsed.scheme == "https" else parsed.scheme
    if (
        scheme != "wss"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("Meta api_base must be an absolute wss:// or https:// URL without credentials or a fragment")
    netloc: Final = f"{parsed.hostname}:{parsed.port}" if parsed.port is not None else parsed.hostname
    return urlunparse((scheme, netloc, "/v1/asr/realtime", "", "", ""))


def _parse_sample_rate(session: Mapping[str, JsonValue]) -> MuseSampleRate:
    beta_format: Final = session.get("input_audio_format")
    audio: Final = _mapping(session.get("audio"), "session.audio")
    audio_input: Final = _mapping(audio.get("input"), "session.audio.input")
    ga_format: Final = audio_input.get("format")
    if beta_format is not None and ga_format is not None:
        raise MuseProtocolError("input audio format must use either beta or GA layout")
    if beta_format is not None:
        if beta_format != "pcm16":
            raise MuseProtocolError("Muse Voice requires pcm16 input audio")
        return 24_000
    if ga_format is None:
        return 24_000
    if isinstance(ga_format, str):
        if ga_format != "pcm16":
            raise MuseProtocolError("Muse Voice requires audio/pcm input audio")
        return 24_000
    format_mapping: Final = _mapping(ga_format, "session.audio.input.format")
    if format_mapping.get("type") != "audio/pcm":
        raise MuseProtocolError("Muse Voice requires audio/pcm input audio")
    channels: Final = format_mapping.get("channels", 1)
    if isinstance(channels, bool) or channels != 1:
        raise MuseProtocolError("Muse Voice requires mono input audio")
    rate: Final = format_mapping.get("rate", 24_000)
    if isinstance(rate, bool) or not isinstance(rate, int) or rate not in SUPPORTED_SAMPLE_RATES:
        raise MuseProtocolError("Muse Voice supports PCM16 at 16000 Hz or 24000 Hz")
    return 16_000 if rate == 16_000 else 24_000


def _parse_mode(session: Mapping[str, JsonValue], audio_input: Mapping[str, JsonValue]) -> MuseMode:
    turn_detection_present: Final = "turn_detection" in session or "turn_detection" in audio_input
    turn_detection: Final = session.get("turn_detection", audio_input.get("turn_detection"))
    if turn_detection_present and turn_detection is None:
        return "PUSH_TO_TALK"
    if turn_detection is None:
        return "ENDPOINTING"
    turn_detection_mapping: Final = _mapping(turn_detection, "turn_detection")
    if turn_detection_mapping.get("type") not in (None, "server_vad"):
        raise MuseProtocolError("Muse Voice supports server_vad turn detection or null")
    return "ENDPOINTING"


def parse_session_update(payload: str, expected_model: str) -> MuseSessionConfig:
    message: Final = _json_object(payload)
    if message.get("type") not in ("session.update", "transcription_session.update"):
        raise MuseProtocolError("expected session.update")
    session: Final = _mapping(message.get("session"), "session")
    if not session:
        raise MuseProtocolError("session.update requires a session object")
    if session.get("type") not in (None, "transcription", "realtime"):
        raise MuseProtocolError("Muse Voice supports transcription sessions only")
    audio: Final = _mapping(session.get("audio"), "session.audio")
    audio_input: Final = _mapping(audio.get("input"), "session.audio.input")
    beta_transcription: Final = session.get("input_audio_transcription")
    ga_transcription: Final = audio_input.get("transcription")
    if beta_transcription is not None and ga_transcription is not None:
        raise MuseProtocolError("input transcription must use either beta or GA layout")
    transcription: Final = _mapping(
        beta_transcription if beta_transcription is not None else ga_transcription,
        "input audio transcription",
    )
    unsupported: Final = tuple(sorted(key for key in transcription if key not in _SUPPORTED_TRANSCRIPTION_KEYS))
    if unsupported:
        verbose_logger.warning("Meta realtime: dropping unsupported transcription settings %s", unsupported)
    requested_model: Final = _string(transcription.get("model"), "transcription model")
    normalized_model: Final = _normalize_model(expected_model)
    if normalized_model != MUSE_MODEL:
        raise MuseProtocolError("unsupported Meta realtime model")
    if requested_model is not None and _normalize_model(requested_model) != normalized_model:
        raise MuseProtocolError("realtime session model cannot be changed")
    language: Final = _string(transcription.get("language"), "language")
    return MuseSessionConfig(
        model=normalized_model,
        mode=_parse_mode(session, audio_input),
        sample_rate=_parse_sample_rate(session),
        language_bias=() if language is None else (normalize_language(language),),
    )


def session_created_event(config: MuseSessionConfig, session_id: str) -> OpenAIRealtimeTranscriptionSessionCreated:
    event: Final[OpenAIRealtimeTranscriptionSessionCreated] = {
        "type": "session.created",
        "event_id": _event_id(),
        "session": config.openai_session(session_id),
    }
    return event


def error_event(message: str) -> OpenAIRealtimeErrorEvent:
    event: Final[OpenAIRealtimeErrorEvent] = {
        "type": "error",
        "error": {"type": "server_error", "message": message},
    }
    return event


def _speech_event(
    event_type: Literal["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"], item_id: str
) -> OpenAIRealtimeInputAudioBufferSpeechEvent:
    event: Final[OpenAIRealtimeInputAudioBufferSpeechEvent] = {
        "type": event_type,
        "event_id": _event_id(),
        "item_id": item_id,
    }
    return event


def _delta_event(item_id: str, delta: str) -> OpenAIRealtimeInputAudioTranscriptionDelta:
    event: Final[OpenAIRealtimeInputAudioTranscriptionDelta] = {
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "delta": delta,
    }
    return event


def _completed_event(
    item_id: str, transcript: str, usage: RealtimeInputAudioTranscriptionUsage | None
) -> OpenAIRealtimeInputAudioTranscriptionCompleted:
    event: Final[OpenAIRealtimeInputAudioTranscriptionCompleted] = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }
    if usage is None:
        return event
    billed: Final[OpenAIRealtimeInputAudioTranscriptionCompleted] = {**event, "usage": usage}
    return billed


def _required_turn_id(message: Mapping[str, JsonValue], event: str) -> str:
    value: Final = message.get("turnId")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise MuseProtocolError(f"{event} event has invalid turnId")
    turn_id: Final = str(value).strip()
    if not turn_id:
        raise MuseProtocolError(f"{event} event has invalid turnId")
    return turn_id


def _new_suffix(previous: str, current: str) -> str:
    return current[len(previous) :] if current.startswith(previous) else ""


@dataclass(slots=True)
class _TurnState:
    item_id: str
    started: bool = False
    start_emitted: bool = False
    latest_partial: str | None = None
    emitted_partial: str = ""
    final_text: str | None = None
    completed_emitted: bool = False
    stopped: bool = False
    stopped_emitted: bool = False

    def finish(self, transcript: str) -> None:
        self.final_text = transcript
        self.stopped = True

    def drain(
        self, take_usage: Callable[[], RealtimeInputAudioTranscriptionUsage | None]
    ) -> Iterator[OpenAIRealtimeEvents]:
        has_content: Final = self.latest_partial is not None or self.final_text is not None
        if (self.started or has_content) and not self.start_emitted:
            self.start_emitted = True
            yield _speech_event("input_audio_buffer.speech_started", self.item_id)
        if self.latest_partial is not None and self.final_text is None:
            delta: Final = _new_suffix(self.emitted_partial, self.latest_partial)
            if delta:
                self.emitted_partial = self.latest_partial
                yield _delta_event(self.item_id, delta)
        if self.stopped and not self.stopped_emitted:
            self.stopped_emitted = True
            yield _speech_event("input_audio_buffer.speech_stopped", self.item_id)
        if self.final_text is not None and self.stopped_emitted and not self.completed_emitted:
            self.completed_emitted = True
            yield _completed_event(self.item_id, self.final_text, take_usage())


class MuseEventTransformer:
    def __init__(self, *, turn_limit: int = 128) -> None:
        self._turns: dict[str, _TurnState] = {}  # mutable-ok: bounded, insertion-ordered per-turn emit state
        self._turn_limit: Final = turn_limit
        self._active_turn_id: str | None = None
        self._mode: MuseMode = "ENDPOINTING"
        self._last_audio_processed_ms: float = 0.0
        self._unbilled_seconds: float = 0.0

    def configure(self, config: MuseSessionConfig) -> None:
        self._mode = config.mode

    def transform(self, message: Mapping[str, JsonValue]) -> tuple[OpenAIRealtimeEvents, ...]:
        event_type: Final = message.get("type")
        if event_type == "error":
            return (error_event(_PROVIDER_ERROR_MESSAGE),)
        if event_type == "audioProgress":
            self._update_audio_progress(message)
            return ()
        turn: Final = self._apply_turn_event(event_type, message)
        if turn is None:
            return ()
        return tuple(turn.drain(self.take_unbilled_usage))

    def take_unbilled_usage(self) -> RealtimeInputAudioTranscriptionUsage | None:
        seconds: Final = self._unbilled_seconds
        if seconds <= 0:
            return None
        self._unbilled_seconds = 0.0
        usage: Final[RealtimeInputAudioTranscriptionDurationUsage] = {"type": "duration", "seconds": seconds}
        return usage

    def _apply_turn_event(self, event_type: JsonValue | None, message: Mapping[str, JsonValue]) -> _TurnState | None:
        match event_type:
            case "speechStart":
                return self._speech_start(message)
            case "transcript":
                return self._transcript(message)
            case "speechEnd":
                return self._speech_end(message)
            case "speechComplete":
                return self._speech_complete(message)
            case _:
                return None

    def _turn(self, turn_id: str) -> _TurnState:
        existing: Final = self._turns.get(turn_id)
        if existing is not None:
            return existing
        created: Final = _TurnState(item_id=turn_id)
        self._turns[turn_id] = created
        if len(self._turns) > self._turn_limit:
            del self._turns[next(iter(self._turns))]
        return created

    def _speech_start(self, message: Mapping[str, JsonValue]) -> _TurnState:
        turn: Final = self._turn(_required_turn_id(message, "speechStart"))
        if turn.stopped:
            return turn
        turn.started = True
        self._active_turn_id = turn.item_id
        return turn

    def _transcript(self, message: Mapping[str, JsonValue]) -> _TurnState | None:
        transcript: Final = message.get("transcript")
        if not isinstance(transcript, str):
            raise MuseProtocolError("transcript event has invalid transcript")
        if not transcript and message.get("turnId") is None and self._active_turn_id is None:
            return None
        turn: Final = self._turn(self._transcript_turn_id(message))
        if message.get("final") is True:
            self._finish(turn, transcript)
        elif turn.final_text is None:
            turn.latest_partial = transcript
        return turn

    def _speech_end(self, message: Mapping[str, JsonValue]) -> _TurnState:
        turn: Final = self._turn(_required_turn_id(message, "speechEnd"))
        turn.stopped = True
        self._release_active(turn)
        return turn

    def _speech_complete(self, message: Mapping[str, JsonValue]) -> _TurnState:
        transcript: Final = message.get("transcript")
        if not isinstance(transcript, str):
            raise MuseProtocolError("speechComplete event has invalid transcript")
        turn: Final = self._turn(_required_turn_id(message, "speechComplete"))
        self._finish(turn, transcript)
        return turn

    def _finish(self, turn: _TurnState, transcript: str) -> None:
        turn.finish(transcript)
        self._release_active(turn)

    def _release_active(self, turn: _TurnState) -> None:
        if self._active_turn_id == turn.item_id:
            self._active_turn_id = None

    def _update_audio_progress(self, message: Mapping[str, JsonValue]) -> None:
        processed_ms: Final = message.get("audioProcessedMs")
        if (
            isinstance(processed_ms, bool)
            or not isinstance(processed_ms, (int, float))
            or not math.isfinite(processed_ms)
            or processed_ms < 0
        ):
            raise MuseProtocolError("audioProgress event has invalid audioProcessedMs")
        if processed_ms <= self._last_audio_processed_ms:
            return
        self._unbilled_seconds += (float(processed_ms) - self._last_audio_processed_ms) / 1000
        self._last_audio_processed_ms = float(processed_ms)

    def _transcript_turn_id(self, message: Mapping[str, JsonValue]) -> str:
        if message.get("turnId") is not None:
            return _required_turn_id(message, "transcript")
        if self._active_turn_id is not None:
            return self._active_turn_id
        if self._mode != "PUSH_TO_TALK":
            raise MuseProtocolError("transcript event is missing turnId outside an active turn")
        turn_id: Final = f"item_{uuid.uuid4().hex}"
        self._active_turn_id = turn_id
        return turn_id


class MetaRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._monotonic: Final = monotonic
        self._sleep: Final = sleep
        self._transformer: Final = MuseEventTransformer()
        self._access_token: str | None = None
        self._config: MuseSessionConfig | None = None
        self._pending_audio: bytes = b""
        self._end_stream_sent: bool = False
        self._pacing_origin: float | None = None
        self._sent_duration: float = 0.0

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseRealtimeConfig contract
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig contract
        token: Final = api_key or get_secret_str("META_API_KEY")
        if token is None:
            raise ValueError("api_key is required for Meta API calls")
        self._access_token = normalize_access_token(token)
        return headers

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        if _normalize_model(model) != MUSE_MODEL:
            raise ValueError(f"Unsupported Meta realtime model: {model}")
        return build_muse_realtime_url(api_base)

    def is_setup_message(self, msg_obj: Mapping[str, object]) -> bool:
        return "authorization" in msg_obj

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> OpenAIRealtimeTranscriptionSessionCreated:
        return session_created_event(_DEFAULT_SESSION_CONFIG, logging_session_id)

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> tuple[str | bytes, ...]:
        request: Final = _json_object(message)
        event_type: Final = request.get("type")
        if event_type in ("session.update", "transcription_session.update"):
            return self._configure(message, model)
        if event_type == "input_audio_buffer.append":
            return self._append_audio(request)
        if event_type == "input_audio_buffer.commit":
            return self._flush_audio(end_stream=self._require_config().mode == "PUSH_TO_TALK")
        if event_type == "input_audio_buffer.end":
            return self._flush_audio(end_stream=True)
        if event_type == "input_audio_buffer.clear":
            self._pending_audio = b""
            return ()
        verbose_logger.debug("Meta realtime: dropping unsupported client event %s", event_type)
        return ()

    async def pace_backend_send(self, message: bytes) -> None:
        now: Final = self._monotonic()
        origin: Final = self._pacing_origin
        effective_origin: Final = (
            now - self._sent_duration if origin is None or now > origin + self._sent_duration else origin
        )
        delay: Final = effective_origin + self._sent_duration - now
        if delay > 0:
            await self._sleep(delay)
        self._pacing_origin = effective_origin
        self._sent_duration += len(message) / self._require_config().bytes_per_second

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        return self._transformer.take_unbilled_usage()

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        payload: Final = message.decode("utf-8") if isinstance(message, bytes) else message
        result: Final[RealtimeResponseTypedDict] = {
            "response": list(self._backend_events(payload)),  # mutable-ok: RealtimeResponseTypedDict.response is a list
            "current_output_item_id": realtime_response_transform_input.get("current_output_item_id"),
            "current_response_id": realtime_response_transform_input.get("current_response_id"),
            "current_delta_chunks": realtime_response_transform_input.get("current_delta_chunks"),
            "current_conversation_id": realtime_response_transform_input.get("current_conversation_id"),
            "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
            "current_delta_type": realtime_response_transform_input.get("current_delta_type"),
            "session_configuration_request": realtime_response_transform_input.get("session_configuration_request"),
        }
        return result

    def _backend_events(self, payload: str) -> tuple[OpenAIRealtimeEvents, ...]:
        frame: Final = _json_object(payload)
        session_id: Final = frame.get("sessionId")
        if session_id is None:
            return self._transformer.transform(frame)
        if not isinstance(session_id, str) or not session_id.strip():
            raise MuseProtocolError("provider returned an invalid handshake response")
        return (session_created_event(self._require_config(), session_id.strip()),)

    def _configure(self, message: str, model: str) -> tuple[str, ...]:
        if self._config is not None:
            verbose_logger.debug("Meta realtime: ignoring session.update after the Muse handshake was sent")
            return ()
        access_token: Final = self._access_token
        if access_token is None:
            raise MuseProtocolError("Meta API key was not validated before the session was configured")
        config: Final = parse_session_update(message, model)
        self._config = config
        self._transformer.configure(config)
        return (json.dumps(config.handshake(access_token), separators=(",", ":")),)

    def _append_audio(self, request: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
        config: Final = self._require_config()
        encoded: Final = request.get("audio")
        if not isinstance(encoded, str):
            raise MuseProtocolError("Audio must be a base64 string")
        if len(encoded) > config.max_encoded_append_bytes:
            raise MuseProtocolError("Audio append exceeds the four-second backlog limit")
        try:
            audio: Final = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise MuseProtocolError("Audio must be valid base64") from None
        if len(audio) % 2:
            raise MuseProtocolError("PCM16 audio must contain complete samples")
        buffered: Final = self._pending_audio + audio
        packet_end: Final = len(buffered) - len(buffered) % config.packet_bytes
        self._pending_audio = buffered[packet_end:]
        return tuple(
            buffered[start : start + config.packet_bytes] for start in range(0, packet_end, config.packet_bytes)
        )

    def _flush_audio(self, *, end_stream: bool) -> tuple[str | bytes, ...]:
        remainder: Final = self._pending_audio
        self._pending_audio = b""
        frames: Final[tuple[bytes, ...]] = (remainder,) if remainder else ()
        if not end_stream or self._end_stream_sent:
            return frames
        self._end_stream_sent = True
        return (*frames, _END_STREAM)

    def _require_config(self) -> MuseSessionConfig:
        if self._config is None:
            raise MuseProtocolError("session.update must configure the Muse session before audio is sent")
        return self._config

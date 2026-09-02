from __future__ import annotations

import json
import math
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.types.realtime import RealtimeInputAudioTranscriptionUsage

MUSE_MODEL: Final = "muse-voice-transcribe-1.0"
SUPPORTED_SAMPLE_RATES: Final = frozenset((16_000, 24_000))
SUPPORTED_MODES: Final = frozenset(("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"))
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
_LANGUAGE_NAMES: Final = {  # mutable-ok: immutable-by-convention language lookup table
    language.casefold(): language for language in SUPPORTED_LANGUAGES
}
_LANGUAGE_CODES: Final = {  # mutable-ok: immutable-by-convention language lookup table
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
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
OpenAIEvent: TypeAlias = Mapping[str, object]


class MuseProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MuseSessionConfig:
    model: str
    mode: Literal["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"]
    sample_rate: Literal[16000, 24000]
    keywords: tuple[str, ...]
    language_bias: tuple[str, ...]

    @property
    def audio_encoding(self) -> Literal["PCM_16KHZ", "PCM_24KHZ"]:
        return "PCM_16KHZ" if self.sample_rate == 16_000 else "PCM_24KHZ"

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * 2

    @property
    def packet_bytes(self) -> int:
        return self.bytes_per_second * 80 // 1000

    def handshake(self, access_token: str) -> Mapping[str, object]:
        base: Final[Mapping[str, object]] = {  # mutable-ok: JSON wire payload
            "mode": self.mode,
            "authorization": {"accessToken": access_token},  # mutable-ok: JSON wire payload
            "audioEncoding": self.audio_encoding,
            "model": self.model,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,
        }
        payload: dict[str, object] = dict(base)  # mutable-ok: incrementally builds JSON wire payload
        if self.keywords:
            payload["keywords"] = list(self.keywords)  # mutable-ok: JSON arrays require concrete lists
        if self.language_bias:
            payload["languageBias"] = list(self.language_bias)  # mutable-ok: JSON arrays require concrete lists
        return payload

    def openai_session(self, session_id: str) -> Mapping[str, object]:
        turn_detection: Final[Mapping[str, object] | None] = (
            None if self.mode == "PUSH_TO_TALK" else {"type": "server_vad"}  # mutable-ok: JSON wire payload
        )
        transcription: dict[str, object] = {  # mutable-ok: incrementally builds JSON wire payload
            "model": self.model,
        }
        if self.language_bias:
            transcription["language"] = self.language_bias[0]
            transcription["language_bias"] = list(  # mutable-ok: JSON arrays require concrete lists
                self.language_bias
            )
        if self.keywords:
            transcription["keywords"] = list(self.keywords)  # mutable-ok: JSON arrays require concrete lists
        return {  # mutable-ok: JSON wire payload
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "model": self.model,
            "audio": {  # mutable-ok: JSON wire payload
                "input": {  # mutable-ok: JSON wire payload
                    "format": {"type": "audio/pcm", "rate": self.sample_rate},  # mutable-ok: JSON wire payload
                    "transcription": transcription,
                    "turn_detection": turn_detection,
                }
            },
        }


@dataclass(slots=True)
class _TurnState:
    item_id: str | None = None
    started: bool = False
    start_emitted: bool = False
    latest_partial: str | None = None
    emitted_partial: str = ""
    final_text: str | None = None
    completed_signal: bool = False
    completed_emitted: bool = False
    stopped: bool = False
    stopped_emitted: bool = False
    speaker: str | None = None


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
        return {}  # mutable-ok: empty JSON object
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


def _normalize_string_sequence(value: JsonValue | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MuseProtocolError(f"{name} must be an array of strings")
    normalized: list[str] = []  # mutable-ok: deduplicates validated language hints before freezing
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise MuseProtocolError(f"{name} entries must be non-empty strings")
        item: str = entry.strip()  # rebind-ok: normalized once for each hint
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalize_language_sequence(value: JsonValue | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_language(item) for item in _normalize_string_sequence(value, "language_bias")))


def _parse_sample_rate(session: Mapping[str, JsonValue]) -> Literal[16000, 24000]:
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
    return rate


def _parse_mode(
    session: Mapping[str, JsonValue], audio_input: Mapping[str, JsonValue]
) -> Literal["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"]:
    explicit: Final = session.get("mode")
    if explicit is not None:
        if not isinstance(explicit, str) or explicit.upper() not in SUPPORTED_MODES:
            raise MuseProtocolError("unsupported Muse Voice mode")
        normalized_mode: Final = explicit.upper()
        if normalized_mode == "PUSH_TO_TALK":
            return "PUSH_TO_TALK"
        if normalized_mode == "DIARIZATION":
            return "DIARIZATION"
        return "ENDPOINTING"
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
    session_type: Final = session.get("type")
    if session_type not in (None, "transcription", "realtime"):
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
    requested_model: Final = _string(transcription.get("model"), "transcription model")
    normalized_model: Final = _normalize_model(expected_model)
    if normalized_model != MUSE_MODEL:
        raise MuseProtocolError("unsupported Meta realtime model")
    if requested_model is not None and _normalize_model(requested_model) != normalized_model:
        raise MuseProtocolError("realtime session model cannot be changed")
    language_value: Final = _string(transcription.get("language"), "language")
    explicit_bias: Final = _normalize_language_sequence(transcription.get("language_bias"))
    language_bias: Final = tuple(
        dict.fromkeys((normalize_language(language_value), *explicit_bias))
        if language_value is not None
        else explicit_bias
    )
    keywords: Final = _normalize_string_sequence(transcription.get("keywords"), "keywords")
    return MuseSessionConfig(
        model=normalized_model,
        mode=_parse_mode(session, audio_input),
        sample_rate=_parse_sample_rate(session),
        keywords=keywords,
        language_bias=language_bias,
    )


def session_created_event(model: str, session_id: str) -> OpenAIEvent:
    normalized_model: Final = _normalize_model(model)
    default_config: Final = MuseSessionConfig(
        model=normalized_model,
        mode="ENDPOINTING",
        sample_rate=24_000,
        keywords=(),
        language_bias=(),
    )
    return {  # mutable-ok: OpenAI-compatible JSON event
        "type": "session.created",
        "event_id": f"event_{uuid.uuid4().hex}",
        "session": default_config.openai_session(session_id),
    }


def session_updated_event(config: MuseSessionConfig, session_id: str) -> OpenAIEvent:
    return {  # mutable-ok: OpenAI-compatible JSON event
        "type": "session.updated",
        "event_id": f"event_{uuid.uuid4().hex}",
        "session": config.openai_session(session_id),
    }


def error_event(error_type: str, code: str, message: str) -> OpenAIEvent:
    return {  # mutable-ok: OpenAI-compatible JSON event
        "type": "error",
        "event_id": f"event_{uuid.uuid4().hex}",
        "error": {  # mutable-ok: nested OpenAI-compatible error object
            "type": error_type,
            "code": code,
            "message": message,
        },
    }


class MuseEventTransformer:
    def __init__(self, *, completed_turn_limit: int = 128) -> None:
        self._turns: OrderedDict[str, _TurnState] = OrderedDict()  # mutable-ok: ordered active-turn state
        self._active_turn_id: str | None = None
        self._mode: Literal["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"] = "ENDPOINTING"
        self._completed_turn_ids: set[str] = set()  # mutable-ok: bounded completed-turn membership
        self._completed_turn_order: deque[str] = deque(  # mutable-ok: bounded completion eviction order
            maxlen=completed_turn_limit
        )
        self._completed_turn_limit: Final = completed_turn_limit
        self._pending_item_ids: deque[str] = deque()  # mutable-ok: FIFO commit correlation state
        self._last_committed_item_id: str | None = None
        self._last_audio_processed_ms: float = 0.0
        self._unassigned_usage_seconds: float = 0.0

    def configure(self, config: MuseSessionConfig) -> None:
        self._mode = config.mode

    def transform(self, payload: str) -> tuple[OpenAIEvent, ...]:
        message: Final = _json_object(payload)
        event_type: Final = message.get("type")
        if event_type == "error":
            return (error_event("server_error", "provider_error", "Meta Muse realtime transcription failed"),)
        if event_type == "audioProgress":
            self._update_audio_progress(message)
            return ()
        if event_type == "speechStart":
            self._speech_start(message)
        elif event_type == "transcript":
            self._transcript(message)
        elif event_type == "speaker":
            self._speaker(message)
        elif event_type == "speechEnd":
            self._speech_end(message)
        elif event_type == "speechComplete":
            self._speech_complete(message)
        else:
            return ()
        return self._drain()

    def commit_item(self) -> tuple[str | None, str]:
        previous_item_id: Final = self._last_committed_item_id
        provider_turn_id: Final = self._active_turn_id
        active_turn: Final = self._turns.get(provider_turn_id) if provider_turn_id is not None else None
        item_id: Final = (
            active_turn.item_id or provider_turn_id
            if active_turn is not None and provider_turn_id is not None
            else f"item_{uuid.uuid4().hex}"
        )
        if active_turn is not None:
            active_turn.item_id = item_id
        else:
            self._pending_item_ids.append(item_id)
        self._last_committed_item_id = item_id
        return previous_item_id, item_id

    def take_unbilled_usage(self) -> RealtimeInputAudioTranscriptionUsage | None:
        seconds: Final = self._unassigned_usage_seconds
        if seconds <= 0:
            return None
        self._unassigned_usage_seconds = 0.0
        return {"type": "duration", "seconds": seconds}  # mutable-ok: typed usage wire payload

    def _turn(self, turn_id: str) -> _TurnState:
        if turn_id in self._completed_turn_ids:
            raise _CompletedTurn
        turn: Final = self._turns.get(turn_id)
        if turn is not None:
            return turn
        created: Final = _TurnState(item_id=self._pending_item_ids.popleft() if self._pending_item_ids else turn_id)
        self._turns[turn_id] = created
        return created

    def _speech_start(self, message: Mapping[str, JsonValue]) -> None:
        turn_id: Final = self._required_turn_id(message, "speechStart")
        try:
            turn: Final = self._turn(turn_id)
        except _CompletedTurn:
            return
        turn.started = True
        self._active_turn_id = turn_id

    def _transcript(self, message: Mapping[str, JsonValue]) -> None:
        transcript: Final = message.get("transcript")
        if not isinstance(transcript, str):
            raise MuseProtocolError("transcript event has invalid transcript")
        if not transcript and message.get("turnId") is None and self._active_turn_id is None:
            return
        turn_id: Final = self._transcript_turn_id(message)
        try:
            turn: Final = self._turn(turn_id)
        except _CompletedTurn:
            return
        final: Final = message.get("final") is True
        if final:
            turn.final_text = transcript
            turn.completed_signal = True
            if self._mode == "PUSH_TO_TALK":
                turn.stopped = True
                if self._active_turn_id == turn_id:
                    self._active_turn_id = None
            return
        if turn.final_text is None:
            turn.latest_partial = transcript

    def _speaker(self, message: Mapping[str, JsonValue]) -> None:
        turn_id: Final = (
            self._required_turn_id(message, "speaker") if message.get("turnId") is not None else self._active_turn_id
        )
        if turn_id is None:
            raise MuseProtocolError("speaker event arrived outside an active turn")
        label: Final = message.get("label")
        if not isinstance(label, str) or not label.strip():
            raise MuseProtocolError("speaker event has invalid label")
        try:
            turn: Final = self._turn(turn_id)
        except _CompletedTurn:
            return
        turn.speaker = label.strip()

    def _speech_end(self, message: Mapping[str, JsonValue]) -> None:
        turn_id: Final = self._required_turn_id(message, "speechEnd")
        try:
            turn: Final = self._turn(turn_id)
        except _CompletedTurn:
            return
        turn.stopped = True
        if self._active_turn_id == turn_id:
            self._active_turn_id = None

    def _speech_complete(self, message: Mapping[str, JsonValue]) -> None:
        turn_id: Final = self._required_turn_id(message, "speechComplete")
        transcript: Final = message.get("transcript")
        if not isinstance(transcript, str):
            raise MuseProtocolError("speechComplete event has invalid transcript")
        try:
            turn: Final = self._turn(turn_id)
        except _CompletedTurn:
            return
        turn.final_text = transcript
        turn.completed_signal = True

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
        self._unassigned_usage_seconds += (float(processed_ms) - self._last_audio_processed_ms) / 1000
        self._last_audio_processed_ms = float(processed_ms)

    def _drain(self) -> tuple[OpenAIEvent, ...]:
        events: list[OpenAIEvent] = []  # mutable-ok: ordered events are frozen to a tuple before return
        while self._turns:
            turn_id: str = next(iter(self._turns))  # rebind-ok: selects the next ordered turn
            turn: _TurnState = self._turns[turn_id]  # rebind-ok: state for the selected turn
            has_content: bool = (  # rebind-ok: evaluated for the selected turn
                turn.latest_partial is not None or turn.final_text is not None
            )
            item_id: str = turn.item_id or turn_id  # rebind-ok: selected for each ordered turn
            if (turn.started or has_content) and not turn.start_emitted:
                turn.start_emitted = True
                events.append(self._speech_event("input_audio_buffer.speech_started", item_id))
            if turn.latest_partial is not None and turn.final_text is None:
                delta: str = self._new_suffix(  # rebind-ok: computed for the selected turn
                    turn.emitted_partial, turn.latest_partial
                )
                if delta:
                    turn.emitted_partial = turn.latest_partial
                    events.append(
                        {  # mutable-ok: OpenAI-compatible JSON event
                            "type": "conversation.item.input_audio_transcription.delta",
                            "event_id": f"event_{uuid.uuid4().hex}",
                            "item_id": item_id,
                            "content_index": 0,
                            "delta": delta,
                        }
                    )
            if turn.stopped and not turn.stopped_emitted:
                turn.stopped_emitted = True
                events.append(self._speech_event("input_audio_buffer.speech_stopped", item_id))
            if turn.final_text is not None and turn.stopped_emitted and not turn.completed_emitted:
                turn.completed_emitted = True
                usage: RealtimeInputAudioTranscriptionUsage | None = (  # rebind-ok: usage assigned per turn
                    self.take_unbilled_usage()
                )
                completed_event: dict[str, object] = {  # mutable-ok: incrementally builds OpenAI JSON event
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": turn.final_text,
                }
                if turn.speaker is not None:
                    completed_event["speaker"] = turn.speaker
                if usage is not None:
                    completed_event["usage"] = usage
                events.append(completed_event)
            if not (turn.completed_emitted and (turn.stopped or turn.completed_signal)):
                break
            del self._turns[turn_id]
            self._remember_completed(turn_id)
        return tuple(events)

    def _remember_completed(self, turn_id: str) -> None:
        if turn_id in self._completed_turn_ids:
            return
        if len(self._completed_turn_order) >= self._completed_turn_limit:
            self._completed_turn_ids.discard(self._completed_turn_order.popleft())
        self._completed_turn_order.append(turn_id)
        self._completed_turn_ids.add(turn_id)

    def _transcript_turn_id(self, message: Mapping[str, JsonValue]) -> str:
        if message.get("turnId") is not None:
            return self._required_turn_id(message, "transcript")
        if self._active_turn_id is not None:
            return self._active_turn_id
        if self._mode != "PUSH_TO_TALK":
            raise MuseProtocolError("transcript event is missing turnId outside an active turn")
        turn_id: Final = f"item_{uuid.uuid4().hex}"
        self._active_turn_id = turn_id
        return turn_id

    @staticmethod
    def _required_turn_id(message: Mapping[str, JsonValue], event: str) -> str:
        value: Final = message.get("turnId")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise MuseProtocolError(f"{event} event has invalid turnId")
        turn_id: Final = str(value).strip()
        if not turn_id:
            raise MuseProtocolError(f"{event} event has invalid turnId")
        return turn_id

    @staticmethod
    def _speech_event(event_type: str, turn_id: str) -> OpenAIEvent:
        return {  # mutable-ok: OpenAI-compatible JSON event
            "type": event_type,
            "event_id": f"event_{uuid.uuid4().hex}",
            "item_id": turn_id,
        }

    @staticmethod
    def _new_suffix(previous: str, current: str) -> str:
        if current.startswith(previous):
            return current[len(previous) :]
        return ""


class _CompletedTurn(Exception):
    pass


def encode_event(event: Mapping[str, object]) -> str:
    return json.dumps(event, separators=(",", ":"))

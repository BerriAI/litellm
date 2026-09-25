import base64
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import JsonValue

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transcription_protocol import (
    RealtimeTranscriptionProtocolError,
    TranscriptionSessionUpdate,
    completed_event,
    decode_pcm16_append,
    delta_event,
    duration_usage,
    error_event,
    json_object,
    json_string,
    parse_transcription_session_update,
    speech_event,
    transcription_session,
    transcription_session_created_event,
    transcription_session_updated_event,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeTranscriptionSession,
)
from litellm.types.llms.sarvam import (
    SarvamAudioInput,
    SarvamConfigUpdate,
    SarvamControlEvent,
    SarvamControlEventType,
    SarvamEndpointing,
    SarvamSampleRate,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.types.utils import LlmProviders

DEFAULT_SARVAM_REALTIME_API_BASE: Final = "wss://api.sarvam.ai"
SARVAM_REALTIME_PATH: Final = "/speech-to-text-realtime/ws"
SARVAM_REALTIME_ENCODING: Final = "linear16"
DEFAULT_SARVAM_REALTIME_MODEL: Final = "saaras:v3-realtime"
REALTIME_ENDPOINT: Final = "/v1/realtime"
SUPPORTED_SAMPLE_RATES: Final = frozenset({8_000, 16_000})
AUTO_LANGUAGE: Final = "auto"
SUPPORTED_LANGUAGE_CODES: Final = (
    "as-IN",
    "bn-IN",
    "brx-IN",
    "doi-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "kok-IN",
    "ks-IN",
    "mai-IN",
    "ml-IN",
    "mni-IN",
    "mr-IN",
    "ne-IN",
    "or-IN",
    "pa-IN",
    "sa-IN",
    "sat-IN",
    "sd-IN",
    "ta-IN",
    "te-IN",
    "ur-IN",
)
_LANGUAGE_BY_CODE: Final = MappingProxyType({code.casefold(): code for code in SUPPORTED_LANGUAGE_CODES})
_LANGUAGE_BY_PRIMARY: Final = MappingProxyType({code.split("-")[0]: code for code in SUPPORTED_LANGUAGE_CODES})
_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "wss": "wss", "http": "ws", "ws": "ws"})
_UTTERANCE_LIMIT: Final = 128


class SarvamProtocolError(RealtimeTranscriptionProtocolError):
    pass


def is_realtime_transcription_model(model: str) -> bool:
    """Support follows the model registry rather than a list in this file, so a new Sarvam realtime model
    only needs a cost map entry. Only an exact hit counts: the registry resolves unknown names to a family."""
    registry_key: Final = f"{LlmProviders.SARVAM.value}/{model}"
    try:
        model_info: Final = litellm.get_model_info(model=registry_key, custom_llm_provider=LlmProviders.SARVAM.value)
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return False
    return (
        model_info["key"] == registry_key
        and model_info.get("mode") == "audio_transcription"
        and REALTIME_ENDPOINT in (model_info.get("supported_endpoints") or ())
    )


def normalize_model(model: str) -> str:
    normalized: Final = model.removeprefix(f"{LlmProviders.SARVAM.value}/").strip()
    if not is_realtime_transcription_model(normalized):
        raise SarvamProtocolError(
            f"unsupported Sarvam realtime model: {model}. Add it to the model cost map with "
            f'"mode": "audio_transcription" and "supported_endpoints": ["{REALTIME_ENDPOINT}"]'
        )
    return normalized


def normalize_language_code(language: str) -> str:
    value: Final = language.strip().replace("_", "-").casefold()
    if not value:
        raise SarvamProtocolError("language must be non-empty")
    if value == AUTO_LANGUAGE:
        return AUTO_LANGUAGE
    documented: Final = _LANGUAGE_BY_CODE.get(value)
    if documented is not None:
        return documented
    mapped: Final = _LANGUAGE_BY_PRIMARY.get(value.split("-", 1)[0])
    if mapped is None:
        raise SarvamProtocolError(f"unsupported Sarvam realtime language: {language}")
    return mapped


@dataclass(frozen=True, slots=True)
class SarvamConnection:
    """Everything fixed when the backend socket opens, before the client's first ``session.update``."""

    model: str
    sample_rate: SarvamSampleRate
    url: str


def _connection_sample_rate(query: str) -> SarvamSampleRate:
    declared: Final = tuple(value for key, value in parse_qsl(query) if key == "sample_rate")
    if not declared:
        return 16_000
    if len(declared) != 1 or not declared[0].isdigit() or int(declared[0]) not in SUPPORTED_SAMPLE_RATES:
        raise ValueError("Sarvam realtime supports sample_rate=16000 or sample_rate=8000")
    return 8_000 if int(declared[0]) == 8_000 else 16_000


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _netloc(hostname: str, port: int | None) -> str:
    host: Final = f"[{hostname}]" if ":" in hostname else hostname
    return host if port is None else f"{host}:{port}"


def build_sarvam_connection(api_base: str | None, model: str) -> SarvamConnection:
    """Sarvam fixes ``encoding`` and ``sample_rate`` at connect time, so the rate is read from ``api_base``.

    ``language_code`` is not: the client's language arrives later in ``session.update`` and is applied with a
    ``config.update``, so the socket always opens on Sarvam's adaptive ``auto``.
    """
    normalized_model: Final = normalize_model(model)
    parsed: Final = urlparse((api_base or DEFAULT_SARVAM_REALTIME_API_BASE).strip())
    scheme: Final = _WEBSOCKET_SCHEMES.get(parsed.scheme)
    if scheme is None or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("Sarvam api_base must be an absolute wss:// or https:// URL without credentials")
    if scheme == "ws" and not _is_loopback(parsed.hostname):
        # The subscription key travels as a connection header, so cleartext is only ever for a local test double.
        raise ValueError("Sarvam api_base must use wss:// or https:// unless it points at a loopback address")
    sample_rate: Final = _connection_sample_rate(parsed.query)
    netloc: Final = _netloc(parsed.hostname, parsed.port)
    query: Final = urlencode(
        (
            ("model", normalized_model),
            ("language_code", AUTO_LANGUAGE),
            ("encoding", SARVAM_REALTIME_ENCODING),
            ("sample_rate", str(sample_rate)),
        )
    )
    return SarvamConnection(
        model=normalized_model,
        sample_rate=sample_rate,
        url=urlunparse((scheme, netloc, SARVAM_REALTIME_PATH, "", query, "")),
    )


@dataclass(frozen=True, slots=True)
class SarvamSessionUpdate:
    language: str | None
    endpointing: SarvamEndpointing


def _validate_audio_format(update: TranscriptionSessionUpdate, sample_rate: SarvamSampleRate) -> None:
    audio_format: Final = update.audio_format
    if audio_format is None:
        return
    if not audio_format.is_pcm16:
        raise SarvamProtocolError("Sarvam realtime requires pcm16 input audio")
    if audio_format.channels not in (None, 1):
        raise SarvamProtocolError("Sarvam realtime requires mono input audio")
    if audio_format.rate not in (None, sample_rate):
        raise SarvamProtocolError(
            f"Sarvam realtime is connected at {sample_rate} Hz; set sample_rate on the deployment api_base to change it"
        )


def _parse_endpointing(update: TranscriptionSessionUpdate, current: SarvamEndpointing) -> SarvamEndpointing:
    """A later ``session.update`` that changes only the language leaves turn detection out entirely, which must
    keep the endpointing the session already runs on rather than reverting it to VAD."""
    if update.turn_detection_disabled:
        return "manual"
    if update.turn_detection is None:
        return current
    if update.turn_detection_type != "server_vad":
        raise SarvamProtocolError("Sarvam realtime supports server_vad turn detection or null")
    return "vad"


def parse_session_update(
    payload: str, connection: SarvamConnection, endpointing: SarvamEndpointing
) -> SarvamSessionUpdate:
    update: Final = parse_transcription_session_update(payload, SarvamProtocolError)
    if update.session_type not in (None, "transcription", "realtime"):
        raise SarvamProtocolError("Sarvam realtime supports transcription sessions only")
    if update.unsupported_transcription_keys:
        verbose_logger.warning(
            "Sarvam realtime: dropping unsupported transcription settings %s", update.unsupported_transcription_keys
        )
    if update.model is not None and normalize_model(update.model) != connection.model:
        raise SarvamProtocolError("realtime session model cannot be changed")
    _validate_audio_format(update, connection.sample_rate)
    return SarvamSessionUpdate(
        language=None if update.language is None else normalize_language_code(update.language),
        endpointing=_parse_endpointing(update, endpointing),
    )


def openai_session(
    connection: SarvamConnection, session_id: str, language: str, server_vad: bool
) -> OpenAIRealtimeTranscriptionSession:
    return transcription_session(
        session_id=session_id,
        model=connection.model,
        sample_rate=connection.sample_rate,
        language=None if language == AUTO_LANGUAGE else language,
        server_vad=server_vad,
    )


def _control(event: SarvamControlEventType) -> str:
    frame: Final[SarvamControlEvent] = {"event": event}
    return json.dumps(frame, separators=(",", ":"))


_SPEECH_START: Final = _control("speech_start")
_SPEECH_END: Final = _control("speech_end")
_FLUSH: Final = _control("flush")
_END_SESSION: Final = _control("end")


@dataclass(slots=True)
class _UtteranceState:
    item_id: str
    start_emitted: bool = False
    emitted_partial: str = ""
    stop_emitted: bool = False
    completed_emitted: bool = False


class _UtteranceTracker:
    """Per-``utterance_idx`` emit state, bounded so a long session cannot grow without limit."""

    def __init__(self, *, limit: int = _UTTERANCE_LIMIT) -> None:
        self._states: dict[int, _UtteranceState] = {}  # mutable-ok: bounded, insertion-ordered emit state
        self._limit: Final = limit

    def state(self, utterance_idx: int) -> _UtteranceState:
        existing: Final = self._states.get(utterance_idx)
        if existing is not None:
            return existing
        created: Final = _UtteranceState(item_id=f"item_{uuid.uuid4().hex}")
        self._states[utterance_idx] = created
        if len(self._states) > self._limit:
            del self._states[next(iter(self._states))]
        return created


def _utterance_index(message: Mapping[str, JsonValue], event: str) -> int:
    value: Final = message.get("utterance_idx")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SarvamProtocolError(f"{event} event has an invalid utterance_idx")
    return value


def _transcript_text(message: Mapping[str, JsonValue], event: str) -> str:
    text: Final = message.get("text")
    if not isinstance(text, str):
        raise SarvamProtocolError(f"{event} event has an invalid text")
    return text


def _audio_seconds(message: Mapping[str, JsonValue]) -> float | None:
    value: Final = message.get("audio_duration_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


class SarvamRealtimeConfig(BaseRealtimeConfig):
    """Translates between OpenAI realtime transcription sessions and Sarvam's ``/speech-to-text-realtime/ws``.

    One instance per session: ``get_complete_url`` pins the connection and the rest of the methods read it back.
    """

    def __init__(self, *, utterance_limit: int = _UTTERANCE_LIMIT) -> None:
        self._utterances: Final = _UtteranceTracker(limit=utterance_limit)
        self._connection: SarvamConnection | None = None
        self._language: str = AUTO_LANGUAGE
        self._endpointing: SarvamEndpointing = "vad"
        self._session_id: str | None = None
        self._speech_open: bool = False
        self._forwarded_audio_bytes: int = 0
        self._reported_seconds: float | None = None

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseRealtimeConfig contract
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig contract
        key: Final = (api_key or get_secret_str("SARVAM_API_KEY") or "").strip()
        if not key:
            raise ValueError("api_key is required for Sarvam API calls")
        return {**headers, "api-subscription-key": key}  # mutable-ok: BaseRealtimeConfig returns a plain header dict

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        connection: Final = build_sarvam_connection(api_base, model)
        self._connection = connection
        return connection.url

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> tuple[str | bytes, ...]:
        request: Final = json_object(message, SarvamProtocolError)
        match request.get("type"):
            case "session.update" | "transcription_session.update":
                return self._configure(message)
            case "input_audio_buffer.append":
                return self._append_audio(request)
            case "input_audio_buffer.commit":
                return self._finalize_utterance()
            case "input_audio_buffer.end":
                return (*self._finalize_utterance(), _END_SESSION)
            case "input_audio_buffer.clear":
                # Sarvam has no buffered-audio discard; audio already forwarded cannot be taken back.
                return ()
            case unsupported:
                verbose_logger.debug("Sarvam realtime: dropping unsupported client event %s", unsupported)
        return ()

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        """Sarvam's own ``audio_duration_s`` is the billed quantity; the forwarded audio only covers a session
        that ended before ``session.end`` arrived (a client that drops the socket mid-utterance)."""
        connection: Final = self._connection
        if connection is None:
            return None
        reported: Final = self._reported_seconds
        seconds: Final = (
            reported if reported is not None else self._forwarded_audio_bytes / (connection.sample_rate * 2)
        )
        self._reported_seconds = None
        self._forwarded_audio_bytes = 0
        return duration_usage(seconds) if seconds > 0 else None

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        payload: Final = message.decode("utf-8") if isinstance(message, bytes) else message
        response: Final[RealtimeResponseTypedDict] = {
            "response": list(self._backend_events(payload)),  # mutable-ok: RealtimeResponseTypedDict.response is a list
            "current_output_item_id": realtime_response_transform_input.get("current_output_item_id"),
            "current_response_id": realtime_response_transform_input.get("current_response_id"),
            "current_delta_chunks": realtime_response_transform_input.get("current_delta_chunks"),
            "current_conversation_id": realtime_response_transform_input.get("current_conversation_id"),
            "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
            "current_delta_type": realtime_response_transform_input.get("current_delta_type"),
            "session_configuration_request": realtime_response_transform_input.get("session_configuration_request"),
        }
        return response

    def _require_connection(self) -> SarvamConnection:
        connection: Final = self._connection
        if connection is None:
            raise SarvamProtocolError("the Sarvam realtime session was used before its url was built")
        return connection

    def _configure(self, message: str) -> tuple[str, ...]:
        update: Final = parse_session_update(message, self._require_connection(), self._endpointing)
        language: Final = update.language or self._language
        language_changed: Final = language != self._language
        endpointing_changed: Final = update.endpointing != self._endpointing
        self._language = language
        self._endpointing = update.endpointing
        if language_changed and endpointing_changed:
            both: Final[SarvamConfigUpdate] = {
                "event": "config.update",
                "language_code": language,
                "endpointing": update.endpointing,
            }
            return (json.dumps(both, separators=(",", ":")),)
        if language_changed:
            language_only: Final[SarvamConfigUpdate] = {"event": "config.update", "language_code": language}
            return (json.dumps(language_only, separators=(",", ":")),)
        if endpointing_changed:
            endpointing_only: Final[SarvamConfigUpdate] = {
                "event": "config.update",
                "endpointing": update.endpointing,
            }
            return (json.dumps(endpointing_only, separators=(",", ":")),)
        return ()

    def _append_audio(self, request: Mapping[str, JsonValue]) -> tuple[str, ...]:
        audio: Final = decode_pcm16_append(request.get("audio"), None, SarvamProtocolError)
        if not audio:
            return ()
        self._forwarded_audio_bytes += len(audio)
        frame: Final[SarvamAudioInput] = {"event": "audio_input", "audio": base64.b64encode(audio).decode("ascii")}
        return (*self._open_speech(), json.dumps(frame, separators=(",", ":")))

    def _open_speech(self) -> tuple[str, ...]:
        """Manual endpointing makes the client responsible for utterance boundaries, which OpenAI clients signal
        by appending audio and committing rather than with explicit speech events."""
        if self._endpointing != "manual" or self._speech_open:
            return ()
        self._speech_open = True
        return (_SPEECH_START,)

    def _finalize_utterance(self) -> tuple[str, ...]:
        if self._endpointing != "manual":
            verbose_logger.debug("Sarvam realtime: input_audio_buffer.commit is a no-op under VAD endpointing")
            return ()
        if not self._speech_open:
            return ()
        self._speech_open = False
        return (_SPEECH_END, _FLUSH)

    def _backend_events(self, payload: str) -> tuple[OpenAIRealtimeEvents, ...]:
        frame: Final = json_object(payload, SarvamProtocolError)
        match frame.get("event"):
            case "session.begin":
                return (self._session_created(frame),)
            case "vad.speech_start":
                return self._speech_started(_utterance_index(frame, "vad.speech_start"))
            case "vad.speech_end":
                return self._speech_stopped(_utterance_index(frame, "vad.speech_end"))
            case "transcript.partial":
                return self._partial(frame)
            case "transcript.final":
                return self._final(frame)
            case "config.updated":
                return self._session_updated()
            case "session.end":
                self._reported_seconds = _audio_seconds(frame)
                return ()
            case "error":
                return (error_event(self._error_message(frame)),)
            case ignored:
                verbose_logger.debug("Sarvam realtime: no client event for backend event %s", ignored)
        return ()

    def _session_created(self, frame: Mapping[str, JsonValue]) -> OpenAIRealtimeEvents:
        request_id: Final = json_string(frame.get("request_id"), "session.begin request_id", SarvamProtocolError)
        if not request_id:
            raise SarvamProtocolError("session.begin event has an invalid request_id")
        self._session_id = request_id
        return transcription_session_created_event(self._openai_session(request_id))

    def _session_updated(self) -> tuple[OpenAIRealtimeEvents, ...]:
        """Sarvam opens the socket on its own defaults and acknowledges each ``config.update`` separately, so a
        client that configures the session after ``session.created`` only learns what took effect from here."""
        session_id: Final = self._session_id
        if session_id is None:
            return ()
        return (transcription_session_updated_event(self._openai_session(session_id)),)

    def _openai_session(self, session_id: str) -> OpenAIRealtimeTranscriptionSession:
        return openai_session(self._require_connection(), session_id, self._language, self._endpointing == "vad")

    def _speech_started(self, utterance_idx: int) -> tuple[OpenAIRealtimeEvents, ...]:
        state: Final = self._utterances.state(utterance_idx)
        if state.start_emitted:
            return ()
        state.start_emitted = True
        return (speech_event("input_audio_buffer.speech_started", state.item_id),)

    def _speech_stopped(self, utterance_idx: int) -> tuple[OpenAIRealtimeEvents, ...]:
        state: Final = self._utterances.state(utterance_idx)
        if state.stop_emitted:
            return ()
        state.stop_emitted = True
        return (*self._speech_started(utterance_idx), speech_event("input_audio_buffer.speech_stopped", state.item_id))

    def _partial(self, frame: Mapping[str, JsonValue]) -> tuple[OpenAIRealtimeEvents, ...]:
        utterance_idx: Final = _utterance_index(frame, "transcript.partial")
        text: Final = _transcript_text(frame, "transcript.partial")
        state: Final = self._utterances.state(utterance_idx)
        if state.completed_emitted:
            return ()
        started: Final = self._speech_started(utterance_idx)
        # Sarvam revises partials mid-utterance, so a partial can drop words it previously emitted. Deltas can
        # only ever append, so a revision emits nothing and the emitted prefix is kept: re-sending from a shorter
        # revision would duplicate words on the client. The authoritative text arrives in the completed event.
        if not text.startswith(state.emitted_partial) or text == state.emitted_partial:
            return started
        delta: Final = text[len(state.emitted_partial) :]
        state.emitted_partial = text
        return (*started, delta_event(state.item_id, delta))

    def _final(self, frame: Mapping[str, JsonValue]) -> tuple[OpenAIRealtimeEvents, ...]:
        utterance_idx: Final = _utterance_index(frame, "transcript.final")
        text: Final = _transcript_text(frame, "transcript.final")
        state: Final = self._utterances.state(utterance_idx)
        if state.completed_emitted:
            return ()
        stopped: Final = self._speech_stopped(utterance_idx)
        state.completed_emitted = True
        return (*stopped, completed_event(state.item_id, text, None))

    def _error_message(self, frame: Mapping[str, JsonValue]) -> str:
        code: Final = json_string(frame.get("code"), "error code", SarvamProtocolError) or "unknown"
        message: Final = json_string(frame.get("message"), "error message", SarvamProtocolError) or ""
        return f"Sarvam realtime error [{code}]{f': {message}' if message else ''}"

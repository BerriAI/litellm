import json
from collections.abc import Mapping
from types import TracebackType
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, ReadOnly

from litellm import verbose_logger
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.realtime.transcription_protocol import (
    SESSION_UPDATE_EVENT_TYPES,
    RealtimeTranscriptionProtocolError,
    completed_event,
    decode_pcm16_append,
    duration_usage,
    error_event,
    ignored_session_settings,
    json_object,
    new_event_id,
    parse_transcription_session_update,
    speech_event,
    transcription_session,
    transcription_session_created_event,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig, RealtimeBackend
from litellm.llms.gnani.common_utils import normalize_language
from litellm.llms.gnani.realtime.audio import WIRE_RATE, GnaniAudioStream
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    OpenAIRealtimeErrorEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeInputAudioBufferCommitted,
    OpenAIRealtimeInputAudioTranscriptionCompleted,
    OpenAIRealtimeTranscriptionSessionUpdated,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

_MANAGED_HEADERS: Final = frozenset(
    ("x-api-key-id", "authorization", "model", "x-model", "x-sample-rate", "language_code", "lang_code")
)
_DEFAULT_LANGUAGE: Final = "indic-auto"
_SERVER_VAD: Final = (None, "server_vad")


class _ActiveSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    language_code: str | None = None
    sample_rate: int | None = None


class _ProviderEvent(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    type: str
    message: str | None = None
    config: _ActiveSettings | None = None
    request_id: str | None = None
    segment_id: str | None = None
    utterance_id: str | None = None
    text: str | None = None
    detected_language_code: str | None = None
    audio_duration_ms: float | None = None


class _Transcript(OpenAIRealtimeInputAudioTranscriptionCompleted):
    detected_language_code: NotRequired[ReadOnly[str]]


class _ErrorEvent(OpenAIRealtimeErrorEvent):
    event_id: ReadOnly[str]


def _language(value: str) -> str:
    hint: Final = value.strip()
    if not hint or "," in hint:
        raise RealtimeTranscriptionProtocolError("Gnani realtime requires a single language or indic-auto")
    return normalize_language(hint)


class _MeteredBackend:
    def __init__(self, backend: RealtimeBackend) -> None:
        self._backend: Final = backend
        self.recv: Final = backend.recv
        self.close: Final = backend.close
        self._unreported_bytes: int = 0

    async def __aenter__(self) -> "_MeteredBackend":
        await self._backend.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._backend.__aexit__(exc_type, exc_value, traceback)

    async def send(self, message: str | bytes) -> None:
        await self._backend.send(message)
        if isinstance(message, bytes):
            self._unreported_bytes += len(message)

    def take_usage(self, seconds: float | None) -> RealtimeInputAudioTranscriptionUsage | None:
        pending: Final = self._unreported_bytes
        taken: Final = pending if seconds is None else min(pending, 2 * round(seconds * WIRE_RATE))
        self._unreported_bytes = pending - taken
        return duration_usage(taken / (2 * WIRE_RATE)) if taken else None


class GnaniRealtimeConfig(BaseRealtimeConfig):
    def __init__(self, drop_params: bool = False, requires_explicit_language: bool = False) -> None:
        self._drop_params: Final = drop_params
        self._requires_explicit_language: Final = requires_explicit_language
        self._backend: _MeteredBackend | None = None
        self._audio: Final = GnaniAudioStream()
        self._language: str = _DEFAULT_LANGUAGE
        self._session_id: str = new_event_id()
        self._previous_item: str | None = None
        self._started_items: frozenset[str] = frozenset()

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        base: Final = urlsplit(api_base or get_secret_str("GNANI_API_BASE") or "https://api.vachana.ai")
        if base.scheme not in ("https", "wss"):
            raise ValueError("Gnani realtime api_base must be an https:// or wss:// URL")
        path: Final = ensure_api_route(base.path, "stt/v3/stream")
        return urlunsplit(("wss", base.netloc, path, base.query, ""))

    def requires_session_configuration(self) -> bool:
        return True

    def validate_environment(
        self, headers: Mapping[str, str], model: str, api_key: str | None = None
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig requires returning a header dict
        key: Final = api_key or get_secret_str("GNANI_API_KEY")
        if not key:
            raise ValueError("Set GNANI_API_KEY or pass api_key for Gnani realtime transcription")
        lower: Final = {name.lower(): value for name, value in headers.items()}
        hints: Final = lower.get("language_code") or lower.get("lang_code") or _DEFAULT_LANGUAGE
        self._language = ",".join(_language(hint) for hint in hints.split(","))
        return {
            **{name: value for name, value in lower.items() if name not in _MANAGED_HEADERS},
            "x-api-key-id": key,
            "model": model,
            "language_code": self._language,
            "x-sample-rate": str(WIRE_RATE),
        }

    def transform_realtime_request(
        self, message: str, model: str, session_configuration_request: str | None = None
    ) -> tuple[str | bytes, ...]:
        request: Final = json_object(message)
        kind: Final = request.get("type")
        if isinstance(kind, str) and kind in SESSION_UPDATE_EVENT_TYPES:
            return (self._update(message),)
        if kind == "input_audio_buffer.append":
            self._require_explicit_language(self._language)
            return self._audio.append(decode_pcm16_append(request.get("audio"), self._audio.max_encoded_append_bytes))
        if kind == "input_audio_buffer.commit":
            return self._audio.commit()
        raise RealtimeTranscriptionProtocolError(f"Gnani transcription does not support client event {kind!r}")

    def _require_explicit_language(self, language: str) -> None:
        if self._requires_explicit_language and (language == _DEFAULT_LANGUAGE or "," in language):
            raise RealtimeTranscriptionProtocolError(
                "Prisma v2.5 requires one explicit language, for example language 'hi'"
            )

    def _update(self, message: str) -> str:
        update: Final = parse_transcription_session_update(message)
        if update.session_type not in (None, "transcription"):
            raise RealtimeTranscriptionProtocolError("Gnani realtime supports transcription sessions only")
        keeps_server_vad: Final = not update.turn_detection_disabled and update.turn_detection_type in _SERVER_VAD
        if not keeps_server_vad and not self._drop_params:
            raise RealtimeTranscriptionProtocolError(
                "Gnani realtime supports server_vad only. Set drop_params to use server_vad instead"
            )
        ignored: Final = ignored_session_settings(update)
        if ignored and not self._drop_params:
            raise RealtimeTranscriptionProtocolError(
                f"Unsupported Gnani transcription settings: {', '.join(ignored)}. Set drop_params to ignore them"
            )
        if ignored:
            verbose_logger.warning("Gnani realtime: dropping unsupported session settings %s", ignored)
        language: Final = _language(update.language) if update.language is not None else None
        if language is not None:
            self._require_explicit_language(language)
        fmt: Final = update.audio_format
        if fmt is not None:
            if not fmt.is_pcm16 or fmt.channels not in (None, 1):
                raise RealtimeTranscriptionProtocolError("Gnani realtime requires mono PCM16 audio")
            self._audio.set_rate(24_000 if fmt.rate is None else fmt.rate)
        self._language = language or self._language
        return json.dumps({"type": "update_settings", **({"language_code": language} if language else {})})

    def wrap_backend(self, backend: RealtimeBackend) -> RealtimeBackend:
        self._backend = _MeteredBackend(backend)
        return self._backend

    def _take_usage(self, seconds: float | None) -> RealtimeInputAudioTranscriptionUsage | None:
        return self._backend.take_usage(seconds) if self._backend is not None else None

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        return self._take_usage(None)

    async def pace_backend_send(self, message: bytes) -> None:
        await self._audio.pace()

    def _session(
        self, event: _ProviderEvent, model: str, kind: Literal["session.created", "session.updated"]
    ) -> OpenAIRealtimeEvents:
        if event.config is not None:
            if event.config.sample_rate not in (None, WIRE_RATE):
                raise ValueError("Gnani negotiated an unexpected audio rate")
            self._language = event.config.language_code or self._language
        session: Final = transcription_session(
            session_id=self._session_id,
            model=model,
            sample_rate=self._audio.rate,
            language=self._language,
            server_vad=True,
        )
        if kind == "session.created":
            return transcription_session_created_event(session)
        updated: Final[OpenAIRealtimeTranscriptionSessionUpdated] = {
            "type": kind,
            "event_id": new_event_id(),
            "session": session,
        }
        return updated

    def _events(self, event: _ProviderEvent, model: str) -> tuple[OpenAIRealtimeEvents, ...]:
        if event.type == "connected":
            self._session_id = event.request_id or self._session_id
            return (self._session(event, model, "session.created"),)
        if event.type == "settings_updated":
            return (self._session(event, model, "session.updated"),)
        if event.type == "speech_start" and event.utterance_id is not None:
            self._started_items = self._started_items | {event.utterance_id}
            return (speech_event("input_audio_buffer.speech_started", event.utterance_id),)
        if event.type == "error":
            error: Final[_ErrorEvent] = {
                **error_event(event.message or "Gnani transcription failed"),
                "event_id": new_event_id(),
            }
            return (error,)
        if event.type != "transcript":
            return ()
        if event.text is None:
            raise ValueError("Gnani returned a transcript without text")
        item: Final = event.utterance_id or event.segment_id or new_event_id()
        stopped: Final = (
            (speech_event("input_audio_buffer.speech_stopped", item),) if item in self._started_items else ()
        )
        self._started_items = self._started_items - {item}
        committed: Final[OpenAIRealtimeInputAudioBufferCommitted] = {
            "type": "input_audio_buffer.committed",
            "event_id": new_event_id(),
            "item_id": item,
            "previous_item_id": self._previous_item,
        }
        self._previous_item = item
        spoken: Final = None if event.audio_duration_ms is None else event.audio_duration_ms / 1000
        transcript: Final = completed_event(item, event.text, self._take_usage(spoken))
        if event.detected_language_code is None:
            return *stopped, committed, transcript
        detected: Final[_Transcript] = {**transcript, "detected_language_code": event.detected_language_code}
        return *stopped, committed, detected

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: object,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        events: Final = self._events(_ProviderEvent.model_validate_json(message), model)
        return {**realtime_response_transform_input, "response": list(events)}

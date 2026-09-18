from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.audio_utils.utils import normalize_transcription_language_to_bcp47
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transcription_protocol import (
    RealtimeTranscriptionProtocolError,
    TranscriptionAudioFormat,
    TranscriptionSessionUpdate,
    completed_event,
    decode_pcm16_append,
    delta_event,
    duration_usage,
    json_object,
    parse_transcription_session_update,
    speech_event,
    transcription_session,
    transcription_session_created_event,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig, RealtimeBackend
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    AUTO_LANGUAGE_CODE,
    DEFAULT_SPEECH_TO_TEXT_LOCATION,
    speech_to_text_host,
    validate_vertex_transcription_location,
    validate_vertex_transcription_project_id,
)
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeTranscriptionSession,
    OpenAIRealtimeTranscriptionSessionCreated,
)
from litellm.types.llms.vertex_ai_speech_to_text import (
    VertexSpeechStreamingConfigure,
    VertexSpeechStreamingConfigured,
    VertexSpeechStreamingDiscardTurn,
    VertexSpeechStreamingEvent,
    VertexSpeechStreamingEventUnion,
    VertexSpeechStreamingFinishTurn,
    VertexSpeechStreamingResponse,
    VertexSpeechStreamingTurnDiscarded,
    VertexSpeechStreamingTurnFinished,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

DEFAULT_SAMPLE_RATE_HERTZ: Final = 24_000
MIN_SAMPLE_RATE_HERTZ: Final = 8_000
MAX_SAMPLE_RATE_HERTZ: Final = 48_000
MAX_AUDIO_MESSAGE_BYTES: Final = 25_000
_SPEECH_TO_TEXT_ENDPOINTS: Final = frozenset({"/v1/audio/transcriptions", "/v1/realtime"})
_VERTEX_MODEL_PREFIX: Final = "vertex_ai/"
_STREAMING_EVENT_ADAPTER: Final = TypeAdapter[VertexSpeechStreamingEventUnion](VertexSpeechStreamingEvent)
_FINISH_TURN_COMMAND: Final = VertexSpeechStreamingFinishTurn().model_dump_json()
_DISCARD_TURN_COMMAND: Final = VertexSpeechStreamingDiscardTurn().model_dump_json()


class ChirpProtocolError(RealtimeTranscriptionProtocolError):
    pass


@dataclass(frozen=True, slots=True)
class SpeechStreamingTarget:
    api_endpoint: str
    recognizer: str
    access_token: str


@dataclass(frozen=True, slots=True)
class ChirpSessionConfig:
    model: str
    language: str | None
    sample_rate: int
    server_vad: bool

    def openai_session(self, session_id: str) -> OpenAIRealtimeTranscriptionSession:
        return transcription_session(
            session_id=session_id,
            model=self.model,
            sample_rate=self.sample_rate,
            language=self.language,
            server_vad=self.server_vad,
        )

    def configure_command(self) -> str:
        return VertexSpeechStreamingConfigure(
            model=self.model,
            language_codes=(AUTO_LANGUAGE_CODE,) if self.language is None else (self.language,),
            sample_rate_hertz=self.sample_rate,
        ).model_dump_json()


def is_vertex_speech_to_text_model(model: str) -> bool:
    try:
        info: Final = litellm.get_model_info(
            model=normalize_speech_to_text_model(model), custom_llm_provider="vertex_ai"
        )
    except Exception:  # noqa: BLE001  # get_model_info raises for unmapped models, which are not Speech-to-Text models
        return False
    if info.get("mode") != "audio_transcription":
        return False
    return _SPEECH_TO_TEXT_ENDPOINTS <= frozenset(info.get("supported_endpoints") or ())


def normalize_speech_to_text_model(model: str) -> str:
    return model.removeprefix(_VERTEX_MODEL_PREFIX)


def default_session_config(model: str) -> ChirpSessionConfig:
    return ChirpSessionConfig(
        model=normalize_speech_to_text_model(model),
        language=None,
        sample_rate=DEFAULT_SAMPLE_RATE_HERTZ,
        server_vad=True,
    )


def parse_chirp_session_update(payload: str, expected_model: str) -> ChirpSessionConfig:
    update: Final = parse_transcription_session_update(payload, ChirpProtocolError)
    if update.session_type not in (None, "transcription", "realtime"):
        raise ChirpProtocolError("Speech-to-Text streaming supports transcription sessions only")
    if update.unsupported_transcription_keys:
        verbose_logger.debug(
            "Speech-to-Text streaming: ignoring unsupported transcription settings %s",
            update.unsupported_transcription_keys,
        )
    model: Final = normalize_speech_to_text_model(expected_model)
    if update.model is not None and normalize_speech_to_text_model(update.model) != model:
        raise ChirpProtocolError("realtime session model cannot be changed")
    return ChirpSessionConfig(
        model=model,
        language=None if update.language is None else normalize_transcription_language_to_bcp47(update.language),
        sample_rate=_parse_sample_rate(update.audio_format),
        server_vad=_parse_server_vad(update),
    )


def _parse_sample_rate(audio_format: TranscriptionAudioFormat | None) -> int:
    if audio_format is None:
        return DEFAULT_SAMPLE_RATE_HERTZ
    if not audio_format.is_pcm16:
        raise ChirpProtocolError("Speech-to-Text streaming requires pcm16 input audio")
    if audio_format.channels not in (None, 1):
        raise ChirpProtocolError("Speech-to-Text streaming requires mono input audio")
    rate: Final = DEFAULT_SAMPLE_RATE_HERTZ if audio_format.rate is None else audio_format.rate
    if not MIN_SAMPLE_RATE_HERTZ <= rate <= MAX_SAMPLE_RATE_HERTZ:
        raise ChirpProtocolError(
            f"Speech-to-Text streaming supports sample rates from {MIN_SAMPLE_RATE_HERTZ} Hz"
            f" to {MAX_SAMPLE_RATE_HERTZ} Hz"
        )
    return rate


def _parse_server_vad(update: TranscriptionSessionUpdate) -> bool:
    if update.turn_detection_disabled:
        return False
    if update.turn_detection_type not in (None, "server_vad"):
        raise ChirpProtocolError("Speech-to-Text streaming supports server_vad turn detection or null")
    return True


def session_created_event(config: ChirpSessionConfig, session_id: str) -> OpenAIRealtimeTranscriptionSessionCreated:
    return transcription_session_created_event(config.openai_session(session_id))


def _normalize_word(word: str) -> str:
    return "".join(char for char in word if char.isalnum()).casefold()


def new_words(previous: str, current: str) -> str:
    previous_words: Final = previous.split()
    current_words: Final = current.split()
    common: Final = next(
        (
            index
            for index, (old, new) in enumerate(zip(previous_words, current_words, strict=False))
            if _normalize_word(old) != _normalize_word(new)
        ),
        min(len(previous_words), len(current_words)),
    )
    appended: Final = " ".join(current_words[common:])
    if not appended:
        return ""
    return f" {appended}" if common else appended


def _join_transcript(committed: str, tail: str) -> str:
    return " ".join(part for part in (committed, tail) if part)


@dataclass(frozen=True, slots=True)
class _Turn:
    item_id: str
    committed: str = ""
    preview: str = ""
    started_emitted: bool = False
    stopped_emitted: bool = False


class ChirpEventTransformer:
    def __init__(self, *, new_item_id: Callable[[], str] = lambda: f"item_{uuid.uuid4().hex}") -> None:
        self._new_item_id: Final = new_item_id
        self._config: ChirpSessionConfig | None = None
        self._session_id: str | None = None
        self._turn: _Turn | None = None
        self._billed_seconds: float = 0.0
        self._reported_seconds: float = 0.0

    def configure(self, config: ChirpSessionConfig, session_id: str) -> None:
        self._config = config
        self._session_id = session_id

    def take_unbilled_usage(self) -> RealtimeInputAudioTranscriptionUsage | None:
        unreported: Final = self._billed_seconds - self._reported_seconds
        if unreported <= 0:
            return None
        self._reported_seconds = self._billed_seconds
        return duration_usage(unreported)

    def transform(self, frame: VertexSpeechStreamingEventUnion) -> tuple[OpenAIRealtimeEvents, ...]:
        match frame:
            case VertexSpeechStreamingConfigured():
                return (session_created_event(self._require_config(), self._require_session_id()),)
            case VertexSpeechStreamingResponse():
                return self._response(frame)
            case VertexSpeechStreamingTurnFinished():
                return self._finish_turn()
            case VertexSpeechStreamingTurnDiscarded():
                self._turn = None
                return ()
            case _:
                assert_never(frame)

    def _response(self, frame: VertexSpeechStreamingResponse) -> tuple[OpenAIRealtimeEvents, ...]:
        self._billed_seconds = max(self._billed_seconds, frame.billed_seconds)
        interim: Final = " ".join(
            result.transcript.strip() for result in frame.results if not result.is_final and result.transcript.strip()
        )
        finals: Final = tuple(
            result.transcript.strip() for result in frame.results if result.is_final and result.transcript.strip()
        )
        begin_events: Final = self._begin() if frame.speech_event == "begin" or interim or finals else ()
        interim_events: Final = self._hypothesis(interim) if interim else ()
        final_events: Final = tuple(event for final in finals for event in self._final(final))
        end_events: Final = self._stop() if frame.speech_event == "end" else ()
        return (*begin_events, *interim_events, *final_events, *end_events)

    def _begin(self) -> tuple[OpenAIRealtimeEvents, ...]:
        if self._turn is None:
            self._turn = _Turn(item_id=self._new_item_id())
        turn: Final = self._turn
        if turn.started_emitted or not self._require_config().server_vad:
            return ()
        self._turn = replace(turn, started_emitted=True)
        return (speech_event("input_audio_buffer.speech_started", turn.item_id),)

    def _stop(self) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._turn
        if turn is None or turn.stopped_emitted or not self._require_config().server_vad:
            return ()
        self._turn = replace(turn, stopped_emitted=True)
        return (speech_event("input_audio_buffer.speech_stopped", turn.item_id),)

    def _hypothesis(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._require_turn()
        hypothesis: Final = _join_transcript(turn.committed, text)
        delta: Final = new_words(turn.preview, hypothesis)
        self._turn = replace(turn, preview=hypothesis)
        return (delta_event(turn.item_id, delta),) if delta else ()

    def _final(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._require_turn()
        committed: Final = _join_transcript(turn.committed, text)
        delta: Final = new_words(turn.preview, committed)
        self._turn = replace(turn, committed=committed, preview=committed)
        delta_events: Final[tuple[OpenAIRealtimeEvents, ...]] = (delta_event(turn.item_id, delta),) if delta else ()
        if not self._require_config().server_vad:
            return delta_events
        return (*delta_events, *self._complete())

    def _finish_turn(self) -> tuple[OpenAIRealtimeEvents, ...]:
        if self._turn is None:
            return ()
        return self._complete()

    def _complete(self) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._require_turn()
        stop_events: Final = self._stop()
        transcript: Final = turn.committed or turn.preview
        self._turn = None
        return (*stop_events, completed_event(turn.item_id, transcript, self.take_unbilled_usage()))

    def _require_turn(self) -> _Turn:
        if self._turn is None:
            self._turn = _Turn(item_id=self._new_item_id())
        return self._turn

    def _require_config(self) -> ChirpSessionConfig:
        if self._config is None:
            raise ChirpProtocolError("session.update must configure the session before the backend responds")
        return self._config

    def _require_session_id(self) -> str:
        if self._session_id is None:
            raise ChirpProtocolError("session.update must configure the session before the backend responds")
        return self._session_id


def _default_backend_factory(target: SpeechStreamingTarget) -> RealtimeBackend:
    from litellm.llms.vertex_ai.audio_transcription.realtime_backend import SpeechStreamingBackend

    return SpeechStreamingBackend(target)


class VertexChirpRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        access_token: str,
        project: str,
        location: str | None,
        backend_factory: Callable[[SpeechStreamingTarget], RealtimeBackend] = _default_backend_factory,
    ) -> None:
        self._access_token: Final = access_token
        self._project: Final = validate_vertex_transcription_project_id(project)
        self._location: Final = validate_vertex_transcription_location(location, DEFAULT_SPEECH_TO_TEXT_LOCATION)
        self._backend_factory: Final = backend_factory
        self._transformer: Final = ChirpEventTransformer()
        self._config: ChirpSessionConfig | None = None
        self._session_id: str | None = None

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseRealtimeConfig contract
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig contract
        return headers

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        if not is_vertex_speech_to_text_model(model):
            raise ValueError(f"Unsupported Speech-to-Text streaming model: {model}")
        return _api_endpoint(api_base) if api_base else speech_to_text_host(self._location)

    async def open_backend(self, url: str, headers: Mapping[str, str]) -> RealtimeBackend | None:
        return self._backend_factory(
            SpeechStreamingTarget(
                api_endpoint=url,
                recognizer=f"projects/{self._project}/locations/{self._location}/recognizers/_",
                access_token=self._access_token,
            )
        )

    def is_setup_message(self, msg_obj: Mapping[str, object]) -> bool:
        return msg_obj.get("kind") == "configure"

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> OpenAIRealtimeTranscriptionSessionCreated:
        self._session_id = logging_session_id
        return session_created_event(default_session_config(model), logging_session_id)

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> tuple[str | bytes, ...]:
        request: Final = json_object(message, ChirpProtocolError)
        event_type: Final = request.get("type")
        if event_type in ("session.update", "transcription_session.update"):
            return self._configure(message, model)
        if event_type == "input_audio_buffer.append":
            return self._append_audio(request)
        if event_type in ("input_audio_buffer.commit", "input_audio_buffer.end"):
            self._require_config()
            return (_FINISH_TURN_COMMAND,)
        if event_type == "input_audio_buffer.clear":
            self._require_config()
            return (_DISCARD_TURN_COMMAND,)
        verbose_logger.debug("Speech-to-Text streaming: dropping unsupported client event %s", event_type)
        return ()

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        return self._transformer.take_unbilled_usage()

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        frame: Final = _STREAMING_EVENT_ADAPTER.validate_json(message)
        events: Final = list(self._transformer.transform(frame))  # mutable-ok: response field is a list
        result: Final[RealtimeResponseTypedDict] = {
            "response": events,
            "current_output_item_id": realtime_response_transform_input.get("current_output_item_id"),
            "current_response_id": realtime_response_transform_input.get("current_response_id"),
            "current_delta_chunks": realtime_response_transform_input.get("current_delta_chunks"),
            "current_conversation_id": realtime_response_transform_input.get("current_conversation_id"),
            "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
            "current_delta_type": realtime_response_transform_input.get("current_delta_type"),
            "session_configuration_request": realtime_response_transform_input.get("session_configuration_request"),
        }
        return result

    def _configure(self, message: str, model: str) -> tuple[str, ...]:
        if self._config is not None:
            verbose_logger.debug("Speech-to-Text streaming: ignoring session.update after the stream was configured")
            return ()
        config: Final = parse_chirp_session_update(message, model)
        self._config = config
        self._transformer.configure(config, self._session_id or f"sess_{uuid.uuid4().hex}")
        return (config.configure_command(),)

    def _append_audio(self, request: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
        self._require_config()
        audio: Final = decode_pcm16_append(request.get("audio"), error=ChirpProtocolError)
        return tuple(
            audio[start : start + MAX_AUDIO_MESSAGE_BYTES] for start in range(0, len(audio), MAX_AUDIO_MESSAGE_BYTES)
        )

    def _require_config(self) -> ChirpSessionConfig:
        if self._config is None:
            raise ChirpProtocolError("session.update must configure the session before audio is sent")
        return self._config


def _api_endpoint(api_base: str) -> str:
    without_scheme: Final = api_base.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0]

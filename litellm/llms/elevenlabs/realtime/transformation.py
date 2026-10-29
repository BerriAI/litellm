import base64
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, get_args
from urllib.parse import urlencode

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transcription_protocol import (
    RealtimeTranscriptionProtocolError,
    TranscriptionAudioFormat,
    build_realtime_wss_url,
    completed_event,
    decode_pcm16_append,
    delta_event,
    duration_usage,
    error_event,
    json_object,
    parse_transcription_session_update,
    speech_event,
    transcription_session,
    transcription_session_created_event,
    uses_server_vad,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig, RealtimeBackend
from litellm.llms.elevenlabs.realtime.backend import ElevenLabsRealtimeBackend
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.elevenlabs import (
    ElevenLabsCommitStrategy,
    ElevenLabsCommittedTranscript,
    ElevenLabsErrorMessage,
    ElevenLabsErrorType,
    ElevenLabsInputAudioChunk,
    ElevenLabsPartialTranscript,
    ElevenLabsRealtimeConfigure,
    ElevenLabsRealtimeOptions,
    ElevenLabsServerMessage,
    ElevenLabsServerMessageUnion,
    ElevenLabsSessionStarted,
    ElevenLabsWarning,
)
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeTranscriptionSession,
    OpenAIRealtimeTranscriptionSessionCreated,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

SCRIBE_REALTIME_MODEL: Final = "scribe_v2_realtime"
DEFAULT_ELEVENLABS_REALTIME_URL: Final = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MAX_AUDIO_CHUNK_BYTES: Final = 32_000
_REALTIME_PATH: Final = "/v1/speech-to-text/realtime"
SUPPORTED_SAMPLE_RATES: Final = frozenset((8_000, 16_000, 22_050, 24_000, 44_100, 48_000))
_DEFAULT_SAMPLE_RATE: Final = 24_000
_BACKEND_NAME: Final = "ElevenLabs Scribe realtime"
_KNOWN_SERVER_MESSAGE_TYPES: Final = frozenset(
    ("session_started", "partial_transcript", "committed_transcript", "warning", *get_args(ElevenLabsErrorType))
)
_SERVER_MESSAGE_ADAPTER: Final = TypeAdapter[ElevenLabsServerMessageUnion](ElevenLabsServerMessage)


class ElevenLabsProtocolError(RealtimeTranscriptionProtocolError):
    pass


def _query_value(value: bool | float | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _optional_item(name: str, value: bool | float | None) -> Iterator[tuple[str, str]]:
    if value is not None:
        yield name, _query_value(value)


def _repeated_items(name: str, values: tuple[str, ...] | None) -> Iterator[tuple[str, str]]:
    yield from ((name, value) for value in values or ())


def _option_query_items(options: ElevenLabsRealtimeOptions) -> Iterator[tuple[str, str]]:
    yield from _repeated_items("keyterms", options.keyterms)
    yield from _repeated_items("secondary_languages", options.secondary_languages)
    yield from _optional_item("no_verbatim", options.no_verbatim)
    yield from _optional_item("filter_background_audio", options.filter_background_audio)
    yield from _optional_item("vad_threshold", options.vad_threshold)
    yield from _optional_item("vad_silence_threshold_secs", options.vad_silence_threshold_secs)
    yield from _optional_item("min_speech_duration_ms", options.min_speech_duration_ms)
    yield from _optional_item("min_silence_duration_ms", options.min_silence_duration_ms)
    yield from _optional_item("enable_logging", options.enable_logging)


@dataclass(frozen=True, slots=True)
class ElevenLabsSessionConfig:
    sample_rate: int
    commit_strategy: ElevenLabsCommitStrategy
    language: str | None
    options: ElevenLabsRealtimeOptions

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * 2

    def query(self) -> str:
        return urlencode(tuple(self._query_items()))

    def openai_session(self, session_id: str) -> OpenAIRealtimeTranscriptionSession:
        return transcription_session(
            session_id=session_id,
            model=SCRIBE_REALTIME_MODEL,
            sample_rate=self.sample_rate,
            language=self.language,
            server_vad=self.commit_strategy == "vad",
        )

    def _query_items(self) -> Iterator[tuple[str, str]]:
        yield "model_id", SCRIBE_REALTIME_MODEL
        yield "audio_format", f"pcm_{self.sample_rate}"
        yield "commit_strategy", self.commit_strategy
        if self.language is not None:
            yield "language_code", self.language
        yield from _option_query_items(self.options)


def default_session_config(options: ElevenLabsRealtimeOptions) -> ElevenLabsSessionConfig:
    return ElevenLabsSessionConfig(
        sample_rate=_DEFAULT_SAMPLE_RATE, commit_strategy="vad", language=None, options=options
    )


def realtime_options_from_params(params: Mapping[str, object]) -> ElevenLabsRealtimeOptions:
    provided: Final = {
        name: params[name] for name in ElevenLabsRealtimeOptions.model_fields if params.get(name) is not None
    }
    try:
        return ElevenLabsRealtimeOptions.model_validate(provided)
    except ValidationError as e:
        raise ValueError(f"Invalid ElevenLabs realtime options: {e}") from None


def normalize_model(model: str) -> str:
    return model.strip().removeprefix("elevenlabs/")


def normalize_language(language: str) -> str:
    primary: Final = language.strip().replace("_", "-").split("-", 1)[0].casefold()
    if not (2 <= len(primary) <= 3 and primary.isascii() and primary.isalpha()):
        raise ElevenLabsProtocolError("language must be an ISO 639-1 or ISO 639-3 code")
    return primary


def _parse_sample_rate(audio_format: TranscriptionAudioFormat | None) -> int:
    if audio_format is None:
        return _DEFAULT_SAMPLE_RATE
    if audio_format.layout == "beta":
        if audio_format.encoding != "pcm16":
            raise ElevenLabsProtocolError(f"{_BACKEND_NAME} requires pcm16 input audio")
        return _DEFAULT_SAMPLE_RATE
    if not audio_format.is_pcm16:
        raise ElevenLabsProtocolError(f"{_BACKEND_NAME} requires audio/pcm input audio")
    if audio_format.channels not in (None, 1):
        raise ElevenLabsProtocolError(f"{_BACKEND_NAME} requires mono input audio")
    rate: Final = _DEFAULT_SAMPLE_RATE if audio_format.rate is None else audio_format.rate
    if rate not in SUPPORTED_SAMPLE_RATES:
        raise ElevenLabsProtocolError(
            f"{_BACKEND_NAME} supports PCM16 at 8000, 16000, 22050, 24000, 44100, or 48000 Hz"
        )
    return rate


def parse_session_update(
    payload: str, options: ElevenLabsRealtimeOptions, *, drop_params: bool = False
) -> ElevenLabsSessionConfig:
    update: Final = parse_transcription_session_update(payload, ElevenLabsProtocolError)
    if update.session_type not in (None, "transcription", "realtime"):
        raise ElevenLabsProtocolError(f"{_BACKEND_NAME} supports transcription sessions only")
    if update.model is not None and normalize_model(update.model) != SCRIBE_REALTIME_MODEL:
        raise ElevenLabsProtocolError("realtime session model cannot be changed")
    if update.unsupported_transcription_keys:
        verbose_logger.warning(
            "ElevenLabs realtime: dropping unsupported transcription settings %s",
            update.unsupported_transcription_keys,
        )
    server_vad: Final = uses_server_vad(update, _BACKEND_NAME, drop_params, ElevenLabsProtocolError)
    return ElevenLabsSessionConfig(
        sample_rate=_parse_sample_rate(update.audio_format),
        commit_strategy="vad" if server_vad else "manual",
        language=None if update.language is None else normalize_language(update.language),
        options=options,
    )


def session_created_event(config: ElevenLabsSessionConfig, session_id: str) -> OpenAIRealtimeTranscriptionSessionCreated:
    return transcription_session_created_event(config.openai_session(session_id))


def parse_server_message(payload: str | bytes) -> ElevenLabsServerMessageUnion | None:
    try:
        text: Final = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError:
        raise ElevenLabsProtocolError("ElevenLabs sent a message that is not valid UTF-8") from None
    message: Final = json_object(text, ElevenLabsProtocolError)
    message_type: Final = message.get("message_type")
    if not isinstance(message_type, str) or message_type not in _KNOWN_SERVER_MESSAGE_TYPES:
        verbose_logger.debug("ElevenLabs realtime: dropping unsupported server message %s", message_type)
        return None
    try:
        return _SERVER_MESSAGE_ADAPTER.validate_python(message)
    except ValidationError:
        raise ElevenLabsProtocolError(f"ElevenLabs sent an invalid {message_type} message") from None


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class _Turn:
    item_id: str
    emitted: str


class ElevenLabsEventTransformer:
    def __init__(self, *, new_item_id: Callable[[], str] = _new_item_id) -> None:
        self._new_item_id: Final = new_item_id
        self._turn: _Turn | None = None
        self._unbilled_seconds: float = 0.0

    def record_audio(self, seconds: float) -> None:
        self._unbilled_seconds += seconds

    def take_unbilled_usage(self) -> RealtimeInputAudioTranscriptionUsage | None:
        seconds: Final = self._unbilled_seconds
        if seconds <= 0:
            return None
        self._unbilled_seconds = 0.0
        return duration_usage(seconds)

    def transform(
        self, message: ElevenLabsServerMessageUnion, config: ElevenLabsSessionConfig
    ) -> tuple[OpenAIRealtimeEvents, ...]:
        match message:
            case ElevenLabsSessionStarted(session_id=session_id):
                return (session_created_event(config, session_id),)
            case ElevenLabsPartialTranscript(text=text):
                return self._partial(text)
            case ElevenLabsCommittedTranscript(text=text):
                return self._committed(text)
            case ElevenLabsWarning(warning=warning):
                verbose_logger.warning("ElevenLabs realtime warning: %s", warning)
                return ()
            case ElevenLabsErrorMessage(message_type=error_type):
                return (error_event(f"ElevenLabs realtime transcription failed: {error_type}"),)
            case _:
                assert_never(message)

    def _partial(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        if not text:
            return ()
        turn: Final = self._turn
        if turn is None:
            started: Final = _Turn(item_id=self._new_item_id(), emitted=text)
            self._turn = started
            return (
                speech_event("input_audio_buffer.speech_started", started.item_id),
                delta_event(started.item_id, text),
            )
        if text == turn.emitted or not text.startswith(turn.emitted):
            return ()
        self._turn = _Turn(item_id=turn.item_id, emitted=text)
        return (delta_event(turn.item_id, text[len(turn.emitted) :]),)

    def _committed(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._turn
        self._turn = None
        if turn is None and not text:
            return ()
        item_id: Final = self._new_item_id() if turn is None else turn.item_id
        started: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (speech_event("input_audio_buffer.speech_started", item_id),) if turn is None else ()
        )
        return (
            *started,
            speech_event("input_audio_buffer.speech_stopped", item_id),
            completed_event(item_id, text, self.take_unbilled_usage()),
        )


def build_elevenlabs_realtime_url(api_base: str | None) -> str:
    return build_realtime_wss_url(
        api_base, default_url=DEFAULT_ELEVENLABS_REALTIME_URL, path=_REALTIME_PATH, provider="ElevenLabs"
    )


def _default_backend_factory(url: str, headers: Mapping[str, str]) -> RealtimeBackend:
    return ElevenLabsRealtimeBackend(url, headers)


class ElevenLabsRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        options: ElevenLabsRealtimeOptions | None = None,
        drop_params: bool = False,
        backend_factory: Callable[[str, Mapping[str, str]], RealtimeBackend] = _default_backend_factory,
        transformer: ElevenLabsEventTransformer | None = None,
    ) -> None:
        self._options: Final = options or ElevenLabsRealtimeOptions()
        self._drop_params: Final = drop_params
        self._backend_factory: Final = backend_factory
        self._transformer: Final = transformer or ElevenLabsEventTransformer()
        self._config: ElevenLabsSessionConfig | None = None

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseRealtimeConfig contract
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig contract
        key: Final = (api_key or get_secret_str("ELEVENLABS_API_KEY") or "").strip()
        if not key:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.")
        return {**headers, "xi-api-key": key}

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        if normalize_model(model) != SCRIBE_REALTIME_MODEL:
            raise ValueError(f"Unsupported ElevenLabs realtime model: {model}")
        return build_elevenlabs_realtime_url(api_base or get_secret_str("ELEVENLABS_API_BASE"))

    async def open_backend(self, url: str, headers: Mapping[str, str]) -> RealtimeBackend | None:
        return self._backend_factory(url, headers)

    def is_setup_message(self, msg_obj: Mapping[str, object]) -> bool:
        return msg_obj.get("kind") == "configure"

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> OpenAIRealtimeTranscriptionSessionCreated:
        return session_created_event(default_session_config(self._options), logging_session_id)

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> tuple[str, ...]:
        request: Final = json_object(message, ElevenLabsProtocolError)
        match request.get("type"):
            case "session.update" | "transcription_session.update":
                return self._configure(message)
            case "input_audio_buffer.append":
                return self._append_audio(request)
            case "input_audio_buffer.commit" | "input_audio_buffer.end":
                return (self._audio_chunk(b"", commit=True),)
            case event_type:
                verbose_logger.debug("ElevenLabs realtime: dropping unsupported client event %s", event_type)
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
        server_message: Final = parse_server_message(message)
        events: Final = (
            () if server_message is None else self._transformer.transform(server_message, self._require_config())
        )
        result: Final[RealtimeResponseTypedDict] = {
            "response": list(events),
            "current_output_item_id": realtime_response_transform_input.get("current_output_item_id"),
            "current_response_id": realtime_response_transform_input.get("current_response_id"),
            "current_delta_chunks": realtime_response_transform_input.get("current_delta_chunks"),
            "current_conversation_id": realtime_response_transform_input.get("current_conversation_id"),
            "current_item_chunks": realtime_response_transform_input.get("current_item_chunks"),
            "current_delta_type": realtime_response_transform_input.get("current_delta_type"),
            "session_configuration_request": realtime_response_transform_input.get("session_configuration_request"),
        }
        return result

    def _configure(self, message: str) -> tuple[str, ...]:
        if self._config is not None:
            verbose_logger.debug("ElevenLabs realtime: ignoring session.update after the session was configured")
            return ()
        config: Final = parse_session_update(message, self._options, drop_params=self._drop_params)
        self._config = config
        return (ElevenLabsRealtimeConfigure(query=config.query()).model_dump_json(),)

    def _append_audio(self, request: Mapping[str, JsonValue]) -> tuple[str, ...]:
        config: Final = self._require_config()
        audio: Final = decode_pcm16_append(request.get("audio"), error=ElevenLabsProtocolError)
        self._transformer.record_audio(len(audio) / config.bytes_per_second)
        return tuple(
            self._audio_chunk(audio[start : start + MAX_AUDIO_CHUNK_BYTES], commit=False)
            for start in range(0, len(audio), MAX_AUDIO_CHUNK_BYTES)
        )

    def _audio_chunk(self, audio: bytes, *, commit: bool) -> str:
        chunk: Final[ElevenLabsInputAudioChunk] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(audio).decode("ascii"),
            "commit": commit,
            "sample_rate": self._require_config().sample_rate,
        }
        return json.dumps(chunk, separators=(",", ":"))

    def _require_config(self) -> ElevenLabsSessionConfig:
        if self._config is None:
            raise ElevenLabsProtocolError("session.update must configure the ElevenLabs session before audio is sent")
        return self._config


def elevenlabs_realtime_config(params: Mapping[str, object], *, drop_params: bool) -> ElevenLabsRealtimeConfig:
    return ElevenLabsRealtimeConfig(options=realtime_options_from_params(params), drop_params=drop_params)

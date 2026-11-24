import asyncio
import base64
import json
import os
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, get_args
from urllib.parse import urlencode

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never
from websockets.exceptions import ConnectionClosed

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
from litellm.llms.elevenlabs.common_utils import SCRIBE_REALTIME_MODEL, ElevenLabsException, normalize_model
from litellm.llms.elevenlabs.realtime.backend import (
    OPEN_TIMEOUT_SECONDS,
    ElevenLabsRealtimeBackend,
    RealtimeConnect,
    open_realtime_websocket,
)
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

DEFAULT_ELEVENLABS_REALTIME_URL: Final = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MAX_AUDIO_CHUNK_BYTES: Final = 32_000
_REALTIME_PATH: Final = "/v1/speech-to-text/realtime"
SUPPORTED_SAMPLE_RATES: Final = frozenset((8_000, 16_000, 22_050, 24_000, 44_100, 48_000))
_DEFAULT_SAMPLE_RATE: Final = 24_000
_BACKEND_NAME: Final = "ElevenLabs Scribe realtime"
SCRIBE_REALTIME_KEYTERMS_PRICING_MODEL: Final = f"elevenlabs/{SCRIBE_REALTIME_MODEL}/keyterms"
_KNOWN_SERVER_MESSAGE_TYPES: Final = frozenset(
    ("session_started", "partial_transcript", "committed_transcript", "warning", *get_args(ElevenLabsErrorType))
)
_SERVER_MESSAGE_ADAPTER: Final = TypeAdapter[ElevenLabsServerMessageUnion](ElevenLabsServerMessage)
_ERROR_STATUS_CODES: Final[Mapping[ElevenLabsErrorType, int]] = MappingProxyType(
    {
        "auth_error": 401,
        "quota_exceeded": 429,
        "rate_limited": 429,
        "invalid_request": 400,
        "input_error": 400,
    }
)


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


@dataclass(frozen=True, slots=True)
class SessionUpdateRejected:
    """A session.update ElevenLabs can't serve. `session_config_or_raise` turns it into the client's error."""

    reason: str


def normalize_language(language: str) -> str | SessionUpdateRejected:
    primary: Final = language.strip().replace("_", "-").split("-", 1)[0].casefold()
    if not (2 <= len(primary) <= 3 and primary.isascii() and primary.isalpha()):
        return SessionUpdateRejected("language must be an ISO 639-1 or ISO 639-3 code")
    return primary


def _parse_sample_rate(audio_format: TranscriptionAudioFormat | None) -> int | SessionUpdateRejected:
    if audio_format is None:
        return _DEFAULT_SAMPLE_RATE
    if audio_format.layout == "beta":
        if audio_format.encoding != "pcm16":
            return SessionUpdateRejected(f"{_BACKEND_NAME} requires pcm16 input audio")
        return _DEFAULT_SAMPLE_RATE
    if not audio_format.is_pcm16:
        return SessionUpdateRejected(f"{_BACKEND_NAME} requires audio/pcm input audio")
    if audio_format.channels not in (None, 1):
        return SessionUpdateRejected(f"{_BACKEND_NAME} requires mono input audio")
    rate: Final = _DEFAULT_SAMPLE_RATE if audio_format.rate is None else audio_format.rate
    if rate not in SUPPORTED_SAMPLE_RATES:
        return SessionUpdateRejected(f"{_BACKEND_NAME} supports PCM16 at 8000, 16000, 22050, 24000, 44100, or 48000 Hz")
    return rate


def parse_session_update(
    payload: str, options: ElevenLabsRealtimeOptions, *, drop_params: bool = False
) -> ElevenLabsSessionConfig | SessionUpdateRejected:
    update: Final = parse_transcription_session_update(payload, ElevenLabsProtocolError)
    if update.session_type not in (None, "transcription", "realtime"):
        return SessionUpdateRejected(f"{_BACKEND_NAME} supports transcription sessions only")
    if update.model is not None and normalize_model(update.model) != SCRIBE_REALTIME_MODEL:
        return SessionUpdateRejected("realtime session model cannot be changed")
    if update.unsupported_transcription_keys:
        verbose_logger.warning(
            "ElevenLabs realtime: dropping unsupported transcription settings %s",
            update.unsupported_transcription_keys,
        )
    server_vad: Final = uses_server_vad(update, _BACKEND_NAME, drop_params, ElevenLabsProtocolError)
    sample_rate: Final = _parse_sample_rate(update.audio_format)
    if isinstance(sample_rate, SessionUpdateRejected):
        return sample_rate
    language: Final = None if update.language is None else normalize_language(update.language)
    if isinstance(language, SessionUpdateRejected):
        return language
    return ElevenLabsSessionConfig(
        sample_rate=sample_rate,
        commit_strategy="vad" if server_vad else "manual",
        language=language,
        options=options,
    )


def session_config_or_raise(outcome: ElevenLabsSessionConfig | SessionUpdateRejected) -> ElevenLabsSessionConfig:
    """The one place a rejected session.update becomes the protocol error the client receives."""
    match outcome:
        case ElevenLabsSessionConfig():
            return outcome
        case SessionUpdateRejected(reason=reason):
            raise ElevenLabsProtocolError(reason)
        case _:
            assert_never(outcome)


def session_created_event(
    config: ElevenLabsSessionConfig, session_id: str
) -> OpenAIRealtimeTranscriptionSessionCreated:
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


def elevenlabs_error(error_type: ElevenLabsErrorType) -> ElevenLabsException:
    return ElevenLabsException(
        status_code=_ERROR_STATUS_CODES.get(error_type, 500),
        message=f"ElevenLabs realtime transcription failed: {error_type}",
    )


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex}"


def _is_wide(text: str) -> bool:
    return any(unicodedata.east_asian_width(char) in ("W", "F") for char in text)


def _streams_by_word(text: str) -> bool:
    words: Final = text.split()
    spaced: Final = text.isascii() or len(words) > 1
    return spaced and not (words and _is_wide(words[-1]))


def _next_delta(emitted: str, stable: str) -> str:
    if stable.startswith(emitted) or not _streams_by_word(stable):
        return stable[len(emitted) :]
    new_words: Final = " ".join(stable.split()[len(emitted.split()) :])
    return f" {new_words}" if new_words and emitted else new_words


def _without_trailing_punctuation(text: str) -> str:
    end: Final = next(
        (index for index in range(len(text), 0, -1) if not unicodedata.category(text[index - 1]).startswith("P")), 0
    )
    return text[:end]


def _stable_text(previous: str | None, text: str) -> str:
    if _streams_by_word(text):
        return " ".join(text.split()[:-1])
    if previous is None:
        return ""
    return _without_trailing_punctuation(os.path.commonprefix((previous, text))[: len(text) - 1])


@dataclass(frozen=True, slots=True)
class _Turn:
    item_id: str
    emitted: str
    previous: str | None


class ElevenLabsEventTransformer:
    def __init__(self, *, new_item_id: Callable[[], str] = _new_item_id) -> None:
        self._new_item_id: Final = new_item_id
        self._turn: _Turn | None = None
        self._unbilled_seconds: float = 0.0
        self._session_started: bool = False
        self._refusal: ElevenLabsException | None = None
        self._uncommitted_audio: bool = False
        # One entry per commit forwarded upstream and not yet answered, oldest first: True when it was a clear
        self._pending_commits: tuple[bool, ...] = ()

    @property
    def refusal(self) -> ElevenLabsException | None:
        return self._refusal

    def record_audio(self, seconds: float) -> None:
        self._unbilled_seconds += seconds
        if seconds > 0:
            self._uncommitted_audio = True

    def commit_audio(self, *, discard: bool) -> bool:
        if not self._uncommitted_audio:
            return False
        self._uncommitted_audio = False
        awaits_kept_transcript: Final = False in self._pending_commits
        self._pending_commits = (*self._pending_commits, discard)
        if discard and not awaits_kept_transcript:
            self._turn = None
        return True

    def _answers_cleared_commit(self) -> bool:
        """Match a committed transcript to the oldest forwarded commit; True when that commit was a clear."""
        if not self._pending_commits:
            return False
        cleared: Final = self._pending_commits[0]
        self._pending_commits = self._pending_commits[1:]
        return cleared

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
                self._session_started = True
                return (session_created_event(config, session_id),)
            case ElevenLabsPartialTranscript(text=text):
                return self._partial(text)
            case ElevenLabsCommittedTranscript(text=text):
                return self._committed(text)
            case ElevenLabsWarning(warning=warning):
                verbose_logger.warning("ElevenLabs realtime warning: %s", warning)
                return ()
            case ElevenLabsErrorMessage(message_type=error_type):
                return (self._report(elevenlabs_error(error_type)),)
            case _:
                assert_never(message)

    def _report(self, error: ElevenLabsException) -> OpenAIRealtimeEvents:
        if not self._session_started and self._refusal is None:
            self._refusal = error
        return error_event(error.message, "invalid_request_error" if error.status_code < 500 else "server_error")

    def _partial(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        if not text or True in self._pending_commits:
            return ()
        turn: Final = self._turn
        current: Final = turn if turn is not None else _Turn(item_id=self._new_item_id(), emitted="", previous=None)
        started: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (speech_event("input_audio_buffer.speech_started", current.item_id),) if turn is None else ()
        )
        delta: Final = _next_delta(current.emitted, _stable_text(current.previous, text))
        self._turn = _Turn(item_id=current.item_id, emitted=current.emitted + delta, previous=text)
        return (*started, delta_event(current.item_id, delta)) if delta else started

    def _committed(self, text: str) -> tuple[OpenAIRealtimeEvents, ...]:
        turn: Final = self._turn
        self._turn = None
        if self._answers_cleared_commit():
            return ()
        if turn is None and not text:
            return ()
        item_id: Final = self._new_item_id() if turn is None else turn.item_id
        started: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (speech_event("input_audio_buffer.speech_started", item_id),) if turn is None else ()
        )
        tail: Final = _next_delta("" if turn is None else turn.emitted, text)
        return (
            *started,
            *((delta_event(item_id, tail),) if tail else ()),
            speech_event("input_audio_buffer.speech_stopped", item_id),
            completed_event(item_id, text, self.take_unbilled_usage()),
        )


def build_elevenlabs_realtime_url(api_base: str | None) -> str:
    return build_realtime_wss_url(
        api_base, default_url=DEFAULT_ELEVENLABS_REALTIME_URL, path=_REALTIME_PATH, provider="ElevenLabs"
    )


def _default_backend_factory(
    url: str, headers: Mapping[str, str], on_audio_sent: Callable[[float], None]
) -> RealtimeBackend:
    return ElevenLabsRealtimeBackend(url, headers, on_audio_sent=on_audio_sent)


class ElevenLabsRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        options: ElevenLabsRealtimeOptions | None = None,
        drop_params: bool = False,
        backend_factory: Callable[
            [str, Mapping[str, str], Callable[[float], None]], RealtimeBackend
        ] = _default_backend_factory,
        transformer: ElevenLabsEventTransformer | None = None,
    ) -> None:
        self._options: Final = options or ElevenLabsRealtimeOptions()
        self._drop_params: Final = drop_params
        self._backend_factory: Final = backend_factory
        self._transformer: Final = transformer or ElevenLabsEventTransformer()
        self._config: ElevenLabsSessionConfig | None = None
        self._session_id: str | None = None

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
        return self._backend_factory(url, headers, self._transformer.record_audio)

    def is_setup_message(self, msg_obj: Mapping[str, object]) -> bool:
        return msg_obj.get("kind") == "configure"

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> OpenAIRealtimeTranscriptionSessionCreated:
        self._session_id = logging_session_id
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
                return self._commit(discard=False)
            case "input_audio_buffer.clear":
                return self._commit(discard=True)
            case event_type:
                verbose_logger.debug("ElevenLabs realtime: dropping unsupported client event %s", event_type)
                return ()

    def session_refusal(self) -> Exception | None:
        return self._transformer.refusal

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        if self._transformer.refusal is not None:
            return None
        return self._transformer.take_unbilled_usage()

    def transcription_addon_pricing_models(self) -> tuple[str, ...]:
        return (SCRIBE_REALTIME_KEYTERMS_PRICING_MODEL,) if self._options.keyterms else ()

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        server_message: Final = parse_server_message(message)
        events: Final = () if server_message is None else self._server_events(server_message)
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

    def _server_events(self, message: ElevenLabsServerMessageUnion) -> tuple[OpenAIRealtimeEvents, ...]:
        if isinstance(message, ElevenLabsSessionStarted):
            verbose_logger.info("ElevenLabs realtime session started: %s", message.session_id)
        reported: Final = (
            message.model_copy(update={"session_id": self._session_id})
            if isinstance(message, ElevenLabsSessionStarted) and self._session_id is not None
            else message
        )
        return self._transformer.transform(reported, self._require_config())

    def _configure(self, message: str) -> tuple[str, ...]:
        if self._config is not None:
            verbose_logger.debug("ElevenLabs realtime: ignoring session.update after the session was configured")
            return ()
        config: Final = session_config_or_raise(
            parse_session_update(message, self._options, drop_params=self._drop_params)
        )
        self._config = config
        return (ElevenLabsRealtimeConfigure(query=config.query()).model_dump_json(),)

    def _commit(self, *, discard: bool) -> tuple[str, ...]:
        self._require_config()
        if not self._transformer.commit_audio(discard=discard):
            return ()
        return (self._audio_chunk(b"", commit=True),)

    def _append_audio(self, request: Mapping[str, JsonValue]) -> tuple[str, ...]:
        self._require_config()
        audio: Final = decode_pcm16_append(request.get("audio"), error=ElevenLabsProtocolError)
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


async def _next_frame(backend: RealtimeBackend) -> str | bytes:
    try:
        return await backend.recv()
    except ConnectionClosed:
        raise ElevenLabsException(
            status_code=502, message="ElevenLabs closed the realtime session before starting it"
        ) from None


async def _session_start_outcome(backend: RealtimeBackend) -> bool:
    while True:
        match parse_server_message(await _next_frame(backend)):
            case ElevenLabsSessionStarted():
                return True
            case ElevenLabsErrorMessage(message_type=error_type):
                raise elevenlabs_error(error_type)
            case _:
                continue


async def elevenlabs_realtime_health_check(
    model: str,
    api_base: str | None,
    api_key: str | None,
    options: ElevenLabsRealtimeOptions,
    *,
    connect: RealtimeConnect = open_realtime_websocket,
    start_timeout: float = OPEN_TIMEOUT_SECONDS,
) -> bool:
    config: Final = ElevenLabsRealtimeConfig()
    url: Final = config.get_complete_url(api_base, model)
    headers: Final = config.validate_environment({}, model, api_key)
    backend: Final = await connect(f"{url}?{default_session_config(options).query()}", headers)
    try:
        return await asyncio.wait_for(_session_start_outcome(backend), start_timeout)
    except asyncio.TimeoutError:
        raise ElevenLabsException(
            status_code=504, message="ElevenLabs did not start the realtime session in time"
        ) from None
    finally:
        await backend.close()

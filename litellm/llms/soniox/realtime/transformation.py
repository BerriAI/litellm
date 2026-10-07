import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType, TracebackType
from typing import Final
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from typing_extensions import Self

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.realtime.transcription_protocol import (
    SESSION_UPDATE_EVENT_TYPES,
    RealtimeTranscriptionProtocolError,
    TranscriptionSessionUpdate,
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
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig, RealtimeBackend
from litellm.llms.soniox.common_utils import get_soniox_api_key
from litellm.types.llms.openai import OpenAIRealtimeEvents, OpenAIRealtimeTranscriptionSessionCreated
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

DEFAULT_SONIOX_REALTIME_URL: Final = "wss://stt-rt.soniox.com/transcribe-websocket"
DEFAULT_SAMPLE_RATE: Final = 24_000
KEEPALIVE_INTERVAL_SECONDS: Final = 5.0
_MAX_PENDING_FRAMES: Final = 64
_FINALIZE: Final = '{"type":"finalize"}'
_KEEPALIVE: Final = '{"type":"keepalive"}'
_END_STREAM: Final = ""
_CONTROL_FRAMES: Final = frozenset((_FINALIZE, _KEEPALIVE, _END_STREAM))
_CLIENT_CONTROL_FRAMES: Final = MappingProxyType(
    {"input_audio_buffer.commit": _FINALIZE, "input_audio_buffer.end": _END_STREAM}
)
SESSION_STARTED_FRAME: Final = '{"type":"litellm.soniox.session_started"}'
_ENDPOINT_TOKEN: Final = "<end>"
_FINALIZED_TOKEN: Final = "<fin>"


class SonioxProtocolError(RealtimeTranscriptionProtocolError):
    pass


class SonioxRealtimeOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    language_hints: tuple[str, ...] | None = None
    language_hints_strict: bool | None = None
    enable_language_identification: bool | None = None
    enable_speaker_diarization: bool | None = None
    context: JsonValue | None = None
    translation: JsonValue | None = None
    client_reference_id: str | None = None
    max_endpoint_delay_ms: int | None = None
    endpoint_sensitivity: float | None = None
    endpoint_latency_adjustment_level: int | None = None


class SonioxStartRequest(SonioxRealtimeOptions):
    model: str
    audio_format: str
    sample_rate: int
    num_channels: int
    enable_endpoint_detection: bool


class _SonioxToken(BaseModel):
    text: str
    is_final: bool = False
    translation_status: str | None = None


class _SonioxResponse(BaseModel):
    tokens: tuple[_SonioxToken, ...] = ()
    total_audio_proc_ms: int | None = None
    error_code: int | None = None
    error_message: str | None = None


def _sample_rate(update: TranscriptionSessionUpdate | None) -> int:
    audio_format: Final = None if update is None else update.audio_format
    if audio_format is None:
        return DEFAULT_SAMPLE_RATE
    if not audio_format.is_pcm16:
        raise SonioxProtocolError("Soniox realtime requires pcm16 input audio")
    if audio_format.channels not in (None, 1):
        raise SonioxProtocolError("Soniox realtime requires mono input audio")
    if audio_format.rate is None:
        return DEFAULT_SAMPLE_RATE
    if audio_format.rate <= 0:
        raise SonioxProtocolError("sample rate must be positive")
    return audio_format.rate


def _language_hints(language: str | None, configured: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if language is None:
        return configured
    return (language, *(hint for hint in configured or () if hint != language))


def build_start_request(
    model: str, update: TranscriptionSessionUpdate | None, options: SonioxRealtimeOptions
) -> SonioxStartRequest:
    return SonioxStartRequest(
        **options.model_dump(exclude={"language_hints"}),
        language_hints=_language_hints(None if update is None else update.language, options.language_hints),
        model=model,
        audio_format="pcm_s16le",
        sample_rate=_sample_rate(update),
        num_channels=1,
        enable_endpoint_detection=update is None or not update.turn_detection_disabled,
    )


def _session_created(session_id: str, start_request: SonioxStartRequest) -> OpenAIRealtimeTranscriptionSessionCreated:
    session: Final = transcription_session(
        session_id=session_id,
        model=start_request.model,
        sample_rate=start_request.sample_rate,
        language=start_request.language_hints[0] if start_request.language_hints else None,
        server_vad=start_request.enable_endpoint_detection,
    )
    return transcription_session_created_event(session)


def _changes_started_rate(update: TranscriptionSessionUpdate, started: SonioxStartRequest) -> bool:
    audio_format: Final = update.audio_format
    if audio_format is None:
        return False
    requested_rate: Final = _sample_rate(update)
    return audio_format.rate is not None and requested_rate != started.sample_rate


def _changes_started_stream(update: TranscriptionSessionUpdate, started: SonioxStartRequest) -> bool:
    turn_detection_set: Final = update.turn_detection is not None or update.turn_detection_disabled
    return (
        _changes_started_rate(update, started)
        or (
            update.language is not None
            and _language_hints(update.language, started.language_hints) != started.language_hints
        )
        or (turn_detection_set and update.turn_detection_disabled == started.enable_endpoint_detection)
    )


def build_soniox_realtime_url(api_base: str | None) -> str:
    if api_base is None:
        return DEFAULT_SONIOX_REALTIME_URL
    if urlparse(api_base).scheme not in ("ws", "wss"):
        raise ValueError("Soniox realtime api_base must be a ws:// or wss:// URL")
    return api_base


class SonioxEventTransformer:
    def __init__(self, *, translated_only: bool) -> None:
        self._translated_only: Final = translated_only
        self._item_id: str | None = None
        self._final_text: str = ""
        self._processed_ms: int = 0
        self._billed_ms: int = 0

    def transform(self, payload: str) -> tuple[OpenAIRealtimeEvents, ...]:
        try:
            response: Final = _SonioxResponse.model_validate_json(payload)
        except ValidationError:
            raise SonioxProtocolError("Soniox returned an invalid realtime response") from None
        if response.error_code is not None or response.error_message is not None:
            return (error_event(f"Soniox realtime error {response.error_code}: {response.error_message}"),)
        self._bill(response.total_audio_proc_ms)
        return tuple(event for token in response.tokens for event in self._token_events(token))

    def take_unbilled_usage(self, stream_ms: int = 0) -> RealtimeInputAudioTranscriptionUsage | None:
        billable_ms: Final = max(self._processed_ms, stream_ms)
        unbilled_ms: Final = billable_ms - self._billed_ms
        if unbilled_ms <= 0:
            return None
        self._billed_ms = billable_ms
        return duration_usage(unbilled_ms / 1000)

    def _bill(self, total_audio_proc_ms: int | None) -> None:
        if total_audio_proc_ms is not None and total_audio_proc_ms > self._processed_ms:
            self._processed_ms = total_audio_proc_ms

    def _token_events(self, token: _SonioxToken) -> tuple[OpenAIRealtimeEvents, ...]:
        if token.text in (_ENDPOINT_TOKEN, _FINALIZED_TOKEN):
            return self._complete(speech_stopped=token.text == _ENDPOINT_TOKEN)
        if self._translated_only and token.translation_status != "translation":
            return ()
        started: Final = self._start()
        if not token.is_final:
            return started
        self._final_text += token.text
        return (*started, delta_event(self._current_item_id(), token.text))

    def _start(self) -> tuple[OpenAIRealtimeEvents, ...]:
        if self._item_id is not None:
            return ()
        item_id: Final = f"item_{uuid.uuid4().hex}"
        self._item_id = item_id
        return (speech_event("input_audio_buffer.speech_started", item_id),)

    def _current_item_id(self) -> str:
        assert self._item_id is not None
        return self._item_id

    def _complete(self, *, speech_stopped: bool) -> tuple[OpenAIRealtimeEvents, ...]:
        item_id: Final = self._item_id
        if item_id is None:
            return ()
        transcript: Final = self._final_text.strip()
        self._item_id = None
        self._final_text = ""
        stopped: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (speech_event("input_audio_buffer.speech_stopped", item_id),) if speech_stopped else ()
        )
        return (*stopped, completed_event(item_id, transcript, self.take_unbilled_usage()))


class SonioxRealtimeBackend:
    def __init__(
        self,
        inner: RealtimeBackend,
        *,
        keepalive_interval: float = KEEPALIVE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner: Final = inner
        self._interval: Final = keepalive_interval
        self._clock: Final = clock
        self._sleep: Final = sleep
        self._frames: Final[asyncio.Queue[str | bytes | Exception]] = asyncio.Queue(maxsize=_MAX_PENDING_FRAMES)
        self._last_sent: float = clock()
        self._started: bool = False
        self._ended: bool = False
        self._tasks: tuple[asyncio.Task[None], ...] = ()

    async def __aenter__(self) -> Self:
        await self._inner.__aenter__()
        self._tasks = (asyncio.create_task(self._pump()), asyncio.create_task(self._keepalive()))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop()
        await self._inner.__aexit__(exc_type, exc_value, traceback)

    async def send(self, message: str | bytes) -> None:
        self._last_sent = self._clock()
        self._ended = self._ended or message == _END_STREAM
        await self._inner.send(message)
        if not self._started and isinstance(message, str) and message not in _CONTROL_FRAMES:
            self._started = True
            await self._frames.put(SESSION_STARTED_FRAME)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        frame: Final = await self._frames.get()
        if isinstance(frame, Exception):
            raise frame
        return frame

    async def close(self) -> None:
        self._stop()
        await self._inner.close()

    def _stop(self) -> None:
        for task in self._tasks:
            task.cancel()

    async def _pump(self) -> None:
        while True:
            try:
                frame = await self._inner.recv()
            except Exception as e:
                await self._frames.put(e)
                return
            await self._frames.put(frame)

    async def _keepalive(self) -> None:
        while not self._ended:
            await self._sleep(self._interval)
            if self._ended or self._clock() - self._last_sent < self._interval:
                continue
            try:
                await self.send(_KEEPALIVE)
            except Exception as e:
                verbose_logger.debug("Soniox realtime: stopping keepalive after send failed: %s", e)
                return


class SonioxRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        options: SonioxRealtimeOptions | None = None,
        wrap: Callable[[RealtimeBackend], RealtimeBackend] = SonioxRealtimeBackend,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._options: Final = options or SonioxRealtimeOptions()
        self._wrap: Final = wrap
        self._clock: Final = clock
        self._transformer: Final = SonioxEventTransformer(translated_only=self._options.translation is not None)
        self._start_request: SonioxStartRequest | None = None
        self._session_id: str = f"sess_{uuid.uuid4().hex}"
        self._opened_at: float | None = None

    @classmethod
    def from_litellm_params(cls, litellm_params: Mapping[str, object]) -> Self:
        return cls(options=SonioxRealtimeOptions.model_validate(dict(litellm_params)))

    def requires_session_configuration(self) -> bool:
        return True

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseRealtimeConfig contract
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig contract
        resolved_key: Final = get_soniox_api_key(api_key)
        if not resolved_key:
            raise ValueError("Missing Soniox API key. Set SONIOX_API_KEY or api_key in the deployment litellm_params")
        return {**headers, "Authorization": f"Bearer {resolved_key}"}

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        return build_soniox_realtime_url(api_base)

    def wrap_backend(self, backend: RealtimeBackend) -> RealtimeBackend:
        self._opened_at = self._clock()
        return self._wrap(backend)

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> OpenAIRealtimeTranscriptionSessionCreated:
        self._session_id = logging_session_id
        return _session_created(logging_session_id, build_start_request(model, None, self._options))

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> tuple[str | bytes, ...]:
        request: Final = json_object(message, SonioxProtocolError)
        event_type: Final = request.get("type")
        if event_type in SESSION_UPDATE_EVENT_TYPES:
            return self._start(model, parse_transcription_session_update(message, SonioxProtocolError))
        if event_type == "input_audio_buffer.append":
            audio: Final = decode_pcm16_append(request.get("audio"), error=SonioxProtocolError)
            return (*self._start(model, None), audio)
        control_frame: Final = _CLIENT_CONTROL_FRAMES.get(event_type) if isinstance(event_type, str) else None
        if control_frame is None:
            verbose_logger.debug("Soniox realtime: dropping unsupported client event %s", event_type)
            return ()
        return (*self._start(model, None), control_frame)

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        opened_at: Final = self._opened_at
        if opened_at is None or self._start_request is None:
            return self._transformer.take_unbilled_usage()
        return self._transformer.take_unbilled_usage(int((self._clock() - opened_at) * 1000))

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        payload: Final = message.decode("utf-8") if isinstance(message, bytes) else message
        result: Final[RealtimeResponseTypedDict] = {
            "response": list(self._backend_events(payload)),
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
        if payload != SESSION_STARTED_FRAME:
            return self._transformer.transform(payload)
        assert self._start_request is not None
        return (_session_created(self._session_id, self._start_request),)

    def _start(self, model: str, update: TranscriptionSessionUpdate | None) -> tuple[str, ...]:
        started: Final = self._start_request
        if started is not None:
            if update is not None and _changes_started_stream(update, started):
                raise SonioxProtocolError("Soniox realtime can't change the session once the stream has started")
            return ()
        start_request: Final = build_start_request(model, update, self._options)
        self._start_request = start_request
        return (start_request.model_dump_json(exclude_none=True),)

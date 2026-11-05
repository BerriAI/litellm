import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from itertools import chain, groupby
from types import MappingProxyType, TracebackType
from typing import Final
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from typing_extensions import NotRequired, ReadOnly, Required, Self, TypedDict

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
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
    uses_server_vad,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig, RealtimeBackend
from litellm.llms.custom_httpx.http_handler import realtime_ssl_for_url
from litellm.llms.soniox.common_utils import SonioxException, get_soniox_api_key
from litellm.types.llms.openai import (
    OpenAIRealtimeErrorEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeInputAudioTranscriptionCompleted,
    OpenAIRealtimeInputAudioTranscriptionDelta,
    OpenAIRealtimeTranscriptionSessionCreated,
)
from litellm.types.realtime import (
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

DEFAULT_SONIOX_REALTIME_URL: Final = "wss://stt-rt.soniox.com/transcribe-websocket"
WEBSOCKET_SCHEMES: Final = MappingProxyType({"wss": "wss", "ws": "ws", "https": "wss", "http": "ws"})
DEFAULT_SAMPLE_RATE: Final = 24_000
KEEPALIVE_INTERVAL_SECONDS: Final = 5.0
_MAX_PENDING_FRAMES: Final = 64
_FINALIZE: Final = '{"type":"finalize"}'
_KEEPALIVE: Final = '{"type":"keepalive"}'
_END_STREAM: Final = ""
_HEALTH_CHECK_SILENCE: Final = bytes(DEFAULT_SAMPLE_RATE // 5)
_CONTROL_FRAMES: Final = frozenset((_FINALIZE, _KEEPALIVE, _END_STREAM))
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


class SonioxSpeechLabels(TypedDict, total=False):
    speaker: ReadOnly[str]
    language: ReadOnly[str]


class SonioxTranscriptSegment(SonioxSpeechLabels):
    text: Required[ReadOnly[str]]


class SonioxTranscriptionDelta(OpenAIRealtimeInputAudioTranscriptionDelta, SonioxSpeechLabels):
    pass


class SonioxTranscriptionCompleted(OpenAIRealtimeInputAudioTranscriptionCompleted):
    segments: NotRequired[ReadOnly[tuple[SonioxTranscriptSegment, ...]]]


class _SonioxToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    is_final: bool = False
    translation_status: str | None = None
    speaker: str | None = None
    language: str | None = None

    @property
    def labelled(self) -> bool:
        return self.speaker is not None or self.language is not None


class _SonioxResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens: tuple[_SonioxToken, ...] = ()
    total_audio_proc_ms: int | None = None
    error_code: int | None = None
    error_message: str | None = None
    finished: bool = False


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


def _language_hints(
    language: str | None, configured: tuple[str, ...] | None, strict: bool | None
) -> tuple[str, ...] | None:
    if not language or (strict and configured):
        return configured
    return (language, *(hint for hint in configured or () if hint != language))


def _uses_endpoint_detection(update: TranscriptionSessionUpdate, drop_params: bool) -> bool:
    return uses_server_vad(update, "Soniox realtime", drop_params, SonioxProtocolError)


def _ignored_settings(update: TranscriptionSessionUpdate) -> tuple[str, ...]:
    vad_tuning: Final = (f"turn_detection.{key}" for key in sorted(update.turn_detection or {}) if key != "type")
    return (*update.unsupported_transcription_keys, *vad_tuning)


def build_start_request(
    model: str, update: TranscriptionSessionUpdate | None, options: SonioxRealtimeOptions, drop_params: bool = False
) -> SonioxStartRequest:
    return SonioxStartRequest(
        **options.model_dump(exclude={"language_hints"}),
        language_hints=_language_hints(
            None if update is None else update.language, options.language_hints, options.language_hints_strict
        ),
        model=model,
        audio_format="pcm_s16le",
        sample_rate=_sample_rate(update),
        num_channels=1,
        enable_endpoint_detection=update is None or _uses_endpoint_detection(update, drop_params),
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


def _changes_started_stream(update: TranscriptionSessionUpdate, started: SonioxStartRequest, drop_params: bool) -> bool:
    turn_detection_set: Final = update.turn_detection is not None or update.turn_detection_disabled
    return (
        _changes_started_rate(update, started)
        or (
            update.language is not None
            and _language_hints(update.language, started.language_hints, started.language_hints_strict)
            != started.language_hints
        )
        or (turn_detection_set and _uses_endpoint_detection(update, drop_params) != started.enable_endpoint_detection)
    )


def build_soniox_realtime_url(api_base: str | None) -> str:
    if api_base is None:
        return DEFAULT_SONIOX_REALTIME_URL
    parsed: Final = urlparse(api_base)
    scheme: Final = WEBSOCKET_SCHEMES.get(parsed.scheme)
    if scheme is None:
        raise ValueError("Soniox realtime api_base must be a wss://, ws://, https://, or http:// URL")
    return urlunparse(parsed._replace(scheme=scheme))


def _labels(token: _SonioxToken) -> SonioxSpeechLabels:
    speaker: Final[SonioxSpeechLabels] = {} if token.speaker is None else {"speaker": token.speaker}
    language: Final[SonioxSpeechLabels] = {} if token.language is None else {"language": token.language}
    return {**speaker, **language}


def _delta(item_id: str, token: _SonioxToken) -> SonioxTranscriptionDelta:
    return {**delta_event(item_id, token.text), **_labels(token)}


def _segment(run: tuple[_SonioxToken, ...]) -> SonioxTranscriptSegment:
    return {**_labels(run[0]), "text": "".join(token.text for token in run).strip()}


def _segments(tokens: tuple[_SonioxToken, ...]) -> tuple[SonioxTranscriptSegment, ...]:
    return tuple(_segment(tuple(run)) for _, run in groupby(tokens, key=lambda token: (token.speaker, token.language)))


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex}"


def _completed(
    item_id: str, tokens: tuple[_SonioxToken, ...], usage: RealtimeInputAudioTranscriptionUsage | None
) -> OpenAIRealtimeInputAudioTranscriptionCompleted:
    completed: Final = completed_event(item_id, "".join(token.text for token in tokens).strip(), usage)
    if not any(token.labelled for token in tokens):
        return completed
    labelled: Final[SonioxTranscriptionCompleted] = {**completed, "segments": _segments(tokens)}
    return labelled


def _soniox_error(response: _SonioxResponse) -> SonioxException | None:
    if response.error_code is None and response.error_message is None:
        return None
    return SonioxException(
        status_code=response.error_code or 500,
        message=f"Soniox realtime error {response.error_code}: {response.error_message}",
    )


class SonioxEventTransformer:
    def __init__(self, *, drops_originals: bool) -> None:
        self._drops_originals: Final = drops_originals
        self._item_id: str | None = None
        self._final_tokens: Final[list[_SonioxToken]] = []  # mutable-ok: a turn's tokens append in O(1) each
        self._finalizes: Final[deque[bool]] = deque()  # mutable-ok: pending finalizes queue and pop in O(1) each
        self._oldest_commit_answered: bool = False
        self._processed_ms: int = 0
        self._billed_ms: int = 0
        self._refusal: SonioxException | None = None

    @property
    def refusal(self) -> SonioxException | None:
        return self._refusal

    def transform(self, payload: str) -> tuple[OpenAIRealtimeEvents, ...]:
        try:
            response: Final = _SonioxResponse.model_validate_json(payload)
        except ValidationError:
            raise SonioxProtocolError("Soniox returned an invalid realtime response") from None
        self._bill(response.total_audio_proc_ms)
        error: Final = _soniox_error(response)
        if error is not None:
            return (self._report(error),)
        return tuple(chain.from_iterable(self._token_events(token) for token in response.tokens))

    def queue_finalize(self, *, discards_turn: bool) -> None:
        if discards_turn and not self._finalizes:
            self._item_id = None
            self._final_tokens.clear()
        self._finalizes.append(discards_turn)

    def take_unbilled_usage(self, stream_ms: int = 0) -> RealtimeInputAudioTranscriptionUsage | None:
        billable_ms: Final = max(self._processed_ms, stream_ms)
        unbilled_ms: Final = billable_ms - self._billed_ms
        if unbilled_ms <= 0:
            return None
        self._billed_ms = billable_ms
        return duration_usage(unbilled_ms / 1000)

    def _report(self, error: SonioxException) -> OpenAIRealtimeErrorEvent:
        if self._processed_ms == 0:
            self._refusal = error
        return error_event(error.message)

    def _bill(self, total_audio_proc_ms: int | None) -> None:
        if total_audio_proc_ms is not None and total_audio_proc_ms > self._processed_ms:
            self._processed_ms = total_audio_proc_ms

    def _token_events(self, token: _SonioxToken) -> tuple[OpenAIRealtimeEvents, ...]:
        if token.text == _FINALIZED_TOKEN:
            return self._finalized()
        if self._discarding:
            return ()
        if token.text == _ENDPOINT_TOKEN:
            return self._end_turn()
        started: Final = self._start()
        if not token.is_final or (self._drops_originals and token.translation_status == "original"):
            return started
        self._final_tokens.append(token)
        return (*started, _delta(self._current_item_id(), token))

    @property
    def _discarding(self) -> bool:
        return bool(self._finalizes) and self._finalizes[0]

    def _finalized(self) -> tuple[OpenAIRealtimeEvents, ...]:
        discarded: Final = self._finalizes.popleft() if self._finalizes else None
        answered: Final = self._oldest_commit_answered
        self._oldest_commit_answered = False
        if discarded:
            return ()
        if self._item_id is not None:
            return self._complete(speech_stopped=False)
        if discarded is None or answered:
            return ()
        return (_completed(_new_item_id(), (), self.take_unbilled_usage()),)

    def _end_turn(self) -> tuple[OpenAIRealtimeEvents, ...]:
        events: Final = self._complete(speech_stopped=True)
        if events and self._finalizes:
            self._oldest_commit_answered = True
        return events

    def _start(self) -> tuple[OpenAIRealtimeEvents, ...]:
        if self._item_id is not None:
            return ()
        item_id: Final = _new_item_id()
        self._item_id = item_id
        return (speech_event("input_audio_buffer.speech_started", item_id),)

    def _current_item_id(self) -> str:
        assert self._item_id is not None
        return self._item_id

    def _complete(self, *, speech_stopped: bool) -> tuple[OpenAIRealtimeEvents, ...]:
        item_id: Final = self._item_id
        if item_id is None:
            return ()
        tokens: Final = tuple(self._final_tokens)
        self._item_id = None
        self._final_tokens.clear()
        stopped: Final[tuple[OpenAIRealtimeEvents, ...]] = (
            (speech_event("input_audio_buffer.speech_stopped", item_id),) if speech_stopped else ()
        )
        return (*stopped, _completed(item_id, tokens, self.take_unbilled_usage()))


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
            except Exception as e:  # noqa: BLE001  # recv() re-raises it, so an upstream failure never hangs the reader
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
            except Exception as e:  # noqa: BLE001  # stops only keepalives; the pump reports the broken socket
                verbose_logger.debug("Soniox realtime: stopping keepalive after send failed: %s", e)
                return


class SonioxRealtimeConfig(BaseRealtimeConfig):
    def __init__(
        self,
        *,
        options: SonioxRealtimeOptions | None = None,
        wrap: Callable[[RealtimeBackend], RealtimeBackend] = SonioxRealtimeBackend,
        clock: Callable[[], float] = time.monotonic,
        drop_params: bool = False,
    ) -> None:
        self._options: Final = options or SonioxRealtimeOptions()
        self._drop_params: Final = drop_params
        self._wrap: Final = wrap
        self._clock: Final = clock
        self._transformer: Final = SonioxEventTransformer(drops_originals=self._options.translation is not None)
        self._start_request: SonioxStartRequest | None = None
        self._session_id: str = f"sess_{uuid.uuid4().hex}"
        self._started_at: float | None = None

    @classmethod
    def from_litellm_params(cls, litellm_params: Mapping[str, object], *, drop_params: bool = False) -> Self:
        return cls(options=SonioxRealtimeOptions.model_validate(dict(litellm_params)), drop_params=drop_params)

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
        if event_type is None:
            raise SonioxProtocolError("Soniox realtime takes OpenAI realtime events, and this message has no type")
        if event_type in SESSION_UPDATE_EVENT_TYPES:
            update: Final = parse_transcription_session_update(message, SonioxProtocolError)
            ignored: Final = _ignored_settings(update)
            if ignored:
                verbose_logger.warning("Soniox realtime: ignoring unsupported session settings %s", ignored)
            return self._start(model, update)
        if event_type == "input_audio_buffer.append":
            audio: Final = decode_pcm16_append(request.get("audio"), error=SonioxProtocolError)
            return (*self._start(model, None), audio)
        if event_type == "input_audio_buffer.commit":
            return self._finalize(model, discards_turn=False)
        if event_type == "input_audio_buffer.clear":
            return () if self._start_request is None else self._finalize(model, discards_turn=True)
        if event_type == "input_audio_buffer.end":
            return (*self._start(model, None), _END_STREAM)
        verbose_logger.debug("Soniox realtime: dropping unsupported client event %s", event_type)
        return ()

    def _finalize(self, model: str, *, discards_turn: bool) -> tuple[str, ...]:
        start: Final = self._start(model, None)
        self._transformer.queue_finalize(discards_turn=discards_turn)
        return (*start, _FINALIZE)

    def session_refusal(self) -> Exception | None:
        return self._transformer.refusal

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        started_at: Final = self._started_at
        if self._transformer.refusal is not None:
            return None
        if started_at is None:
            return self._transformer.take_unbilled_usage()
        return self._transformer.take_unbilled_usage(int((self._clock() - started_at) * 1000))

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
            if update is not None and _changes_started_stream(update, started, self._drop_params):
                raise SonioxProtocolError("Soniox realtime can't change the session once the stream has started")
            return ()
        start_request: Final = build_start_request(model, update, self._options, self._drop_params)
        self._start_request = start_request
        self._started_at = self._clock()
        return (start_request.model_dump_json(exclude_none=True),)


def _finishes_stream(frame: str | bytes) -> bool:
    response: Final = _SonioxResponse.model_validate_json(frame)
    error: Final = _soniox_error(response)
    if error is not None:
        raise error
    return response.finished


async def soniox_realtime_health_check(
    model: str, api_base: str | None, api_key: str | None, options: SonioxRealtimeOptions
) -> bool:
    import websockets

    config: Final = SonioxRealtimeConfig()
    url: Final = config.get_complete_url(api_base, model)
    headers: Final = config.validate_environment({}, model, api_key)
    start_request: Final = build_start_request(model, None, options)
    async with websockets.connect(
        url,
        additional_headers=headers,
        max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ssl=realtime_ssl_for_url(url),
    ) as backend:
        await backend.send(start_request.model_dump_json(exclude_none=True))
        await backend.send(_HEALTH_CHECK_SILENCE)
        await backend.send(_END_STREAM)
        async for frame in backend:
            if _finishes_stream(frame):
                return True
    raise SonioxException(status_code=502, message="Soniox closed the realtime stream without finishing it")

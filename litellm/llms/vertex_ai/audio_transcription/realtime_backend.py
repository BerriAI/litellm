import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import TypeAdapter
from typing_extensions import Self, assert_never
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from litellm import verbose_logger
from litellm.llms.vertex_ai.audio_transcription.realtime_transformation import SpeechStreamingTarget
from litellm.types.llms.vertex_ai_speech_to_text import (
    VertexSpeechStreamingCommand,
    VertexSpeechStreamingCommandUnion,
    VertexSpeechStreamingConfigure,
    VertexSpeechStreamingConfigured,
    VertexSpeechStreamingDiscardTurn,
    VertexSpeechStreamingFinishTurn,
    VertexSpeechStreamingResponse,
    VertexSpeechStreamingResult,
    VertexSpeechStreamingTurnDiscarded,
    VertexSpeechStreamingTurnFinished,
)

if TYPE_CHECKING:
    from google.cloud.speech_v2.types import (
        StreamingRecognitionConfig,
        StreamingRecognizeRequest,
        StreamingRecognizeResponse,
    )

SPEECH_SDK_INSTALL_HINT: Final = (
    "google-cloud-speech is not installed. Install with `pip install 'litellm[stt-vertex-chirp]'`."
)
STREAM_FAILURE_CLOSE_CODE: Final = 1011
STREAM_ROTATION_SECONDS: Final = 240.0
STREAM_ROTATION_DEADLINE_SECONDS: Final = 280.0
REQUEST_QUEUE_SIZE: Final = 64
OUTBOX_SIZE: Final = 256
_LINK_QUEUE_SIZE: Final = 64
_CLOSE_REASON_MAX_CHARS: Final = 120
_CONFIGURED_EVENT: Final = VertexSpeechStreamingConfigured().model_dump_json()
_TURN_FINISHED_EVENT: Final = VertexSpeechStreamingTurnFinished().model_dump_json()
_TURN_DISCARDED_EVENT: Final = VertexSpeechStreamingTurnDiscarded().model_dump_json()
_COMMAND_ADAPTER: Final = TypeAdapter[VertexSpeechStreamingCommandUnion](VertexSpeechStreamingCommand)
_TIMEDELTA_ADAPTER: Final = TypeAdapter(timedelta)
_SPEECH_EVENTS: Final[MappingProxyType[str, Literal["begin", "end"]]] = MappingProxyType(
    {
        "SPEECH_ACTIVITY_BEGIN": "begin",
        "SPEECH_ACTIVITY_END": "end",
        "END_OF_SINGLE_UTTERANCE": "end",
    }
)


class ClosableTransport(Protocol):
    def close(self) -> Awaitable[None]: ...


class SpeechStreamingClient(Protocol):
    def streaming_recognize(
        self, requests: "AsyncIterator[StreamingRecognizeRequest] | None" = None
    ) -> "Awaitable[AsyncIterable[StreamingRecognizeResponse]]": ...

    @property
    def transport(self) -> ClosableTransport: ...


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    reason: str


@dataclass(frozen=True, slots=True)
class _Closed:
    pass


def open_speech_client(target: SpeechStreamingTarget) -> SpeechStreamingClient:
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechAsyncClient
        from google.oauth2.credentials import Credentials
    except ImportError as e:
        raise ImportError(SPEECH_SDK_INSTALL_HINT) from e
    return SpeechAsyncClient(
        credentials=Credentials(token=target.access_token),
        transport="grpc_asyncio",
        client_options=ClientOptions(api_endpoint=target.api_endpoint),
    )


def _streaming_config(command: VertexSpeechStreamingConfigure) -> "StreamingRecognitionConfig":
    from google.cloud.speech_v2.types import (
        ExplicitDecodingConfig,
        RecognitionConfig,
        StreamingRecognitionConfig,
        StreamingRecognitionFeatures,
    )

    return StreamingRecognitionConfig(
        config=RecognitionConfig(
            explicit_decoding_config=ExplicitDecodingConfig(
                encoding=ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=command.sample_rate_hertz,
                audio_channel_count=1,
            ),
            model=command.model,
            language_codes=command.language_codes,
        ),
        streaming_features=StreamingRecognitionFeatures(interim_results=True, enable_voice_activity_events=True),
    )


def _response_event(response: "StreamingRecognizeResponse", billed_seconds: float) -> str:
    return VertexSpeechStreamingResponse(
        speech_event=_SPEECH_EVENTS.get(response.speech_event_type.name, "none"),
        results=tuple(
            VertexSpeechStreamingResult(
                transcript=result.alternatives[0].transcript if result.alternatives else "",
                is_final=result.is_final,
            )
            for result in response.results
        ),
        billed_seconds=billed_seconds,
    ).model_dump_json()


def _billed_seconds(response: "StreamingRecognizeResponse") -> float:
    return _TIMEDELTA_ADAPTER.validate_python(response.metadata.total_billed_duration).total_seconds()


def _normal_closure() -> ConnectionClosedOK:
    return ConnectionClosedOK(rcvd=Close(1000, ""), sent=None)


class _RecognizeStream:
    def __init__(
        self,
        *,
        client: SpeechStreamingClient,
        request_type: "type[StreamingRecognizeRequest]",
        first_request: "StreamingRecognizeRequest",
        opened_at: float,
    ) -> None:
        self._client: Final = client
        self._request_type: Final = request_type
        self.opened_at: Final = opened_at
        self._requests: Final[asyncio.Queue[StreamingRecognizeRequest | None]] = asyncio.Queue(
            maxsize=REQUEST_QUEUE_SIZE
        )
        self._requests.put_nowait(first_request)
        self.speech_active: bool = False
        self.billed_seconds: float = 0.0
        self._cancelled: bool = False
        self._task: asyncio.Task[None] | None = None

    async def send_audio(self, audio: bytes) -> None:
        await self._requests.put(self._request_type(audio=audio))

    async def half_close(self) -> None:
        await self._requests.put(None)

    def cancel(self) -> None:
        self._cancelled = True
        if self._task is not None:
            self._task.cancel()

    async def relay(self, outbox: "asyncio.Queue[str | _StreamFailure | _Closed]", billed_before: float) -> float:
        if self._cancelled:
            return 0.0
        task: Final = asyncio.create_task(self._forward(outbox, billed_before))
        self._task = task
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.wait((task,))
            raise
        return self.billed_seconds

    async def _forward(self, outbox: "asyncio.Queue[str | _StreamFailure | _Closed]", billed_before: float) -> None:
        try:
            responses: Final = await self._client.streaming_recognize(self._drain())
            async for response in responses:
                self._note(response)
                await outbox.put(_response_event(response, billed_before + self.billed_seconds))
        except Exception as e:  # noqa: BLE001  # task boundary: a swallowed failure would hang the client session
            verbose_logger.warning("Google Speech-to-Text streaming failed: %s", e)
            await outbox.put(_StreamFailure(reason=f"Google Speech-to-Text streaming failed: {e}"))

    def _note(self, response: "StreamingRecognizeResponse") -> None:
        activity: Final = _SPEECH_EVENTS.get(response.speech_event_type.name)
        if activity is not None:
            self.speech_active = activity == "begin"
        self.billed_seconds = max(self.billed_seconds, _billed_seconds(response))

    async def _drain(self) -> "AsyncIterator[StreamingRecognizeRequest]":
        while (request := await self._requests.get()) is not None:
            yield request


class SpeechStreamingBackend:
    def __init__(
        self,
        target: SpeechStreamingTarget,
        *,
        client_factory: Callable[[SpeechStreamingTarget], SpeechStreamingClient] = open_speech_client,
        clock: Callable[[], float] = time.monotonic,
        rotation_seconds: float = STREAM_ROTATION_SECONDS,
        rotation_deadline_seconds: float = STREAM_ROTATION_DEADLINE_SECONDS,
    ) -> None:
        self._target: Final = target
        self._client_factory: Final = client_factory
        self._clock: Final = clock
        self._rotation_seconds: Final = rotation_seconds
        self._rotation_deadline_seconds: Final = rotation_deadline_seconds
        self._outbox: Final[asyncio.Queue[str | _StreamFailure | _Closed]] = asyncio.Queue(maxsize=OUTBOX_SIZE)
        self._links: Final[asyncio.Queue[_RecognizeStream | str]] = asyncio.Queue(maxsize=_LINK_QUEUE_SIZE)
        self._pump: asyncio.Task[None] | None = None
        self._client: SpeechStreamingClient | None = None
        self._config: StreamingRecognitionConfig | None = None
        self._turn: tuple[_RecognizeStream, ...] = ()
        self._billed_before: float = 0.0
        self._closed: bool = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def send(self, message: str | bytes) -> None:
        if self._closed:
            raise _normal_closure()
        if isinstance(message, bytes):
            await self._send_audio(message)
            return
        command: Final = _COMMAND_ADAPTER.validate_json(message)
        match command:
            case VertexSpeechStreamingConfigure():
                self._config = _streaming_config(command)
                await self._link(_CONFIGURED_EVENT)
            case VertexSpeechStreamingFinishTurn():
                await self._finish_turn()
            case VertexSpeechStreamingDiscardTurn():
                await self._discard_turn()
            case _:
                assert_never(command)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        if self._closed and self._outbox.empty():
            raise _normal_closure()
        item: Final = await self._outbox.get()
        match item:
            case _StreamFailure():
                raise ConnectionClosedError(
                    rcvd=Close(STREAM_FAILURE_CLOSE_CODE, item.reason[:_CLOSE_REASON_MAX_CHARS]), sent=None
                )
            case _Closed():
                raise _normal_closure()
            case str():
                return item
            case _:
                assert_never(item)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._turn = ()
        pump: Final = self._pump
        if pump is not None:
            pump.cancel()
            await asyncio.wait((pump,))
        client: Final = self._client
        self._client = None
        if client is not None:
            await client.transport.close()
        if not self._outbox.full():
            self._outbox.put_nowait(_Closed())

    async def _link(self, item: _RecognizeStream | str) -> None:
        if self._pump is None:
            self._pump = asyncio.create_task(self._pump_links())
        await self._links.put(item)

    async def _pump_links(self) -> None:
        while True:
            await self._relay(await self._links.get())

    async def _relay(self, link: _RecognizeStream | str) -> None:
        match link:
            case str():
                await self._outbox.put(link)
            case _RecognizeStream():
                self._billed_before += await link.relay(self._outbox, self._billed_before)
            case _:
                assert_never(link)

    async def _send_audio(self, audio: bytes) -> None:
        stream: Final = await self._turn_stream()
        await stream.send_audio(audio)

    async def _turn_stream(self) -> _RecognizeStream:
        current: Final = self._turn[-1] if self._turn else None
        if current is not None and not self._expired(current):
            return current
        if current is not None:
            await current.half_close()
        stream: Final = await self._open_stream()
        self._turn = (*self._turn, stream)
        return stream

    def _expired(self, stream: _RecognizeStream) -> bool:
        elapsed: Final = self._clock() - stream.opened_at
        if elapsed >= self._rotation_deadline_seconds:
            return True
        return elapsed >= self._rotation_seconds and not stream.speech_active

    async def _open_stream(self) -> _RecognizeStream:
        from google.cloud.speech_v2.types import StreamingRecognizeRequest

        config: Final = self._config
        if config is None:
            raise RuntimeError("audio was sent before the Speech-to-Text stream was configured")
        if self._client is None:
            self._client = self._client_factory(self._target)
        stream: Final = _RecognizeStream(
            client=self._client,
            request_type=StreamingRecognizeRequest,
            first_request=StreamingRecognizeRequest(recognizer=self._target.recognizer, streaming_config=config),
            opened_at=self._clock(),
        )
        await self._link(stream)
        return stream

    async def _finish_turn(self) -> None:
        turn: Final = self._turn
        self._turn = ()
        if turn:
            await turn[-1].half_close()
        await self._link(_TURN_FINISHED_EVENT)

    async def _discard_turn(self) -> None:
        turn: Final = self._turn
        self._turn = ()
        for stream in turn:
            stream.cancel()
        await self._link(_TURN_DISCARDED_EVENT)

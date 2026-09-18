import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Final, Literal, Protocol, Self

from pydantic import TypeAdapter
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


class _RecognizeStream:
    def __init__(
        self,
        *,
        client: SpeechStreamingClient,
        request_type: "type[StreamingRecognizeRequest]",
        first_request: "StreamingRecognizeRequest",
        outbox: "asyncio.Queue[str | _StreamFailure | _Closed]",
        previous: "_RecognizeStream | None",
        opened_at: float,
    ) -> None:
        self._client: Final = client
        self._request_type: Final = request_type
        self._outbox: Final = outbox
        self._previous: _RecognizeStream | None = previous
        self.opened_at: Final = opened_at
        self._requests: Final[asyncio.Queue[StreamingRecognizeRequest | None]] = asyncio.Queue()
        self._requests.put_nowait(first_request)
        self._billed_seconds: float = 0.0
        self._base_billed_seconds: float = 0.0
        self._task: Final = asyncio.create_task(self._run())

    @property
    def billed_seconds(self) -> float:
        return self._base_billed_seconds + self._billed_seconds

    def send_audio(self, audio: bytes) -> None:
        self._requests.put_nowait(self._request_type(audio=audio))

    def half_close(self) -> None:
        self._requests.put_nowait(None)

    async def wait(self) -> None:
        await asyncio.gather(self._task, return_exceptions=True)

    async def cancel(self) -> None:
        self._task.cancel()
        await self.wait()

    async def cancel_chain(self) -> None:
        previous: Final = self._previous
        self._task.cancel()
        if previous is not None:
            await previous.cancel_chain()
        await self.wait()

    async def _run(self) -> None:
        previous: Final = self._previous
        if previous is not None:
            await previous.wait()
            self._base_billed_seconds = previous.billed_seconds
            self._previous = None
        try:
            responses: Final = await self._client.streaming_recognize(self._drain())
            async for response in responses:
                self._billed_seconds = max(self._billed_seconds, _billed_seconds(response))
                await self._outbox.put(_response_event(response, self.billed_seconds))
            await self._outbox.put(_TURN_FINISHED_EVENT)
        except asyncio.CancelledError:
            self._outbox.put_nowait(_TURN_DISCARDED_EVENT)
            raise
        except Exception as e:  # noqa: BLE001  # task boundary: a swallowed failure would hang the client session
            verbose_logger.warning("Google Speech-to-Text streaming failed: %s", e)
            await self._outbox.put(_StreamFailure(reason=f"Google Speech-to-Text streaming failed: {e}"))

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
    ) -> None:
        self._target: Final = target
        self._client_factory: Final = client_factory
        self._clock: Final = clock
        self._rotation_seconds: Final = rotation_seconds
        self._outbox: Final[asyncio.Queue[str | _StreamFailure | _Closed]] = asyncio.Queue()
        self._client: SpeechStreamingClient | None = None
        self._config: StreamingRecognitionConfig | None = None
        self._stream: _RecognizeStream | None = None
        self._last_stream: _RecognizeStream | None = None

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
        if isinstance(message, bytes):
            self._send_audio(message)
            return
        command: Final = _COMMAND_ADAPTER.validate_json(message)
        match command:
            case VertexSpeechStreamingConfigure():
                self._config = _streaming_config(command)
                await self._outbox.put(_CONFIGURED_EVENT)
            case VertexSpeechStreamingFinishTurn():
                self._finish_turn()
            case VertexSpeechStreamingDiscardTurn():
                await self._discard_turn()

    async def recv(self, decode: bool | None = None) -> str | bytes:
        item: Final = await self._outbox.get()
        match item:
            case _StreamFailure():
                raise ConnectionClosedError(
                    rcvd=Close(STREAM_FAILURE_CLOSE_CODE, item.reason[:_CLOSE_REASON_MAX_CHARS]), sent=None
                )
            case _Closed():
                raise ConnectionClosedOK(rcvd=Close(1000, ""), sent=None)
            case str():
                return item

    async def close(self) -> None:
        self._stream = None
        last_stream: Final = self._last_stream
        self._last_stream = None
        if last_stream is not None:
            await last_stream.cancel_chain()
        client: Final = self._client
        self._client = None
        if client is not None:
            await client.transport.close()
        self._outbox.put_nowait(_Closed())

    def _send_audio(self, audio: bytes) -> None:
        self._rotate_expiring_stream()
        stream: Final = self._stream if self._stream is not None else self._open_stream()
        stream.send_audio(audio)

    def _rotate_expiring_stream(self) -> None:
        stream: Final = self._stream
        if stream is None or self._clock() - stream.opened_at < self._rotation_seconds:
            return
        self._stream = None
        stream.half_close()

    def _open_stream(self) -> _RecognizeStream:
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
            outbox=self._outbox,
            previous=self._last_stream,
            opened_at=self._clock(),
        )
        self._stream = stream
        self._last_stream = stream
        return stream

    def _finish_turn(self) -> None:
        stream: Final = self._stream
        self._stream = None
        if stream is None:
            self._outbox.put_nowait(_TURN_FINISHED_EVENT)
            return
        stream.half_close()

    async def _discard_turn(self) -> None:
        stream: Final = self._stream
        self._stream = None
        if stream is None:
            await self._outbox.put(_TURN_DISCARDED_EVENT)
            return
        await stream.cancel()

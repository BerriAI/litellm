import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import Self
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidHandshake, InvalidStatus
from websockets.frames import Close

from litellm import verbose_logger
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.llms.base_llm.realtime.transformation import RealtimeBackend
from litellm.types.llms.elevenlabs import ElevenLabsRealtimeConfigure

RealtimeConnect: TypeAlias = Callable[[str, Mapping[str, str]], Awaitable[RealtimeBackend]]

_OPEN_TIMEOUT_SECONDS: Final = 8.0


class _AudioChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: Literal["input_audio_chunk"]
    audio_base_64: str
    commit: bool
    sample_rate: int


_AUDIO_CHUNK_ADAPTER: Final = TypeAdapter(_AudioChunk)


async def open_realtime_websocket(url: str, headers: Mapping[str, str]) -> RealtimeBackend:
    from websockets.asyncio.client import connect

    from litellm.llms.custom_httpx.http_handler import realtime_ssl_for_url

    connection: Final = await connect(
        url,
        additional_headers=dict(headers),
        max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ssl=realtime_ssl_for_url(url),
        open_timeout=_OPEN_TIMEOUT_SECONDS,
    )
    return _WebsocketBackend(connection)


class _WebsocketBackend:
    def __init__(self, connection: ClientConnection) -> None:
        self._connection: Final = connection

    async def __aenter__(self) -> Self:
        await self._connection.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._connection.__aexit__(exc_type, exc_value, traceback)

    async def send(self, message: str | bytes) -> None:
        await self._connection.send(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        return await self._connection.recv(decode)

    async def close(self) -> None:
        await self._connection.close()


def _normal_closure() -> ConnectionClosedOK:
    return ConnectionClosedOK(rcvd=Close(1000, ""), sent=None)


def _failure_close(error: OSError | TimeoutError | asyncio.TimeoutError | InvalidHandshake) -> Close:
    reason: Final = (
        f"ElevenLabs realtime connection failed (HTTP {error.response.status_code})"
        if isinstance(error, InvalidStatus)
        else "ElevenLabs realtime connection failed"
    )
    verbose_logger.warning(reason)
    return Close(1011, reason)


def _configure_query(message: str | bytes) -> str:
    try:
        return ElevenLabsRealtimeConfigure.model_validate_json(message).query
    except ValidationError:
        raise RuntimeError("audio was sent before the ElevenLabs realtime session was configured") from None


class ElevenLabsRealtimeBackend:
    def __init__(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        connect: RealtimeConnect = open_realtime_websocket,
        on_audio_sent: Callable[[float], None] | None = None,
    ) -> None:
        self._url: Final = url
        self._headers: Final = headers
        self._connect: Final = connect
        self._on_audio_sent: Final = on_audio_sent
        self._ready: Final = asyncio.Event()
        self._upstream: RealtimeBackend | None = None
        self._closed: bool = False
        self._opening: bool = False
        self._failure: Close | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def _closure(self) -> ConnectionClosedError | ConnectionClosedOK:
        failure: Final = self._failure
        if failure is not None:
            return ConnectionClosedError(rcvd=failure, sent=None)
        return _normal_closure()

    async def send(self, message: str | bytes) -> None:
        upstream: Final = self._upstream
        if upstream is not None:
            await upstream.send(message)
            if self._on_audio_sent is not None and isinstance(message, str):
                try:
                    chunk: Final = _AUDIO_CHUNK_ADAPTER.validate_json(message)
                except ValidationError:
                    return
                encoded: Final = chunk.audio_base_64
                audio_bytes: Final = len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("=")))
                if chunk.sample_rate > 0:
                    self._on_audio_sent(audio_bytes / (chunk.sample_rate * 2))
            return
        if self._closed or self._failure is not None:
            raise self._closure()
        if self._opening:
            raise RuntimeError("the ElevenLabs realtime session is already being configured")
        query: Final = _configure_query(message)
        self._opening = True
        try:
            opened: Final = await self._connect(f"{self._url}?{query}", self._headers)
        except (OSError, TimeoutError, asyncio.TimeoutError, InvalidHandshake) as error:
            if self._closed:
                raise _normal_closure() from None
            self._failure = _failure_close(error)
            self._ready.set()
            return
        finally:
            self._opening = False
        if self._closed:
            await opened.close()
            raise _normal_closure()
        self._upstream = opened
        self._ready.set()

    async def recv(self, decode: bool | None = None) -> str | bytes:
        await self._ready.wait()
        upstream: Final = self._upstream
        if upstream is None:
            raise self._closure()
        return await upstream.recv(decode)

    async def close(self) -> None:
        self._closed = True
        self._ready.set()
        upstream: Final = self._upstream
        if upstream is not None:
            await upstream.close()

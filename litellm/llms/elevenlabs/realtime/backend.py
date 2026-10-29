import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Final

from pydantic import ValidationError
from typing_extensions import Self
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.llms.base_llm.realtime.transformation import RealtimeBackend
from litellm.types.llms.elevenlabs import ElevenLabsRealtimeConfigure

RealtimeConnect = Callable[[str, Mapping[str, str]], Awaitable[RealtimeBackend]]

_OPEN_TIMEOUT_SECONDS: Final = 8.0


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


def _configure_query(message: str | bytes) -> str:
    try:
        return ElevenLabsRealtimeConfigure.model_validate_json(message).query
    except ValidationError:
        raise RuntimeError("audio was sent before the ElevenLabs realtime session was configured") from None


class ElevenLabsRealtimeBackend:
    def __init__(
        self, url: str, headers: Mapping[str, str], *, connect: RealtimeConnect = open_realtime_websocket
    ) -> None:
        self._url: Final = url
        self._headers: Final = headers
        self._connect: Final = connect
        self._ready: Final = asyncio.Event()
        self._upstream: RealtimeBackend | None = None
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
        upstream: Final = self._upstream
        if upstream is not None:
            await upstream.send(message)
            return
        if self._closed:
            raise _normal_closure()
        opened: Final = await self._connect(f"{self._url}?{_configure_query(message)}", self._headers)
        if self._closed:
            await opened.close()
            raise _normal_closure()
        self._upstream = opened
        self._ready.set()

    async def recv(self, decode: bool | None = None) -> str | bytes:
        await self._ready.wait()
        upstream: Final = self._upstream
        if upstream is None:
            raise _normal_closure()
        return await upstream.recv(decode)

    async def close(self) -> None:
        self._closed = True
        self._ready.set()
        upstream: Final = self._upstream
        if upstream is not None:
            await upstream.close()

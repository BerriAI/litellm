from collections.abc import AsyncIterator, Callable
from typing import Final

import httpx

HTTP2_STREAMS_PER_CONNECTION: Final = 100


class _Lane:
    __slots__ = ("in_flight", "transport")

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self.transport: Final = transport
        self.in_flight = 0

    def acquire(self) -> None:
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1


class _Lanes:
    def __init__(
        self,
        new_transport: Callable[[], httpx.AsyncBaseTransport],
        streams_per_connection: int,
        max_connections: int | None,
    ) -> None:
        self._new_transport: Final = new_transport
        self._streams_per_connection: Final = streams_per_connection
        self._max_connections: Final = max_connections
        self.lanes: tuple[_Lane, ...] = ()

    def acquire(self) -> _Lane:
        free: Final = next((lane for lane in self.lanes if lane.in_flight < self._streams_per_connection), None)
        lane: Final = free if free is not None else self._open_or_share()
        lane.acquire()
        return lane

    def _open_or_share(self) -> _Lane:
        if self._max_connections is not None and len(self.lanes) >= self._max_connections:
            return min(self.lanes, key=lambda lane: lane.in_flight)
        return self._open()

    def _open(self) -> _Lane:
        lane: Final = _Lane(self._new_transport())
        self.lanes = (*self.lanes, lane)
        return lane


class _AsyncReleasingStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, release: Callable[[], None]) -> None:
        self._stream: Final = stream
        self._release: Final = release

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        self._release()
        await self._stream.aclose()


def _with_stream(response: httpx.Response, stream: httpx.AsyncByteStream) -> httpx.Response:
    return httpx.Response(response.status_code, headers=response.headers, stream=stream, extensions=response.extensions)


class AsyncHTTP2ConnectionSpread(httpx.AsyncBaseTransport):
    def __init__(
        self,
        new_transport: Callable[[], httpx.AsyncBaseTransport],
        streams_per_connection: int = HTTP2_STREAMS_PER_CONNECTION,
        max_connections: int | None = None,
    ) -> None:
        self._lanes: Final = _Lanes(new_transport, streams_per_connection, max_connections)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        lane: Final = self._lanes.acquire()
        try:
            response: Final = await lane.transport.handle_async_request(request)
        except BaseException:
            lane.release()
            raise
        stream: Final = response.stream
        if not isinstance(stream, httpx.AsyncByteStream):
            lane.release()
            return response
        return _with_stream(response, _AsyncReleasingStream(stream, lane.release))

    async def aclose(self) -> None:
        for lane in self._lanes.lanes:
            await lane.transport.aclose()

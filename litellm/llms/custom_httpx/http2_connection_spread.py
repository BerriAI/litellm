import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Final, Generic, TypeVar

import httpx

HTTP2_STREAMS_PER_CONNECTION: Final = 100

_TransportT: Final = TypeVar("_TransportT", httpx.BaseTransport, httpx.AsyncBaseTransport)


class _Lane(Generic[_TransportT]):
    __slots__ = ("in_flight", "transport")

    def __init__(self, transport: _TransportT) -> None:
        self.transport: Final = transport
        self.in_flight = 0

    def acquire(self) -> None:
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1


class _Lanes(Generic[_TransportT]):
    def __init__(
        self, new_transport: Callable[[], _TransportT], streams_per_connection: int, max_connections: int | None
    ) -> None:
        self._new_transport: Final = new_transport
        self._streams_per_connection: Final = streams_per_connection
        self._max_connections: Final = max_connections
        self.lanes: tuple[_Lane[_TransportT], ...] = ()

    def acquire(self) -> _Lane[_TransportT]:
        free: Final = next((lane for lane in self.lanes if lane.in_flight < self._streams_per_connection), None)
        lane: Final = free if free is not None else self._open_or_share()
        lane.acquire()
        return lane

    def _open_or_share(self) -> _Lane[_TransportT]:
        if self._max_connections is not None and len(self.lanes) >= self._max_connections:
            return min(self.lanes, key=lambda lane: lane.in_flight)
        return self._open()

    def _open(self) -> _Lane[_TransportT]:
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


class _ReleasingStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, release: Callable[[], None]) -> None:
        self._stream: Final = stream
        self._release: Final = release

    def __iter__(self) -> Iterator[bytes]:
        yield from self._stream

    def close(self) -> None:
        self._release()
        self._stream.close()


def _with_stream(response: httpx.Response, stream: httpx.SyncByteStream | httpx.AsyncByteStream) -> httpx.Response:
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


class HTTP2ConnectionSpread(httpx.BaseTransport):
    def __init__(
        self,
        new_transport: Callable[[], httpx.BaseTransport],
        streams_per_connection: int = HTTP2_STREAMS_PER_CONNECTION,
        max_connections: int | None = None,
    ) -> None:
        self._lanes: Final = _Lanes(new_transport, streams_per_connection, max_connections)
        self._lock: Final = threading.Lock()

    def _release(self, lane: _Lane[httpx.BaseTransport]) -> None:
        with self._lock:
            lane.release()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            lane: Final = self._lanes.acquire()
        try:
            response: Final = lane.transport.handle_request(request)
        except BaseException:
            self._release(lane)
            raise
        stream: Final = response.stream
        if not isinstance(stream, httpx.SyncByteStream):
            self._release(lane)
            return response
        return _with_stream(response, _ReleasingStream(stream, lambda: self._release(lane)))

    def close(self) -> None:
        for lane in self._lanes.lanes:
            lane.transport.close()

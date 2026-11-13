import weakref
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
        new_transport: Callable[[httpx.Limits], httpx.AsyncBaseTransport],
        limits: httpx.Limits,
        streams_per_connection: int,
    ) -> None:
        self._new_transport: Final = new_transport
        self._limits: Final = limits
        self._streams_per_connection: Final = streams_per_connection
        self.lanes: tuple[_Lane, ...] = ()

    def acquire(self) -> _Lane:
        free: Final = next((lane for lane in self.lanes if lane.in_flight < self._streams_per_connection), None)
        lane: Final = free if free is not None else self._open_or_share()
        lane.acquire()
        return lane

    def _open_or_share(self) -> _Lane:
        cap: Final = self._limits.max_connections
        if cap is not None and len(self.lanes) * self._streams_per_connection >= cap:
            return min(self.lanes, key=lambda lane: lane.in_flight)
        return self._open()

    def _open(self) -> _Lane:
        lane: Final = _Lane(self._new_transport(self._next_lane_limits()))
        self.lanes = (*self.lanes, lane)
        return lane

    def _next_lane_limits(self) -> httpx.Limits:
        cap: Final = self._limits.max_connections
        if cap is None:
            return self._limits
        return httpx.Limits(
            max_connections=min(self._streams_per_connection, cap - len(self.lanes) * self._streams_per_connection),
            max_keepalive_connections=self._limits.max_keepalive_connections,
            keepalive_expiry=self._limits.keepalive_expiry,
        )


class _AsyncReleasingStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, release: Callable[[], None]) -> None:
        self._stream: Final = stream
        self._release: Final = weakref.finalize(self, release)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except BaseException:
            self._release()
            raise

    async def aclose(self) -> None:
        self._release()
        await self._stream.aclose()


def _allowed_over_http2(name: bytes, value: bytes) -> bool:
    return name.lower() != b"te" or value.strip().lower() == b"trailers"


def _without_te_codings(request: httpx.Request) -> httpx.Request:
    if request.url.scheme != "https":
        return request
    kept: Final = tuple((name, value) for name, value in request.headers.raw if _allowed_over_http2(name, value))
    if len(kept) == len(request.headers.raw):
        return request
    return httpx.Request(
        request.method, request.url, headers=kept, stream=request.stream, extensions=request.extensions
    )


def _with_stream(response: httpx.Response, stream: httpx.AsyncByteStream) -> httpx.Response:
    return httpx.Response(response.status_code, headers=response.headers, stream=stream, extensions=response.extensions)


class AsyncHTTP2ConnectionSpread(httpx.AsyncBaseTransport):
    def __init__(
        self,
        new_transport: Callable[[httpx.Limits], httpx.AsyncBaseTransport],
        limits: httpx.Limits,
        streams_per_connection: int = HTTP2_STREAMS_PER_CONNECTION,
    ) -> None:
        self._lanes: Final = _Lanes(new_transport, limits, streams_per_connection)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        lane: Final = self._lanes.acquire()
        try:
            response: Final = await lane.transport.handle_async_request(_without_te_codings(request))
        except BaseException:
            lane.release()
            raise
        stream: Final = response.stream
        if not isinstance(stream, httpx.AsyncByteStream):
            lane.release()
            return response
        return _with_stream(response, _AsyncReleasingStream(stream, lane.release))

    def has_request_in_flight(self) -> bool:
        return any(lane.in_flight > 0 for lane in self._lanes.lanes)

    async def aclose(self) -> None:
        for lane in self._lanes.lanes:
            await lane.transport.aclose()

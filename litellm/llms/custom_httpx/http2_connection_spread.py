import asyncio
import inspect
import itertools
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

import h2.events
import h2.settings
import httpcore
import httpx

HTTP2_STREAMS_PER_CONNECTION: Final = 100
_REMOTE_SETTINGS_RECEIVED: Final = "http2.receive_remote_settings.complete"
_HTTP1_REQUEST_STARTED: Final = "http11.send_request_headers.started"

_Origin: TypeAlias = tuple[str, str, int | None]
_Trace: TypeAlias = Callable[[str, Mapping[str, object]], Awaitable[None]]


class _Lane:
    __slots__ = ("in_flight", "transport")

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self.transport: Final = transport
        self.in_flight = 0

    def acquire(self) -> None:
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1


def _is_tunnel_setup(info: Mapping[str, object]) -> bool:
    request: Final = info.get("request")
    return isinstance(request, httpcore.Request) and request.method == b"CONNECT"


class _OriginLanes:
    def __init__(self, most_streams_per_connection: int, may_speak_http2: bool) -> None:
        self._most_streams_per_connection: Final = most_streams_per_connection
        self._probe_settled = asyncio.Event()
        self._probing = False
        self.lanes: tuple[_Lane, ...] = ()
        self.streams_per_connection: int | None = None if may_speak_http2 else most_streams_per_connection

    def streams_or_most(self) -> int:
        return self.streams_per_connection or self._most_streams_per_connection

    def open_lane(self, transport: httpx.AsyncBaseTransport) -> _Lane:
        lane: Final = _Lane(transport)
        self.lanes = (*self.lanes, lane)
        return lane

    def total_streams(self) -> int:
        return self.streams_or_most() * len(self.lanes)

    def free_lane(self) -> _Lane | None:
        streams: Final = self.streams_or_most()
        return next((lane for lane in self.lanes if lane.in_flight < streams), None)

    async def take_turn(self) -> bool:
        while self.streams_per_connection is None and self._probing:
            await self._probe_settled.wait()
        self._probing = self.streams_per_connection is None
        return self._probing

    def end_probe(self, answered: bool) -> None:
        if answered and self.streams_per_connection is None:
            self.streams_per_connection = self._most_streams_per_connection
        self._probing = False
        self._wake_waiters()

    def observe(self, event: str, info: Mapping[str, object]) -> None:
        settings: Final = info.get("return_value") if event == _REMOTE_SETTINGS_RECEIVED else None
        if isinstance(settings, h2.events.RemoteSettingsChanged):
            self.streams_per_connection = self._streams_allowed_by(settings)
            self._wake_waiters()
            return
        if event == _HTTP1_REQUEST_STARTED and self.streams_per_connection is None and not _is_tunnel_setup(info):
            self.streams_per_connection = self._most_streams_per_connection
            self._wake_waiters()

    def _streams_allowed_by(self, settings: h2.events.RemoteSettingsChanged) -> int:
        advertised: Final = settings.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
        allowed: Final = 0 if advertised is None else min(advertised.new_value, self._most_streams_per_connection)
        return allowed or self.streams_per_connection or 1

    def _wake_waiters(self) -> None:
        self._probe_settled.set()
        self._probe_settled = asyncio.Event()


class _Lanes:
    def __init__(
        self,
        new_transport: Callable[[httpx.Limits], httpx.AsyncBaseTransport],
        limits: httpx.Limits,
        most_streams_per_connection: int,
    ) -> None:
        self._new_transport: Final = new_transport
        self._limits: Final = limits
        self._most_streams_per_connection: Final = most_streams_per_connection
        self._origins: MappingProxyType[_Origin, _OriginLanes] = MappingProxyType({})

    def for_origin(self, origin: _Origin) -> _OriginLanes:
        known: Final = self._origins.get(origin)
        if known is not None:
            return known
        created: Final = _OriginLanes(self._most_streams_per_connection, may_speak_http2=origin[0] == "https")
        self._origins = MappingProxyType({**self._origins, origin: created})
        return created

    def all_lanes(self) -> tuple[_Lane, ...]:
        return tuple(itertools.chain.from_iterable(origin.lanes for origin in self._origins.values()))

    def acquire(self, origin: _OriginLanes) -> _Lane:
        free: Final = origin.free_lane()
        lane: Final = free if free is not None else self._open_or_share(origin)
        lane.acquire()
        return lane

    def _room(self) -> int | None:
        cap: Final = self._limits.max_connections
        if cap is None:
            return None
        return cap - sum(origin.total_streams() for origin in self._origins.values())

    def _open_or_share(self, origin: _OriginLanes) -> _Lane:
        room: Final = self._room()
        if room is not None and room <= 0:
            return min(origin.lanes or self.all_lanes(), key=lambda lane: lane.in_flight)
        return self._open(origin, room)

    def _open(self, origin: _OriginLanes, room: int | None) -> _Lane:
        max_connections: Final = None if room is None else min(origin.streams_or_most(), room)
        return origin.open_lane(self._new_transport(self._lane_limits(max_connections)))

    def _lane_limits(self, max_connections: int | None) -> httpx.Limits:
        if max_connections is None:
            return self._limits
        return httpx.Limits(
            max_connections=max_connections,
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


def _headers_for_lane(request: httpx.Request) -> tuple[tuple[bytes, bytes], ...]:
    if request.url.scheme != "https":
        return tuple(request.headers.raw)
    return tuple((name, value) for name, value in request.headers.raw if _allowed_over_http2(name, value))


def _tracer(origin: _OriginLanes, caller_trace: object) -> _Trace:
    async def trace(event: str, info: Mapping[str, object]) -> None:
        origin.observe(event, info)
        if not callable(caller_trace):
            return
        outcome: Final = caller_trace(event, info)
        if inspect.isawaitable(outcome):
            await outcome

    return trace


def _for_lane(request: httpx.Request, origin: _OriginLanes) -> httpx.Request:
    caller_trace: Final[object] = request.extensions.get("trace")
    return httpx.Request(
        request.method,
        request.url,
        headers=_headers_for_lane(request),
        stream=request.stream,
        extensions={**request.extensions, "trace": _tracer(origin, caller_trace)},
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
        origin: Final = self._lanes.for_origin((request.url.scheme, request.url.host, request.url.port))
        probing: Final = await origin.take_turn()
        try:
            response: Final = await self._send_on_a_lane(request, origin)
        except BaseException:
            if probing:
                origin.end_probe(answered=False)
            raise
        if probing:
            origin.end_probe(answered=True)
        return response

    async def _send_on_a_lane(self, request: httpx.Request, origin: _OriginLanes) -> httpx.Response:
        lane: Final = self._lanes.acquire(origin)
        try:
            response: Final = await lane.transport.handle_async_request(_for_lane(request, origin))
        except BaseException:
            lane.release()
            raise
        stream: Final = response.stream
        if not isinstance(stream, httpx.AsyncByteStream):
            lane.release()
            return response
        return _with_stream(response, _AsyncReleasingStream(stream, lane.release))

    def has_request_in_flight(self) -> bool:
        return any(lane.in_flight > 0 for lane in self._lanes.all_lanes())

    async def aclose(self) -> None:
        for lane in self._lanes.all_lanes():
            await lane.transport.aclose()

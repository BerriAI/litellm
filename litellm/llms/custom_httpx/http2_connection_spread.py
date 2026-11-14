import asyncio
import inspect
import itertools
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import h2.events
import h2.settings
import httpcore
import httpx

HTTP2_STREAMS_PER_CONNECTION: Final = 100
HTTP2_CONNECTIONS_PER_LANE: Final = 2
_REMOTE_SETTINGS_RECEIVED: Final = "http2.receive_remote_settings.complete"
_HTTP1_REQUEST_STARTED: Final = "http11.send_request_headers.started"

_Origin: TypeAlias = tuple[str, str, int | None]
_Trace: TypeAlias = Callable[[str, Mapping[str, object]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _Multiplexed:
    streams: int


@dataclass(frozen=True, slots=True)
class _OneRequestPerConnection:
    pass


_Protocol: TypeAlias = _Multiplexed | _OneRequestPerConnection | None
_HTTP1: Final = _OneRequestPerConnection()


class _Lane:
    __slots__ = ("in_flight", "most_connections", "streams", "transport")

    def __init__(self, transport: httpx.AsyncBaseTransport, most_connections: int | None) -> None:
        self.transport: Final = transport
        self.most_connections: Final = most_connections
        self.in_flight = 0
        self.streams: int | None = None

    def learn_streams(self, settings: h2.events.RemoteSettingsChanged, most_streams_per_connection: int) -> int:
        advertised: Final = settings.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
        allowed: Final = 0 if advertised is None else min(advertised.new_value, most_streams_per_connection)
        self.streams = allowed or self.streams or 1
        return self.streams

    def acquire(self) -> None:
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1


def _is_tunnel_setup(info: Mapping[str, object]) -> bool:
    request: Final = info.get("request")
    return isinstance(request, httpcore.Request) and request.method == b"CONNECT"


def _least_busy(lanes: tuple[_Lane, ...]) -> _Lane:
    return min(lanes, key=lambda lane: lane.in_flight)


def _most_spare(lanes: tuple[_Lane, ...]) -> _Lane:
    return max(lanes, key=lambda lane: (lane.most_connections or 0) - lane.in_flight)


class _OriginLanes:
    def __init__(self, most_streams_per_connection: int, may_speak_http2: bool) -> None:
        self._most_streams_per_connection: Final = most_streams_per_connection
        self._probe_settled = asyncio.Event()
        self._probing = False
        self.lanes: tuple[_Lane, ...] = ()
        self.protocol: _Protocol = None if may_speak_http2 else _HTTP1

    def requests_on(self, lane: _Lane) -> int:
        match self.protocol:
            case _Multiplexed(streams=streams):
                return lane.streams or streams
            case _OneRequestPerConnection():
                return lane.most_connections or self._most_streams_per_connection
            case None:
                return 1

    def connections_for_a_new_lane(self) -> int:
        match self.protocol:
            case _OneRequestPerConnection():
                return self._most_streams_per_connection
            case _Multiplexed(streams=streams):
                return min(streams, HTTP2_CONNECTIONS_PER_LANE)
            case None:
                return HTTP2_CONNECTIONS_PER_LANE

    def open_lane(self, transport: httpx.AsyncBaseTransport, most_connections: int | None) -> _Lane:
        lane: Final = _Lane(transport, most_connections)
        self.lanes = (*self.lanes, lane)
        return lane

    def reserved(self) -> int:
        return sum(lane.most_connections or 0 for lane in self.lanes)

    def free_lane(self) -> _Lane | None:
        return next((lane for lane in self.lanes if lane.in_flight < self.requests_on(lane)), None)

    async def take_turn(self) -> bool:
        while self.protocol is None and self._probing:
            await self._probe_settled.wait()
        self._probing = self.protocol is None
        return self._probing

    def end_probe(self, answered: bool) -> None:
        if answered and self.protocol is None:
            self.protocol = _Multiplexed(self._most_streams_per_connection)
        self._probing = False
        self._wake_waiters()

    def observe(self, event: str, info: Mapping[str, object], lane: _Lane) -> None:
        settings: Final = info.get("return_value") if event == _REMOTE_SETTINGS_RECEIVED else None
        if isinstance(settings, h2.events.RemoteSettingsChanged):
            self.protocol = _Multiplexed(lane.learn_streams(settings, self._most_streams_per_connection))
            self._wake_waiters()
            return
        if event == _HTTP1_REQUEST_STARTED and self.protocol is None and not _is_tunnel_setup(info):
            self.protocol = _HTTP1
            self._wake_waiters()

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

    def tracked(self, origin: _Origin) -> _OriginLanes | None:
        known: Final = self._origins.get(origin)
        if known is not None:
            return known
        room: Final = self._room()
        if room is not None and room <= 0:
            return None
        created: Final = _OriginLanes(self._most_streams_per_connection, may_speak_http2=origin[0] == "https")
        self._origins = MappingProxyType({**self._origins, origin: created})
        return created

    def all_lanes(self) -> tuple[_Lane, ...]:
        return tuple(itertools.chain.from_iterable(origin.lanes for origin in self._origins.values()))

    def acquire(self, origin: _OriginLanes | None) -> _Lane:
        lane: Final = _most_spare(self.all_lanes()) if origin is None else self._free_open_or_shared(origin)
        lane.acquire()
        return lane

    def _free_open_or_shared(self, origin: _OriginLanes) -> _Lane:
        free: Final = origin.free_lane()
        if free is not None:
            return free
        room: Final = self._room()
        if room is not None and room <= 0:
            return _least_busy(origin.lanes) if origin.lanes else _most_spare(self.all_lanes())
        return self._open(origin, room)

    def _room(self) -> int | None:
        cap: Final = self._limits.max_connections
        if cap is None:
            return None
        return cap - sum(origin.reserved() for origin in self._origins.values())

    def _open(self, origin: _OriginLanes, room: int | None) -> _Lane:
        max_connections: Final = None if room is None else min(origin.connections_for_a_new_lane(), room)
        return origin.open_lane(self._new_transport(self._lane_limits(max_connections)), max_connections)

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


def _tracer(origin: _OriginLanes, lane: _Lane, caller_trace: object) -> _Trace:
    async def trace(event: str, info: Mapping[str, object]) -> None:
        origin.observe(event, info, lane)
        if not callable(caller_trace):
            return
        outcome: Final = caller_trace(event, info)
        if inspect.isawaitable(outcome):
            await outcome

    return trace


def _for_lane(request: httpx.Request, origin: _OriginLanes | None, lane: _Lane) -> httpx.Request:
    caller_trace: Final[object] = request.extensions.get("trace")
    return httpx.Request(
        request.method,
        request.url,
        headers=_headers_for_lane(request),
        stream=request.stream,
        extensions=request.extensions
        if origin is None
        else {**request.extensions, "trace": _tracer(origin, lane, caller_trace)},
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
        origin: Final = self._lanes.tracked((request.url.scheme, request.url.host, request.url.port))
        if origin is None:
            return await self._send_on_a_lane(request, None)
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

    async def _send_on_a_lane(self, request: httpx.Request, origin: _OriginLanes | None) -> httpx.Response:
        lane: Final = self._lanes.acquire(origin)
        try:
            response: Final = await lane.transport.handle_async_request(_for_lane(request, origin, lane))
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

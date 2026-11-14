import asyncio
import copy
import inspect
import itertools
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import h2.events
import h2.exceptions
import h2.settings
import httpcore
import httpx

HTTP2_STREAMS_PER_CONNECTION: Final = 100
HTTP2_CONNECTIONS_PER_LANE: Final = 2
_REMOTE_SETTINGS_RECEIVED: Final = "http2.receive_remote_settings.complete"
_HTTP1_REQUEST_STARTED: Final = "http11.send_request_headers.started"
_HTTP2_REQUEST_HEADERS_FAILED: Final = "http2.send_request_headers.failed"

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
    __slots__ = ("idle_since", "in_flight", "most_connections", "streams", "streams_left_open", "transport")

    def __init__(self, transport: httpx.AsyncBaseTransport, most_connections: int | None) -> None:
        self.transport: Final = transport
        self.most_connections: Final = most_connections
        self.in_flight = 0
        self.idle_since = time.monotonic()
        self.streams: int | None = None
        self.streams_left_open = 0

    def learn_streams(self, settings: h2.events.RemoteSettingsChanged, most_streams_per_connection: int) -> int:
        advertised: Final = settings.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
        allowed: Final = 0 if advertised is None else min(advertised.new_value, most_streams_per_connection)
        self.streams = allowed or self.streams or 1
        return self.streams

    def acquire(self) -> None:
        self.in_flight += 1

    def release(self, stream_lost: bool) -> None:
        self.in_flight -= 1
        self.streams_left_open += stream_lost
        if self.in_flight == 0:
            self.idle_since = time.monotonic()


def _is_tunnel_setup(info: Mapping[str, object]) -> bool:
    request: Final = info.get("request")
    return isinstance(request, httpcore.Request) and request.method == b"CONNECT"


def _is_refused_for_want_of_a_stream(event: str, info: Mapping[str, object]) -> bool:
    return event == _HTTP2_REQUEST_HEADERS_FAILED and isinstance(
        info.get("exception"), h2.exceptions.TooManyStreamsError
    )


def _least_busy(lanes: tuple[_Lane, ...]) -> _Lane:
    return min(lanes, key=lambda lane: lane.in_flight)


def _most_spare(lanes: tuple[_Lane, ...]) -> _Lane:
    return max(lanes, key=lambda lane: (lane.most_connections or 0) - lane.in_flight)


def _enough_to_free(lanes: tuple[_Lane, ...], connections: int) -> frozenset[_Lane]:
    freed_before: Final = itertools.accumulate((lane.most_connections or 0 for lane in lanes), initial=0)
    return frozenset(lane for lane, freed in zip(lanes, freed_before) if freed < connections)


def _expired(idle: tuple[_Lane, ...], keepalive_expiry: float | None, now: float) -> frozenset[_Lane]:
    if keepalive_expiry is None:
        return frozenset()
    return frozenset(lane for lane in idle if now - lane.idle_since > keepalive_expiry)


def _past_keepalive_limit(idle: tuple[_Lane, ...], max_keepalive_connections: int | None) -> frozenset[_Lane]:
    if max_keepalive_connections is None:
        return frozenset()
    newest_first: Final = sorted(idle, key=lambda lane: lane.idle_since, reverse=True)
    kept_through: Final = itertools.accumulate(lane.most_connections or 0 for lane in newest_first)
    return frozenset(lane for lane, kept in zip(newest_first, kept_through) if kept > max_keepalive_connections)


def _shared_with_waiters(failure: BaseException | None) -> httpx.TransportError | None:
    if isinstance(failure, (httpx.ConnectError, httpx.ConnectTimeout)) or not isinstance(failure, httpx.TransportError):
        return None
    return failure


def _raise_shared(failure: httpx.TransportError | None) -> None:
    if failure is not None:
        raise copy.copy(failure) from failure


def _pool_timeout(request: httpx.Request) -> float | None:
    timeouts: Final[object] = request.extensions.get("timeout")
    match timeouts:
        case {"pool": int() | float() as pool}:
            return float(pool)
        case _:
            return None


class _OriginLanes:
    def __init__(self, most_streams_per_connection: int, may_speak_http2: bool) -> None:
        self._most_streams_per_connection: Final = most_streams_per_connection
        self._probe: asyncio.Future[httpx.TransportError | None] | None = None
        self._requests = 0
        self.lanes: tuple[_Lane, ...] = ()
        self.protocol: _Protocol = None if may_speak_http2 else _HTTP1

    def enter(self) -> None:
        self._requests += 1

    def leave(self) -> None:
        self._requests -= 1

    def unused(self) -> bool:
        return not self.lanes and self._requests == 0

    def requests_on(self, lane: _Lane) -> int:
        match self.protocol:
            case _Multiplexed(streams=streams):
                return (lane.streams or streams) - lane.streams_left_open
            case _OneRequestPerConnection():
                return lane.most_connections or self._most_streams_per_connection
            case None:
                return 1

    def connections_for_a_new_lane(self) -> int:
        match self.protocol:
            case _Multiplexed(streams=streams):
                return min(streams, HTTP2_CONNECTIONS_PER_LANE)
            case _OneRequestPerConnection() | None:
                return HTTP2_CONNECTIONS_PER_LANE

    def open_lane(self, transport: httpx.AsyncBaseTransport, most_connections: int | None) -> _Lane:
        lane: Final = _Lane(transport, most_connections)
        self.lanes = (*self.lanes, lane)
        return lane

    def reserved(self) -> int:
        return sum(lane.most_connections or 0 for lane in self.lanes)

    def free_lane(self) -> _Lane | None:
        return next((lane for lane in self.lanes if lane.in_flight < self.requests_on(lane)), None)

    def usable_lanes(self) -> tuple[_Lane, ...]:
        return tuple(lane for lane in self.lanes if self.requests_on(lane) > 0)

    def spent_idle_lanes(self) -> tuple[_Lane, ...]:
        return tuple(lane for lane in self.lanes if lane.in_flight == 0 and self.requests_on(lane) <= 0)

    def drop(self, lanes: frozenset[_Lane]) -> None:
        self.lanes = tuple(lane for lane in self.lanes if lane not in lanes)

    async def take_turn(self) -> bool:
        while self.protocol is None and self._probe is not None:
            _raise_shared(await asyncio.shield(self._probe))
        if self.protocol is not None:
            return False
        self._probe = asyncio.get_running_loop().create_future()
        return True

    def end_probe(self, failure: BaseException | None) -> None:
        if failure is None and self.protocol is None:
            self.protocol = _Multiplexed(self._most_streams_per_connection)
        self._settle_probe(None if self.protocol is not None else _shared_with_waiters(failure))
        self._probe = None

    def observe(self, event: str, info: Mapping[str, object], lane: _Lane) -> None:
        settings: Final = info.get("return_value") if event == _REMOTE_SETTINGS_RECEIVED else None
        if isinstance(settings, h2.events.RemoteSettingsChanged):
            self.protocol = _Multiplexed(lane.learn_streams(settings, self._most_streams_per_connection))
            self._settle_probe(None)
            return
        if event == _HTTP1_REQUEST_STARTED and self.protocol is None and not _is_tunnel_setup(info):
            self.protocol = _HTTP1
            self._settle_probe(None)

    def _settle_probe(self, failure: httpx.TransportError | None) -> None:
        if self._probe is not None and not self._probe.done():
            self._probe.set_result(failure)


@dataclass(frozen=True, slots=True)
class _Placement:
    lane: _Lane
    origin: _OriginLanes | None


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
        self._closing: tuple[_Lane, ...] = ()
        self._freed: asyncio.Future[None] | None = None

    def enter(self, origin: _Origin) -> _OriginLanes | None:
        tracked: Final = self._tracked(origin)
        if tracked is not None:
            tracked.enter()
        return tracked

    def all_lanes(self) -> tuple[_Lane, ...]:
        return tuple(itertools.chain.from_iterable(origin.lanes for origin in self._origins.values()))

    def acquire(self, origin: _OriginLanes | None) -> _Placement | None:
        self._retire_idle_lanes()
        placement: Final = _Placement(_most_spare(self.all_lanes()), None) if origin is None else self._place(origin)
        if placement is not None:
            placement.lane.acquire()
        return placement

    def release(self, lane: _Lane, stream_lost: bool) -> None:
        lane.release(stream_lost)
        self._retire_idle_lanes()
        freed: Final = self._freed
        self._freed = None
        if freed is not None and not freed.done():
            freed.set_result(None)

    def freed(self) -> asyncio.Future[None]:
        if self._freed is None:
            self._freed = asyncio.get_running_loop().create_future()
        return self._freed

    async def close_dropped(self) -> None:
        for lane in self._closing:
            await lane.transport.aclose()
            self._closing = tuple(waiting for waiting in self._closing if waiting is not lane)

    def _tracked(self, origin: _Origin) -> _OriginLanes | None:
        known: Final = self._origins.get(origin)
        if known is not None and known.lanes:
            return known
        room: Final = self._make_room(HTTP2_CONNECTIONS_PER_LANE)
        if room is not None and room <= 0:
            return None
        if known is not None:
            return known
        created: Final = _OriginLanes(self._most_streams_per_connection, may_speak_http2=origin[0] == "https")
        self._origins = MappingProxyType({**self._origins, origin: created})
        return created

    def _place(self, origin: _OriginLanes) -> _Placement | None:
        free: Final = origin.free_lane()
        if free is not None:
            return _Placement(free, origin)
        wanted: Final = origin.connections_for_a_new_lane()
        room: Final = self._make_room(wanted)
        if room is None or room > 0:
            return _Placement(self._open(origin, None if room is None else min(wanted, room)), origin)
        if isinstance(origin.protocol, _OneRequestPerConnection):
            return None
        usable: Final = origin.usable_lanes()
        if usable:
            return _Placement(_least_busy(usable), origin)
        return _Placement(_most_spare(self.all_lanes()), None)

    def _retire_idle_lanes(self) -> None:
        spent: Final = frozenset(
            itertools.chain.from_iterable(known.spent_idle_lanes() for known in self._origins.values())
        )
        idle: Final = tuple(lane for lane in self.all_lanes() if lane.in_flight == 0 and lane not in spent)
        expired: Final = _expired(idle, self._limits.keepalive_expiry, time.monotonic())
        kept: Final = tuple(lane for lane in idle if lane not in expired)
        self._close_when_free(spent | expired | _past_keepalive_limit(kept, self._limits.max_keepalive_connections))

    def _make_room(self, connections: int) -> int | None:
        room: Final = self._room()
        if room is None or room >= connections:
            return room
        idle: Final = tuple(lane for lane in self.all_lanes() if lane.in_flight == 0)
        self._close_when_free(_enough_to_free(idle, connections - room))
        return self._room()

    def _close_when_free(self, lanes: frozenset[_Lane]) -> None:
        if not lanes:
            return
        for origin in self._origins.values():
            origin.drop(lanes)
        self._closing = (*self._closing, *lanes)
        self._origins = MappingProxyType({key: known for key, known in self._origins.items() if not known.unused()})

    def _room(self) -> int | None:
        cap: Final = self._limits.max_connections
        if cap is None:
            return None
        return cap - sum(origin.reserved() for origin in self._origins.values())

    def _open(self, origin: _OriginLanes, max_connections: int | None) -> _Lane:
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
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        release: Callable[[bool], None],
        close_dropped: Callable[[], Awaitable[None]],
    ) -> None:
        self._stream: Final = stream
        self._release: Final = release
        self._release_dropped: Final = weakref.finalize(self, release, True)
        self._close_dropped: Final = close_dropped

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except BaseException:
            self._release_once()
            raise

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            self._release_once()
        await self._close_dropped()

    def _release_once(self) -> None:
        if self._release_dropped.detach() is not None:
            self._release(False)


def _allowed_over_http2(name: bytes, value: bytes) -> bool:
    return name.lower() != b"te" or value.strip().lower() == b"trailers"


def _headers_for_lane(request: httpx.Request) -> tuple[tuple[bytes, bytes], ...]:
    if request.url.scheme != "https":
        return tuple(request.headers.raw)
    return tuple((name, value) for name, value in request.headers.raw if _allowed_over_http2(name, value))


def _tracer(origin: _OriginLanes, lane: _Lane, refused: asyncio.Event, caller_trace: object) -> _Trace:
    async def trace(event: str, info: Mapping[str, object]) -> None:
        origin.observe(event, info, lane)
        if _is_refused_for_want_of_a_stream(event, info):
            refused.set()
        if not callable(caller_trace):
            return
        outcome: Final = caller_trace(event, info)
        if inspect.isawaitable(outcome):
            await outcome

    return trace


def _for_lane(
    request: httpx.Request, origin: _OriginLanes | None, lane: _Lane, refused: asyncio.Event
) -> httpx.Request:
    caller_trace: Final[object] = request.extensions.get("trace")
    return httpx.Request(
        request.method,
        request.url,
        headers=_headers_for_lane(request),
        stream=request.stream,
        extensions=request.extensions
        if origin is None
        else {**request.extensions, "trace": _tracer(origin, lane, refused, caller_trace)},
    )


def _releaser(lanes: _Lanes, placement: _Placement) -> Callable[[bool], None]:
    return lambda stream_lost: lanes.release(placement.lane, stream_lost and placement.origin is not None)


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
        origin: Final = self._lanes.enter((request.url.scheme, request.url.host, request.url.port))
        if origin is None:
            return await self._send_on_a_lane(request, None)
        try:
            return await self._send_in_turn(request, origin)
        finally:
            origin.leave()

    async def _send_in_turn(self, request: httpx.Request, origin: _OriginLanes) -> httpx.Response:
        probing: Final = await origin.take_turn()
        try:
            response: Final = await self._send_on_a_lane(request, origin)
        except BaseException as failure:
            if probing:
                origin.end_probe(failure)
            raise
        if probing:
            origin.end_probe(None)
        return response

    async def _send_on_a_lane(self, request: httpx.Request, origin: _OriginLanes | None) -> httpx.Response:
        while (sent := await self._try_a_lane(request, origin)) is None:
            pass
        return sent

    async def _try_a_lane(self, request: httpx.Request, origin: _OriginLanes | None) -> httpx.Response | None:
        placement: Final = await self._placement(request, origin)
        lane: Final = placement.lane
        refused: Final = asyncio.Event()
        try:
            await self._lanes.close_dropped()
            response: Final = await lane.transport.handle_async_request(
                _for_lane(request, placement.origin, lane, refused)
            )
        except BaseException as failure:
            self._lanes.release(lane, stream_lost=refused.is_set())
            if refused.is_set() and isinstance(failure, httpx.LocalProtocolError):
                return None
            raise
        stream: Final = response.stream
        if not isinstance(stream, httpx.AsyncByteStream):
            self._lanes.release(lane, stream_lost=False)
            return response
        return _with_stream(
            response, _AsyncReleasingStream(stream, _releaser(self._lanes, placement), self._lanes.close_dropped)
        )

    async def _placement(self, request: httpx.Request, origin: _OriginLanes | None) -> _Placement:
        placement: Final = self._lanes.acquire(origin)
        if placement is not None:
            return placement
        try:
            return await asyncio.wait_for(self._placement_once_freed(origin), _pool_timeout(request))
        except asyncio.TimeoutError:
            raise httpx.PoolTimeout("No connection to this upstream freed up in time", request=request) from None

    async def _placement_once_freed(self, origin: _OriginLanes | None) -> _Placement:
        while (placement := self._lanes.acquire(origin)) is None:
            await asyncio.shield(self._lanes.freed())
        return placement

    def has_request_in_flight(self) -> bool:
        return any(lane.in_flight > 0 for lane in self._lanes.all_lanes())

    async def aclose(self) -> None:
        await self._lanes.close_dropped()
        for lane in self._lanes.all_lanes():
            await lane.transport.aclose()

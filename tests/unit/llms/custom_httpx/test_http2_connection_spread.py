import asyncio
import gc
import inspect
import tracemalloc
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final, cast

import h2.events
import h2.exceptions
import h2.settings
import httpcore
import httpx
import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.llms.custom_httpx.http2_connection_spread import (
    END_OF_STREAM_WAIT_SECONDS,
    HTTP2_CONNECTIONS_PER_LANE,
    HTTP2_STREAMS_PER_CONNECTION,
    AsyncHTTP2ConnectionSpread,
)

URL: Final = "https://search.example/v1/search"
OTHER_URL: Final = "https://other.example/v1/search"
BURST: Final = HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
UNCAPPED: Final = httpx.Limits(max_connections=None)
TWO_LANES: Final = httpx.Limits(max_connections=2 * HTTP2_CONNECTIONS_PER_LANE)
THREE_LANES: Final = httpx.Limits(max_connections=3 * HTTP2_CONNECTIONS_PER_LANE)
MORE_STREAMS_THAN_WE_RUN: Final = 2 * HTTP2_STREAMS_PER_CONNECTION
FEW_STREAMS: Final = 5

_Trace = Callable[[str, Mapping[str, object]], Awaitable[None]]


async def _emit(request: httpx.Request, event: str, info: Mapping[str, object]) -> None:
    trace: Final[object] = request.extensions.get("trace")
    if trace is not None:
        await cast(_Trace, trace)(event, info)


def _remote_settings(max_concurrent_streams: int | None) -> h2.events.RemoteSettingsChanged:
    advertised: Final = (
        {}
        if max_concurrent_streams is None
        else {h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: max_concurrent_streams}
    )
    return h2.events.RemoteSettingsChanged.from_settings(h2.settings.Settings(client=False), advertised)


async def _emit_settings(request: httpx.Request, streams: int | None) -> None:
    await _emit(request, "http2.receive_remote_settings.complete", {"return_value": _remote_settings(streams)})


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: tuple[str, ...] = ()

    async def __call__(self, event: str, _info: Mapping[str, object]) -> None:
        self.events = (*self.events, event)


def _spoken_over(request: httpx.Request) -> dict[str, bytes]:
    return {"http_version": b"HTTP/2" if request.url.scheme == "https" else b"HTTP/1.1"}


class _Body(httpx.AsyncByteStream):
    def __init__(self, connection: "_GatedAsyncConnection") -> None:
        self._connection: Final = connection
        self._read_to_its_end = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"ok"
        self._read_to_its_end = True

    async def aclose(self) -> None:
        if not self._read_to_its_end:
            self._connection.unread_ends += 1


class _GatedAsyncConnection(httpx.AsyncBaseTransport):
    def __init__(self, connections: "_AsyncConnections") -> None:
        self._connections: Final = connections
        self._settings_received = False
        self.streams: int | None = None
        self.unread_ends = 0
        self.hosts: frozenset[str] = frozenset()
        self.in_flight = 0
        self.peak = 0
        self.served = 0
        self.refused = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self._connections.refuse_if_due(request)
        if request.url.scheme == "https":
            await _emit(request, "http2.send_request_headers.started", {})
            await self._refuse_past_the_stream_limit(request)
        self.hosts = self.hosts | {request.url.netloc.decode()}
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self._learn_settings(request)
            await self._connections.gate.wait()
        except asyncio.CancelledError:
            self.in_flight -= 1
            self.unread_ends += 1
            raise
        self.in_flight -= 1
        self.unread_ends = 0
        self.served += 1
        return httpx.Response(200, stream=_Body(self), extensions=_spoken_over(request))

    async def _refuse_past_the_stream_limit(self, request: httpx.Request) -> None:
        if self.streams is None or self.in_flight + self.unread_ends < self.streams:
            return
        self.refused += 1
        refusal: Final = h2.exceptions.TooManyStreamsError(f"Max outbound streams is {self.streams}")
        await _emit(request, "http2.send_request_headers.failed", {"exception": refusal})
        raise httpx.LocalProtocolError(str(refusal), request=request)

    async def _learn_settings(self, request: httpx.Request) -> None:
        if self._settings_received:
            self.streams = await self._connections.resend_settings(request) or self.streams
            return
        self._settings_received = True
        self.streams = await self._connections.send_settings(request, self)

    async def aclose(self) -> None:
        await self._connections.closes.wait()
        self.closed = True


class _AsyncConnections:
    def __init__(
        self,
        gate: asyncio.Event,
        advertised: Mapping[str, int | None] = MappingProxyType({}),
        through_tunnel: bool = False,
        refusals: int = 0,
        refusal: type[httpx.TransportError] = httpx.ConnectError,
        reports_settings: bool = True,
        resends_settings: bool = False,
        first_frames: tuple[int, ...] = (),
    ) -> None:
        self.gate: Final = gate
        self.settings_arrive: Final = asyncio.Event()
        self.settings_arrive.set()
        self.closes: Final = asyncio.Event()
        self.closes.set()
        self.advertised: Final = advertised
        self.readvertised: Mapping[str, int | None] = advertised
        self._first_frames: Final = first_frames
        self._through_tunnel: Final = through_tunnel
        self._refusals_left = refusals
        self._refusal: Final = refusal
        self._reports_settings: Final = reports_settings
        self._resends_settings: Final = resends_settings
        self.opened: tuple[_GatedAsyncConnection, ...] = ()

    def __call__(self, _limits: httpx.Limits) -> _GatedAsyncConnection:
        connection: Final = _GatedAsyncConnection(self)
        self.opened = (*self.opened, connection)
        return connection

    async def refuse_if_due(self, request: httpx.Request) -> None:
        if self._refusals_left == 0:
            return
        self._refusals_left -= 1
        await asyncio.sleep(0)
        raise self._refusal("refused", request=request)

    async def send_settings(self, request: httpx.Request, connection: _GatedAsyncConnection) -> int | None:
        if not self._reports_settings or request.url.scheme == "http":
            return None
        if self._through_tunnel:
            tunnel: Final = httpcore.Request(b"CONNECT", str(request.url))
            await _emit(request, "http11.send_request_headers.started", {"request": tunnel})
        await self.settings_arrive.wait()
        index: Final = self.opened.index(connection)
        streams: Final = (
            self._first_frames[index]
            if index < len(self._first_frames)
            else self.advertised.get(request.url.netloc.decode(), MORE_STREAMS_THAN_WE_RUN)
        )
        await _emit_settings(request, streams)
        return streams

    async def resend_settings(self, request: httpx.Request) -> int | None:
        if not self._resends_settings:
            return None
        streams: Final = self.readvertised.get(request.url.netloc.decode(), MORE_STREAMS_THAN_WE_RUN)
        await _emit_settings(request, streams)
        return streams

    def in_flight(self) -> int:
        return sum(connection.in_flight for connection in self.opened)

    def peaks_serving(self, host: str) -> list[int]:
        return [connection.peak for connection in self.opened if host in connection.hosts]


class _RefusedConnection(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)


class _RefusedConnections:
    def __init__(self) -> None:
        self.opened = 0

    def __call__(self, _limits: httpx.Limits) -> _RefusedConnection:
        self.opened += 1
        return _RefusedConnection()


class _HTTP1Connections:
    def __init__(self, gate: asyncio.Event, gates_by_path: Mapping[str, asyncio.Event] = MappingProxyType({})) -> None:
        self._gate: Final = gate
        self._gates_by_path: Final = gates_by_path
        self.requested = 0
        self.open = 0
        self.most_open = 0
        self.pools = 0

    def __call__(self, limits: httpx.Limits) -> "_HTTP1Pool":
        assert limits.max_connections is not None
        self.pools += 1
        return _HTTP1Pool(self, asyncio.Semaphore(limits.max_connections))

    async def serve(self, request: httpx.Request, pool_slots: asyncio.Semaphore) -> None:
        self.requested += 1
        async with pool_slots:
            self.open += 1
            self.most_open = max(self.most_open, self.open)
            sent: Final = httpcore.Request(request.method, str(request.url))
            await _emit(request, "http11.send_request_headers.started", {"request": sent})
            await self._gates_by_path.get(request.url.path, self._gate).wait()
            self.open -= 1


class _HTTP1Pool(httpx.AsyncBaseTransport):
    def __init__(self, connections: _HTTP1Connections, slots: asyncio.Semaphore) -> None:
        self._connections: Final = connections
        self._slots: Final = slots

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self._connections.serve(request, self._slots)
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))


class _MixedConnections:
    def __init__(self, gate: asyncio.Event, http2_streams: Mapping[str, int]) -> None:
        self._gate: Final = gate
        self._http2_streams: Final = http2_streams
        self.requested = 0
        self.open = 0
        self.most_open = 0

    def __call__(self, limits: httpx.Limits) -> "_MixedPool":
        assert limits.max_connections is not None
        return _MixedPool(self, asyncio.Semaphore(limits.max_connections))

    async def serve(self, request: httpx.Request, pool_slots: asyncio.Semaphore) -> None:
        self.requested += 1
        streams: Final = self._http2_streams.get(request.url.netloc.decode())
        async with pool_slots:
            self.open += 1
            self.most_open = max(self.most_open, self.open)
            if streams is None:
                sent: Final = httpcore.Request(request.method, str(request.url))
                await _emit(request, "http11.send_request_headers.started", {"request": sent})
                await self._gate.wait()
            else:
                await _emit(request, "http2.receive_remote_settings.complete", {"return_value": _remote_settings(streams)})
            self.open -= 1


class _MixedPool(httpx.AsyncBaseTransport):
    def __init__(self, connections: _MixedConnections, slots: asyncio.Semaphore) -> None:
        self._connections: Final = connections
        self._slots: Final = slots

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self._connections.serve(request, self._slots)
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0)
    pytest.fail("condition never held")


@pytest.mark.asyncio
async def test_async_spread_opens_another_connection_only_once_streams_run_out():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(BURST))
        await _until(lambda: connections.in_flight() == BURST)
        assert [connection.peak for connection in connections.opened] == [
            HTTP2_STREAMS_PER_CONNECTION,
            BURST - HTTP2_STREAMS_PER_CONNECTION,
        ]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}
        assert (await client.get(URL)).text == "ok"
        assert [connection.served for connection in connections.opened] == [
            HTTP2_STREAMS_PER_CONNECTION + 1,
            BURST - HTTP2_STREAMS_PER_CONNECTION,
        ]
    assert all(connection.closed for connection in connections.opened)


@pytest.mark.parametrize(
    "limits", [UNCAPPED, httpx.Limits(max_connections=HTTP2_STREAMS_PER_CONNECTION)], ids=["uncapped", "capped"]
)
@pytest.mark.parametrize(
    ("advertised", "through_tunnel", "streams"),
    [(FEW_STREAMS, False, FEW_STREAMS), (FEW_STREAMS, True, FEW_STREAMS), (None, False, 1)],
    ids=["few-streams", "few-streams-through-proxy-tunnel", "no-stream-limit-sent"],
)
@pytest.mark.asyncio
async def test_async_spread_opens_another_connection_at_the_upstreams_stream_limit(
    advertised: int | None, through_tunnel: bool, streams: int, limits: httpx.Limits
):
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(
        gate, advertised=MappingProxyType({"search.example": advertised}), through_tunnel=through_tunnel
    )
    connections.settings_arrive.clear()
    burst: Final = 4 * streams
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, limits)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == 1)

        connections.settings_arrive.set()
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [streams] * 4

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_async_spread_learns_the_stream_limit_from_the_next_request_when_the_first_cannot_connect(
    refusal: type[httpx.TransportError],
):
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(
        gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}), refusals=1, refusal=refusal
    )
    connections.settings_arrive.clear()
    burst: Final = 4 * FEW_STREAMS
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        refused: Final = asyncio.create_task(client.get(URL))
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: refused.done() and connections.in_flight() == 1)

        connections.settings_arrive.set()
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [FEW_STREAMS] * 4

        gate.set()
        with pytest.raises(refusal):
            await refused
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


class _OpensAfterOneFailure:
    def __init__(self, connections: _AsyncConnections) -> None:
        self._connections: Final = connections
        self._failed = False

    def __call__(self, limits: httpx.Limits) -> _GatedAsyncConnection:
        if not self._failed:
            self._failed = True
            raise ImportError("socksio is not installed")
        return self._connections(limits)


@pytest.mark.asyncio
async def test_async_spread_lets_the_next_request_through_when_opening_a_connection_fails():
    gate: Final = asyncio.Event()
    gate.set()
    spread: Final = AsyncHTTP2ConnectionSpread(_OpensAfterOneFailure(_AsyncConnections(gate)), UNCAPPED)
    async with httpx.AsyncClient(transport=spread) as client:
        failed, served = await asyncio.wait_for(
            asyncio.gather(client.get(URL), client.get(URL), return_exceptions=True), timeout=5
        )
    assert isinstance(failed, ImportError)
    assert isinstance(served, httpx.Response) and served.status_code == 200


@pytest.mark.asyncio
async def test_async_spread_counts_and_closes_the_connection_a_retry_opens_at_the_cap_after_the_first_failed():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    one_lane: Final = httpx.Limits(max_connections=HTTP2_CONNECTIONS_PER_LANE)
    spread: Final = AsyncHTTP2ConnectionSpread(_OpensAfterOneFailure(connections), one_lane)
    async with httpx.AsyncClient(transport=spread) as client:
        with pytest.raises(ImportError):
            await client.get(URL)
        assert (await client.get(OTHER_URL)).text == "ok"

        gate.clear()
        retry: Final = asyncio.create_task(client.get(URL))
        await _until(lambda: connections.in_flight() == 1)
        other: Final = asyncio.create_task(client.get(OTHER_URL))
        await _until(lambda: connections.in_flight() == 2)
        assert len(connections.opened) == 2

        gate.set()
        assert [response.text for response in await asyncio.gather(retry, other)] == ["ok", "ok"]
    assert all(connection.closed for connection in connections.opened)


@pytest.mark.asyncio
async def test_async_spread_raises_a_pool_timeout_when_the_first_connection_to_an_upstream_does_not_open_in_time():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    connections.settings_arrive.clear()
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        first: Final = asyncio.create_task(client.get(URL))
        await _until(lambda: connections.in_flight() == 1)

        with pytest.raises(httpx.PoolTimeout):
            await asyncio.wait_for(client.get(URL, timeout=httpx.Timeout(None, pool=0.05)), 1)

        connections.settings_arrive.set()
        assert (await first).text == "ok"
        assert len(connections.opened) == 1


@pytest.mark.asyncio
async def test_async_spread_fails_the_waiting_requests_with_the_first_requests_error_when_it_breaks_after_connecting():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate, refusals=1, refusal=httpx.RemoteProtocolError)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        outcomes: Final = await asyncio.gather(*(client.get(URL) for _ in range(FEW_STREAMS)), return_exceptions=True)
        assert all(isinstance(outcome, httpx.RemoteProtocolError) for outcome in outcomes)
        assert len(connections.opened) == 1

        assert (await client.get(URL)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("later_advertised", "streams"),
    [(None, FEW_STREAMS), (2 * FEW_STREAMS, 2 * FEW_STREAMS)],
    ids=["later-frame-leaves-the-limit-out", "later-frame-raises-the-limit"],
)
async def test_async_spread_follows_the_stream_limit_of_the_upstreams_latest_settings_frame(
    later_advertised: int | None, streams: int
):
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(
        gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}), resends_settings=True
    )
    burst: Final = 2 * 2 * FEW_STREAMS
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        assert (await client.get(URL)).status_code == 200
        connections.readvertised = MappingProxyType({"search.example": later_advertised})
        gate.clear()

        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [streams] * (burst // streams)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_keeps_each_connection_within_its_own_stream_limit_when_one_upstreams_connections_differ():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, first_frames=(FEW_STREAMS, MORE_STREAMS_THAN_WE_RUN))
    connections.settings_arrive.clear()
    burst: Final = 4 * FEW_STREAMS
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == 1)

        connections.settings_arrive.set()
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [FEW_STREAMS, burst - FEW_STREAMS]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("other_url", "other_netloc"),
    [(OTHER_URL, "other.example"), ("https://search.example:8443/v1/search", "search.example:8443")],
    ids=["other-host", "same-host-other-port"],
)
async def test_async_spread_learns_each_upstreams_stream_limit_on_its_own(other_url: str, other_netloc: str):
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}))
    burst: Final = 4 * FEW_STREAMS
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(url)) for url in (URL, other_url) * burst)
        await _until(lambda: connections.in_flight() == 2 * burst)
        assert connections.peaks_serving("search.example") == [FEW_STREAMS] * 4
        assert connections.peaks_serving(other_netloc) == [burst]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_keeps_each_upstream_on_its_own_connections_once_the_connection_cap_is_reached():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    overflow: Final = HTTP2_STREAMS_PER_CONNECTION // 10
    urls: Final = (URL,) * HTTP2_STREAMS_PER_CONNECTION + (OTHER_URL,) * overflow + (URL,) * overflow
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, TWO_LANES)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(url)) for url in urls)
        await _until(lambda: connections.in_flight() == len(urls))
        assert [connection.hosts for connection in connections.opened] == [
            frozenset({"search.example"}),
            frozenset({"other.example"}),
        ]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_keeps_tracking_new_upstreams_while_a_plain_http_upstream_has_a_request_running():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}))
    burst: Final = 4 * FEW_STREAMS
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=HTTP2_STREAMS_PER_CONNECTION))
    ) as client:
        plain: Final = asyncio.create_task(client.get("http://plain.example/v1/search"))
        await _until(lambda: connections.in_flight() == 1)

        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == 1 + burst)
        assert connections.peaks_serving("search.example") == [FEW_STREAMS] * 4

        gate.set()
        assert {response.status_code for response in await asyncio.gather(plain, *pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_runs_the_most_streams_once_a_transport_answers_without_reporting_a_limit():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate, reports_settings=False)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        assert (await client.get(URL)).status_code == 200
        gate.clear()
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(BURST))
        await _until(lambda: connections.in_flight() == BURST)
        assert [connection.peak for connection in connections.opened] == [
            HTTP2_STREAMS_PER_CONNECTION,
            BURST - HTTP2_STREAMS_PER_CONNECTION,
        ]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_passes_connection_events_on_to_the_callers_own_trace():
    gate: Final = asyncio.Event()
    gate.set()
    trace: Final = _TraceRecorder()
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(_AsyncConnections(gate), UNCAPPED)) as client:
        assert (await client.get(URL, extensions={"trace": trace})).status_code == 200
    assert trace.events == ("http2.send_request_headers.started", "http2.receive_remote_settings.complete")


class _TeHeaderRecorder(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.te_values: tuple[str | None, ...] = ()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.te_values = (*self.te_values, request.headers.get("te"))
        return httpx.Response(200)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "sent_te", "forwarded_te"),
    [
        ("http://search.example/v1/search", "gzip", "gzip"),
        (URL, "gzip", None),
        (URL, "trailers", "trailers"),
    ],
    ids=["plain-http-keeps-it", "https-drops-it", "https-keeps-trailers"],
)
async def test_async_spread_drops_a_te_header_only_where_http2_refuses_it(
    url: str, sent_te: str, forwarded_te: str | None
):
    recorder: Final = _TeHeaderRecorder()
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: recorder, UNCAPPED)) as client:
        assert (await client.get(url, headers={"te": sent_te})).status_code == 200
    assert recorder.te_values == (forwarded_te,)


@pytest.mark.asyncio
async def test_async_spread_sends_a_plain_http_burst_at_once_without_waiting_for_a_stream_limit():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, reports_settings=False)
    burst: Final = 2 * FEW_STREAMS
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        pending: Final = tuple(
            asyncio.create_task(client.get("http://search.example/v1/search")) for _ in range(burst)
        )
        await _until(lambda: connections.in_flight() == burst)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_shares_the_least_busy_connection_once_an_upstream_has_its_share_of_the_cap():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    burst: Final = 2 * HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, THREE_LANES)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [burst // 2, burst // 2]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_keeps_http1_connections_within_the_connection_cap():
    gate: Final = asyncio.Event()
    connections: Final = _HTTP1Connections(gate)
    cap: Final = HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
    burst: Final = 2 * cap
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=cap))
    ) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.open == cap - HTTP2_CONNECTIONS_PER_LANE)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}
    assert connections.most_open == cap - HTTP2_CONNECTIONS_PER_LANE


@pytest.mark.asyncio
async def test_async_spread_keeps_connections_for_another_upstream_while_one_upstream_has_requests_waiting():
    slow: Final = asyncio.Event()
    fast: Final = asyncio.Event()
    fast.set()
    connections: Final = _HTTP1Connections(fast, gates_by_path=MappingProxyType({"/slow": slow}))
    cap: Final = 3 * HTTP2_CONNECTIONS_PER_LANE
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=cap))
    ) as client:
        held: Final = tuple(asyncio.create_task(client.get("http://busy.example/slow")) for _ in range(cap))
        await _until(lambda: connections.requested == cap - HTTP2_CONNECTIONS_PER_LANE)

        for _ in range(2):
            assert (await asyncio.wait_for(client.get("http://other.example/fast"), 1)).status_code == 200
        assert connections.open == cap - HTTP2_CONNECTIONS_PER_LANE

        slow.set()
        assert {response.status_code for response in await asyncio.gather(*held)} == {200}


@pytest.mark.asyncio
async def test_async_spread_keeps_another_upstreams_idle_connections_when_closing_them_lets_no_request_through():
    slow: Final = asyncio.Event()
    fast: Final = asyncio.Event()
    fast.set()
    connections: Final = _HTTP1Connections(fast, gates_by_path=MappingProxyType({"/slow": slow}))
    share: Final = 2 * HTTP2_CONNECTIONS_PER_LANE
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, THREE_LANES)) as client:
        held: Final = tuple(asyncio.create_task(client.get("http://busy.example/slow")) for _ in range(share))
        await _until(lambda: connections.requested == share)
        assert (await client.get("http://other.example/fast")).status_code == 200
        pools_before: Final = connections.pools

        waiting: Final = asyncio.create_task(client.get("http://busy.example/slow"))
        await asyncio.sleep(0)
        assert (await client.get("http://other.example/fast")).status_code == 200
        assert connections.pools == pools_before

        slow.set()
        assert {response.status_code for response in await asyncio.gather(*held, waiting)} == {200}


@pytest.mark.asyncio
async def test_async_spread_gives_a_plain_http_request_at_the_connection_cap_the_first_connection_that_frees():
    slow: Final = asyncio.Event()
    fast: Final = asyncio.Event()
    connections: Final = _HTTP1Connections(fast, MappingProxyType({"/slow": slow}))
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, THREE_LANES)) as client:
        slow_calls: Final = tuple(
            asyncio.create_task(client.get("http://plain.example/slow")) for _ in range(HTTP2_CONNECTIONS_PER_LANE)
        )
        await _until(lambda: connections.open == HTTP2_CONNECTIONS_PER_LANE)
        fast_calls: Final = tuple(
            asyncio.create_task(client.get("http://plain.example/fast")) for _ in range(HTTP2_CONNECTIONS_PER_LANE + 1)
        )
        await _until(lambda: connections.open == 2 * HTTP2_CONNECTIONS_PER_LANE)

        fast.set()
        await _until(lambda: all(call.done() for call in fast_calls))
        assert not any(call.done() for call in slow_calls)

        slow.set()
        assert {response.status_code for response in await asyncio.gather(*slow_calls, *fast_calls)} == {200}


@pytest.mark.asyncio
async def test_async_spread_raises_a_pool_timeout_when_no_connection_frees_in_time_at_the_connection_cap():
    gate: Final = asyncio.Event()
    connections: Final = _HTTP1Connections(gate)
    share: Final = 2 * HTTP2_CONNECTIONS_PER_LANE
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, THREE_LANES)) as client:
        holding: Final = tuple(asyncio.create_task(client.get("http://plain.example/x")) for _ in range(share))
        await _until(lambda: connections.open == share)
        patient: Final = asyncio.create_task(client.get("http://plain.example/x"))

        with pytest.raises(httpx.PoolTimeout):
            await asyncio.wait_for(client.get("http://plain.example/x", timeout=httpx.Timeout(None, pool=0.05)), 1)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*holding, patient)} == {200}
        assert (await client.get("http://plain.example/x")).status_code == 200


@pytest.mark.asyncio
async def test_async_spread_hands_an_idle_upstreams_connections_to_another_upstream_at_the_connection_cap():
    gate: Final = asyncio.Event()
    connections: Final = _MixedConnections(gate, MappingProxyType({"search.example": FEW_STREAMS}))
    cap: Final = 3 * HTTP2_CONNECTIONS_PER_LANE
    first_burst: Final = 2 * FEW_STREAMS
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=cap))
    ) as client:
        streams: Final = await asyncio.gather(
            *(client.send(client.build_request("GET", URL), stream=True) for _ in range(first_burst))
        )
        for response in streams:
            await response.aread()

        pending: Final = tuple(asyncio.create_task(client.get("http://plain.example/x.png")) for _ in range(cap))
        await _until(lambda: connections.requested == first_burst + cap - HTTP2_CONNECTIONS_PER_LANE)
        assert connections.open == cap - HTTP2_CONNECTIONS_PER_LANE

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_closes_a_dropped_connection_even_when_the_request_closing_it_is_cancelled():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, TWO_LANES)) as client:
        for host in ("first.example", "second.example"):
            assert (await client.get(f"https://{host}/v1/search")).text == "ok"

        connections.closes.clear()
        cancelled: Final = asyncio.create_task(client.get(OTHER_URL))
        await _until(lambda: len(connections.opened) == 3)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        connections.closes.set()
    assert all(connection.closed for connection in connections.opened)


@pytest.mark.asyncio
async def test_async_spread_keeps_memory_flat_while_a_capped_client_reaches_new_upstreams():
    gate: Final = asyncio.Event()
    gate.set()
    upstreams: Final = 200
    spread_source: Final = tracemalloc.Filter(True, inspect.getfile(AsyncHTTP2ConnectionSpread))
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(_AsyncConnections(gate), TWO_LANES)) as client:
        tracemalloc.start()
        try:
            for index in range(upstreams):
                assert (await client.get(f"https://upstream-{index}.example/v1/search")).text == "ok"
            gc.collect()
            retained: Final = tracemalloc.take_snapshot().filter_traces((spread_source,))
        finally:
            tracemalloc.stop()
    assert sum(stat.size for stat in retained.statistics("filename")) < 16 * upstreams


@pytest.mark.asyncio
async def test_async_spread_closes_connections_left_idle_past_the_keepalive_expiry():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    expiry: Final = 0.05
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=None, keepalive_expiry=expiry))
    ) as client:
        busy_past_the_expiry: Final = asyncio.create_task(client.get(URL))
        await asyncio.sleep(2 * expiry)
        gate.set()
        assert (await busy_past_the_expiry).text == "ok"
        assert (await client.get(URL)).text == "ok"
        assert len(connections.opened) == 1

        await asyncio.sleep(2 * expiry)

        assert (await client.get(OTHER_URL)).text == "ok"
        assert [connection.closed for connection in connections.opened] == [True, False]


@pytest.mark.asyncio
async def test_async_spread_keeps_no_more_idle_connections_than_the_keepalive_limit_after_a_burst():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    lanes: Final = 4
    burst: Final = lanes * HTTP2_CONNECTIONS_PER_LANE
    limits: Final = httpx.Limits(
        max_connections=burst + HTTP2_CONNECTIONS_PER_LANE, max_keepalive_connections=HTTP2_CONNECTIONS_PER_LANE
    )
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, limits)) as client:
        pending: Final = tuple(asyncio.create_task(client.get("http://plain.example/v1/search")) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}
        assert len(connections.opened) == lanes
        assert [connection.closed for connection in connections.opened].count(False) == 1


@pytest.mark.asyncio
async def test_async_spread_keeps_as_many_idle_http2_connections_as_the_keepalive_limit_after_a_burst():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}))
    lanes: Final = 4
    burst: Final = lanes * FEW_STREAMS
    limits: Final = httpx.Limits(
        max_connections=(lanes + 1) * HTTP2_CONNECTIONS_PER_LANE, max_keepalive_connections=lanes
    )
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, limits)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}
        assert len(connections.opened) == lanes
        assert [connection.closed for connection in connections.opened] == [False] * lanes


@pytest.mark.asyncio
async def test_async_spread_keeps_its_one_idle_connection_when_the_keepalive_limit_is_one():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    limits: Final = httpx.Limits(max_connections=2 * HTTP2_CONNECTIONS_PER_LANE, max_keepalive_connections=1)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, limits)) as client:
        for _ in range(3):
            assert (await client.get(URL)).text == "ok"
        assert len(connections.opened) == 1
        assert not connections.opened[0].closed


@pytest.mark.asyncio
async def test_async_spread_keeps_connections_within_the_cap_after_upstreams_report_low_stream_limits():
    gate: Final = asyncio.Event()
    single_stream_hosts: Final = tuple(f"single-stream-{index}.example" for index in range(4))
    connections: Final = _MixedConnections(gate, MappingProxyType({host: 1 for host in single_stream_hosts}))
    cap: Final = len(single_stream_hosts)
    burst: Final = 3 * cap
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=cap))
    ) as client:
        for host in single_stream_hosts:
            assert (await client.get(f"https://{host}/v1/search")).status_code == 200

        pending: Final = tuple(asyncio.create_task(client.get("https://http1.example/x.png")) for _ in range(burst))
        await _until(lambda: connections.open == cap - HTTP2_CONNECTIONS_PER_LANE)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}
    assert connections.most_open == cap - HTTP2_CONNECTIONS_PER_LANE


@pytest.mark.asyncio
async def test_async_spread_runs_a_request_on_every_connection_the_cap_leaves_when_the_upstream_allows_one_stream():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": 1}))
    cap: Final = 2 * FEW_STREAMS
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=cap))
    ) as client:
        first: Final = asyncio.create_task(client.get(URL))
        await _until(lambda: connections.in_flight() == 1)

        after_the_first: Final = cap - 2 * HTTP2_CONNECTIONS_PER_LANE
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(after_the_first))
        await _until(lambda: connections.in_flight() == 1 + after_the_first)
        assert [connection.peak for connection in connections.opened] == [1] * (1 + after_the_first)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(first, *pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_runs_another_upstreams_burst_at_once_after_the_first_upstream_reports_many_streams():
    gate: Final = asyncio.Event()
    connections: Final = _MixedConnections(gate, MappingProxyType({"search.example": MORE_STREAMS_THAN_WE_RUN}))
    burst: Final = 2 * FEW_STREAMS
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, httpx.Limits(max_connections=HTTP2_STREAMS_PER_CONNECTION))
    ) as client:
        assert (await client.get(URL)).status_code == 200

        pending: Final = tuple(asyncio.create_task(client.get("https://http1.example/x.png")) for _ in range(burst))
        await _until(lambda: connections.open == burst)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_sends_an_upstream_first_seen_at_the_cap_to_the_connections_with_the_most_room():
    gate: Final = asyncio.Event()
    connections: Final = _MixedConnections(gate, MappingProxyType({"search.example": MORE_STREAMS_THAN_WE_RUN}))
    filling: Final = 2 * HTTP2_CONNECTIONS_PER_LANE - 1
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, TWO_LANES)) as client:
        filled: Final = (
            *(asyncio.create_task(client.get("http://http1.example/x.png")) for _ in range(HTTP2_CONNECTIONS_PER_LANE)),
            asyncio.create_task(client.get("http://second.example/x.png")),
        )
        await _until(lambda: connections.open == filling)

        first_seen: Final = tuple(asyncio.create_task(client.get("http://other.example/x.png")) for _ in range(2))
        await _until(lambda: connections.open == filling + 1)

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*filled, *first_seen)} == {200}


@pytest.mark.asyncio
async def test_async_spread_sends_a_new_upstreams_burst_at_once_when_the_connection_cap_is_already_reached():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    filling: Final = 2 * HTTP2_STREAMS_PER_CONNECTION
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, TWO_LANES)) as client:
        two_upstreams: Final = (URL, "https://filling.example/v1/search") * HTTP2_STREAMS_PER_CONNECTION
        pending: Final = tuple(asyncio.create_task(client.get(url)) for url in two_upstreams)
        await _until(lambda: connections.in_flight() == filling)

        connections.settings_arrive.clear()
        first_seen: Final = tuple(asyncio.create_task(client.get(OTHER_URL)) for _ in range(FEW_STREAMS))
        await _until(lambda: connections.in_flight() == filling + FEW_STREAMS)
        assert len(connections.opened) == 2

        gate.set()
        connections.settings_arrive.set()
        assert {response.status_code for response in await asyncio.gather(*pending, *first_seen)} == {200}


@pytest.mark.asyncio
async def test_async_spread_holds_a_stream_until_its_response_is_closed():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        open_responses: Final = tuple(
            [
                await client.send(client.build_request("GET", URL), stream=True)
                for _ in range(HTTP2_STREAMS_PER_CONNECTION)
            ]
        )
        await client.get(URL)
        assert len(connections.opened) == 2

        for response in open_responses:
            await response.aread()
        await client.get(URL)
        assert [connection.served for connection in connections.opened] == [HTTP2_STREAMS_PER_CONNECTION + 1, 1]


@pytest.mark.asyncio
async def test_async_spread_moves_off_a_connection_whose_only_stream_was_cancelled_while_awaiting_its_response():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": 1}))
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        cancelled: Final = asyncio.create_task(client.get(URL))
        await _until(lambda: connections.in_flight() == 1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert connections.opened[0].closed

        gate.set()
        assert (await client.get(URL)).text == "ok"
        assert [connection.served for connection in connections.opened] == [0, 1]


class _BodyGivenUpAfterItsFirstChunk(httpx.AsyncByteStream):
    def __init__(self, after_first_chunk: Callable[[], Awaitable[None]]) -> None:
        self._after_first_chunk: Final = after_first_chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"o"
        await self._after_first_chunk()
        yield b"k"


class _OneStreamConnection(httpx.AsyncBaseTransport):
    def __init__(self, after_first_chunk: Callable[[], Awaitable[None]]) -> None:
        self._after_first_chunk: Final = after_first_chunk
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await _emit(request, "http2.send_request_headers.started", {})
        await _emit_settings(request, 1)
        return httpx.Response(
            200, stream=_BodyGivenUpAfterItsFirstChunk(self._after_first_chunk), extensions=_spoken_over(request)
        )

    async def aclose(self) -> None:
        self.closed = True


class _BodyClosedUnderItsReader(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self._closed: Final = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"o"
        await self._closed.wait()
        raise httpx.RemoteProtocolError("stream closed while it was read")

    async def aclose(self) -> None:
        self._closed.set()


class _UnendingStreamConnection(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await _emit(request, "http2.send_request_headers.started", {})
        await _emit_settings(request, 1)
        return httpx.Response(200, stream=_BodyClosedUnderItsReader(), extensions=_spoken_over(request))

    async def aclose(self) -> None:
        self.closed = True


async def _never_answer() -> None:
    await asyncio.Event().wait()


async def _time_out() -> None:
    raise httpx.ReadTimeout("no chunk in time")


async def _end_shortly() -> None:
    await asyncio.sleep(END_OF_STREAM_WAIT_SECONDS / 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(("rest_of_body", "retired"), [(_never_answer, True), (_end_shortly, False), (_time_out, False)])
async def test_async_spread_retires_a_one_stream_connection_closed_early_only_while_its_response_is_still_arriving(
    rest_of_body: Callable[[], Awaitable[None]], retired: bool
):
    connection: Final = _OneStreamConnection(rest_of_body)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        async with client.stream("GET", URL) as response:
            chunks: Final = response.aiter_raw()
            assert await anext(chunks) == b"o"
        assert connection.closed is retired


@pytest.mark.asyncio
async def test_async_spread_keeps_the_connection_when_a_read_is_abandoned_just_before_its_response_is_closed():
    connection: Final = _OneStreamConnection(_end_shortly)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        async with client.stream("GET", URL) as response:
            async for _chunk in response.aiter_raw():
                break
        assert not connection.closed


@pytest.mark.asyncio
async def test_async_spread_retires_a_one_stream_connection_when_closing_its_response_is_cancelled_before_its_end():
    connection: Final = _OneStreamConnection(_end_shortly)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        response: Final = await client.send(client.build_request("GET", URL), stream=True)
        chunks: Final = response.aiter_raw()
        assert await anext(chunks) == b"o"
        closing: Final = asyncio.create_task(response.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert connection.closed


@pytest.mark.asyncio
async def test_async_spread_closes_a_response_another_task_is_still_reading():
    connection: Final = _UnendingStreamConnection()
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        response: Final = await client.send(client.build_request("GET", URL), stream=True)
        chunks: Final = response.aiter_raw()
        assert await anext(chunks) == b"o"
        reading: Final = asyncio.create_task(anext(chunks))
        await asyncio.sleep(0)

        await response.aclose()

        with pytest.raises(httpx.RemoteProtocolError):
            await reading
        assert connection.closed


@pytest.mark.asyncio
async def test_async_spread_closes_a_plain_http_response_early_without_waiting_for_its_end():
    connection: Final = _OneStreamConnection(_never_answer)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        response: Final = await client.send(client.build_request("GET", "http://search.example/v1/search"), stream=True)
        chunks: Final = response.aiter_raw()
        assert await anext(chunks) == b"o"
        await asyncio.wait_for(response.aclose(), END_OF_STREAM_WAIT_SECONDS / 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after_first_chunk", "given_up"), [(_never_answer, asyncio.TimeoutError), (_time_out, httpx.ReadTimeout)]
)
async def test_async_spread_closes_a_one_stream_connection_once_a_read_of_its_response_is_given_up(
    after_first_chunk: Callable[[], Awaitable[None]], given_up: type[BaseException]
):
    connection: Final = _OneStreamConnection(after_first_chunk)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED)) as client:
        async with client.stream("GET", URL) as response:
            chunks: Final = response.aiter_raw()
            assert await anext(chunks) == b"o"
            with pytest.raises(given_up):
                await asyncio.wait_for(anext(chunks), 0.05)
            assert connection.closed


@pytest.mark.asyncio
async def test_async_spread_stays_on_one_connection_when_streams_are_closed_before_their_end_one_after_another():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}))
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        for _ in range(3 * FEW_STREAMS):
            async with client.stream("POST", URL) as response:
                assert response.status_code == 200
        assert len(connections.opened) == 1


@pytest.mark.asyncio
async def test_async_spread_frees_the_stream_when_the_request_fails():
    connections: Final = _RefusedConnections()
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        for _ in range(BURST):
            with pytest.raises(httpx.ConnectError):
                await client.get(URL)
    assert connections.opened == 1


async def _open_streams_without_closing(client: httpx.AsyncClient, count: int) -> int:
    responses: Final = tuple([await client.send(client.build_request("GET", URL), stream=True) for _ in range(count)])
    return sum(not response.is_closed for response in responses)


@pytest.mark.asyncio
async def test_async_spread_frees_the_streams_of_responses_dropped_without_closing():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    spread: Final = AsyncHTTP2ConnectionSpread(connections, UNCAPPED)
    async with httpx.AsyncClient(transport=spread) as client:
        assert await _open_streams_without_closing(client, HTTP2_STREAMS_PER_CONNECTION) == HTTP2_STREAMS_PER_CONNECTION
        gc.collect()

        assert spread.has_request_in_flight() is False
        assert (await client.get(URL)).text == "ok"
        assert [connection.closed for connection in connections.opened] == [True, False]


class _TwoChunks(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            yield b"o"
            yield b"k"
        finally:
            self.closed = True


class _TwoChunkConnection(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.stream: Final = _TwoChunks()
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=self.stream)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def cyclic_gc_off() -> Iterator[None]:
    gc.disable()
    yield
    gc.enable()


async def _read_one_chunk_and_drop(client: httpx.AsyncClient) -> bytes:
    response: Final = await client.send(client.build_request("GET", URL), stream=True)
    return await anext(response.aiter_raw())


def _reaped(closer: EvictedClientCloser, client: httpx.AsyncClient) -> bool:
    closer.reap()
    return client.is_closed


@pytest.mark.asyncio
@pytest.mark.usefixtures("cyclic_gc_off")
async def test_an_evicted_client_closes_once_a_partly_read_response_is_dropped_without_closing():
    connection: Final = _TwoChunkConnection()
    client: Final = httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(lambda _limits: connection, UNCAPPED))
    closer: Final = EvictedClientCloser(grace_seconds=0.0)
    closer.mark_owned(client)
    closer.schedule(client)

    assert await _read_one_chunk_and_drop(client) == b"o"

    await _until(lambda: _reaped(closer, client))
    await _until(lambda: connection.closed and connection.stream.closed)

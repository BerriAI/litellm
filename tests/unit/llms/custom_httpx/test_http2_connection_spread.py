import asyncio
import gc
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final, cast

import h2.events
import h2.settings
import httpcore
import httpx
import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.llms.custom_httpx.http2_connection_spread import (
    HTTP2_STREAMS_PER_CONNECTION,
    AsyncHTTP2ConnectionSpread,
)

URL: Final = "https://search.example/v1/search"
OTHER_URL: Final = "https://other.example/v1/search"
BURST: Final = HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
UNCAPPED: Final = httpx.Limits(max_connections=None)
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


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: tuple[str, ...] = ()

    async def __call__(self, event: str, _info: Mapping[str, object]) -> None:
        self.events = (*self.events, event)


class _GatedAsyncConnection(httpx.AsyncBaseTransport):
    def __init__(self, connections: "_AsyncConnections") -> None:
        self._connections: Final = connections
        self._settings_received = False
        self.hosts: frozenset[str] = frozenset()
        self.in_flight = 0
        self.peak = 0
        self.served = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self._connections.refuse_if_due(request)
        self.hosts = self.hosts | {request.url.netloc.decode()}
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        if self._settings_received:
            await self._connections.resend_settings(request)
        else:
            self._settings_received = True
            await self._connections.send_settings(request)
        await self._connections.gate.wait()
        self.in_flight -= 1
        self.served += 1
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    async def aclose(self) -> None:
        self.closed = True


class _AsyncConnections:
    def __init__(
        self,
        gate: asyncio.Event,
        advertised: Mapping[str, int | None] = MappingProxyType({}),
        through_tunnel: bool = False,
        refusals: int = 0,
        reports_settings: bool = True,
        resends_settings: bool = False,
    ) -> None:
        self.gate: Final = gate
        self.settings_arrive: Final = asyncio.Event()
        self.settings_arrive.set()
        self.advertised = advertised
        self._through_tunnel: Final = through_tunnel
        self._refusals_left = refusals
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
        raise httpx.ConnectError("refused", request=request)

    async def send_settings(self, request: httpx.Request) -> None:
        if not self._reports_settings:
            return
        if self._through_tunnel:
            tunnel: Final = httpcore.Request(b"CONNECT", str(request.url))
            await _emit(request, "http11.send_request_headers.started", {"request": tunnel})
        await self.settings_arrive.wait()
        await self._emit_settings(request)

    async def resend_settings(self, request: httpx.Request) -> None:
        if self._resends_settings:
            await self._emit_settings(request)

    async def _emit_settings(self, request: httpx.Request) -> None:
        streams: Final = self.advertised.get(request.url.netloc.decode(), MORE_STREAMS_THAN_WE_RUN)
        await _emit(request, "http2.receive_remote_settings.complete", {"return_value": _remote_settings(streams)})

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
    def __init__(self, gate: asyncio.Event) -> None:
        self._gate: Final = gate
        self.requested = 0
        self.open = 0

    def __call__(self, limits: httpx.Limits) -> "_HTTP1Pool":
        assert limits.max_connections is not None
        return _HTTP1Pool(self, asyncio.Semaphore(limits.max_connections))

    async def serve(self, request: httpx.Request, pool_slots: asyncio.Semaphore) -> None:
        self.requested += 1
        async with pool_slots:
            self.open += 1
            sent: Final = httpcore.Request(request.method, str(request.url))
            await _emit(request, "http11.send_request_headers.started", {"request": sent})
            await self._gate.wait()
            self.open -= 1


class _HTTP1Pool(httpx.AsyncBaseTransport):
    def __init__(self, connections: _HTTP1Connections, slots: asyncio.Semaphore) -> None:
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
    ("advertised", "through_tunnel", "streams"),
    [(FEW_STREAMS, False, FEW_STREAMS), (FEW_STREAMS, True, FEW_STREAMS), (None, False, 1)],
    ids=["few-streams", "few-streams-through-proxy-tunnel", "no-stream-limit-sent"],
)
@pytest.mark.asyncio
async def test_async_spread_opens_another_connection_at_the_upstreams_stream_limit(
    advertised: int | None, through_tunnel: bool, streams: int
):
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(
        gate, advertised=MappingProxyType({"search.example": advertised}), through_tunnel=through_tunnel
    )
    connections.settings_arrive.clear()
    burst: Final = 4 * streams
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, UNCAPPED)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == 1)

        connections.settings_arrive.set()
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [streams] * 4

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_learns_the_stream_limit_from_the_next_request_when_the_first_fails():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(
        gate, advertised=MappingProxyType({"search.example": FEW_STREAMS}), refusals=1
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
        with pytest.raises(httpx.ConnectError):
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
        connections.advertised = MappingProxyType({"search.example": later_advertised})
        gate.clear()

        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [streams] * (burst // streams)

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
    two_connections_of_streams: Final = httpx.Limits(max_connections=2 * HTTP2_STREAMS_PER_CONNECTION)
    urls: Final = (URL,) * HTTP2_STREAMS_PER_CONNECTION + (OTHER_URL,) * overflow + (URL,) * overflow
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, two_connections_of_streams)
    ) as client:
        pending: Final = tuple(asyncio.create_task(client.get(url)) for url in urls)
        await _until(lambda: connections.in_flight() == len(urls))
        assert [connection.hosts for connection in connections.opened] == [
            frozenset({"search.example"}),
            frozenset({"other.example"}),
        ]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


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
    assert trace.events == ("http2.receive_remote_settings.complete",)


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
async def test_async_spread_shares_the_least_busy_connection_once_the_connection_cap_is_reached():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    burst: Final = 2 * HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
    two_connections_of_streams: Final = httpx.Limits(max_connections=2 * HTTP2_STREAMS_PER_CONNECTION)
    async with httpx.AsyncClient(
        transport=AsyncHTTP2ConnectionSpread(connections, two_connections_of_streams)
    ) as client:
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
        await _until(lambda: connections.requested == burst)
        assert connections.open == cap

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


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
            await response.aclose()
        await client.get(URL)
        assert [connection.served for connection in connections.opened] == [HTTP2_STREAMS_PER_CONNECTION + 1, 1]


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
        assert len(connections.opened) == 1


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

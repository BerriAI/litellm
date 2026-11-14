import asyncio
import gc
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Final

import httpx
import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.llms.custom_httpx.http2_connection_spread import (
    HTTP2_STREAMS_PER_CONNECTION,
    AsyncHTTP2ConnectionSpread,
)

URL: Final = "https://search.example/v1/search"
BURST: Final = HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2
UNCAPPED: Final = httpx.Limits(max_connections=None)


class _GatedAsyncConnection(httpx.AsyncBaseTransport):
    def __init__(self, gate: asyncio.Event) -> None:
        self._gate: Final = gate
        self.in_flight = 0
        self.peak = 0
        self.served = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await self._gate.wait()
        self.in_flight -= 1
        self.served += 1
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    async def aclose(self) -> None:
        self.closed = True


class _AsyncConnections:
    def __init__(self, gate: asyncio.Event) -> None:
        self._gate: Final = gate
        self.opened: tuple[_GatedAsyncConnection, ...] = ()

    def __call__(self, _limits: httpx.Limits) -> _GatedAsyncConnection:
        connection: Final = _GatedAsyncConnection(self._gate)
        self.opened = (*self.opened, connection)
        return connection

    def in_flight(self) -> int:
        return sum(connection.in_flight for connection in self.opened)


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

    async def serve(self, pool_slots: asyncio.Semaphore) -> None:
        self.requested += 1
        async with pool_slots:
            self.open += 1
            await self._gate.wait()
            self.open -= 1


class _HTTP1Pool(httpx.AsyncBaseTransport):
    def __init__(self, connections: _HTTP1Connections, slots: asyncio.Semaphore) -> None:
        self._connections: Final = connections
        self._slots: Final = slots

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self._connections.serve(self._slots)
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

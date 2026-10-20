import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
import pytest

from litellm.llms.custom_httpx.http2_connection_spread import (
    HTTP2_STREAMS_PER_CONNECTION,
    AsyncHTTP2ConnectionSpread,
    HTTP2ConnectionSpread,
)

URL: Final = "https://search.example/v1/search"
BURST: Final = HTTP2_STREAMS_PER_CONNECTION + HTTP2_STREAMS_PER_CONNECTION // 2


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

    def __call__(self) -> _GatedAsyncConnection:
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

    def __call__(self) -> _RefusedConnection:
        self.opened += 1
        return _RefusedConnection()


class _GatedConnection(httpx.BaseTransport):
    def __init__(self, gate: threading.Event) -> None:
        self._gate: Final = gate
        self._lock: Final = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.served = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        assert self._gate.wait(timeout=10)
        with self._lock:
            self.in_flight -= 1
            self.served += 1
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))


class _Connections:
    def __init__(self, gate: threading.Event) -> None:
        self._gate: Final = gate
        self.opened: tuple[_GatedConnection, ...] = ()

    def __call__(self) -> _GatedConnection:
        connection: Final = _GatedConnection(self._gate)
        self.opened = (*self.opened, connection)
        return connection

    def in_flight(self) -> int:
        return sum(connection.in_flight for connection in self.opened)


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0)
    pytest.fail("condition never held")


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline: Final = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.01)


@pytest.mark.asyncio
async def test_async_spread_opens_another_connection_only_once_streams_run_out():
    gate: Final = asyncio.Event()
    connections: Final = _AsyncConnections(gate)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections)) as client:
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
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections, max_connections=2)) as client:
        pending: Final = tuple(asyncio.create_task(client.get(URL)) for _ in range(burst))
        await _until(lambda: connections.in_flight() == burst)
        assert [connection.peak for connection in connections.opened] == [burst // 2, burst // 2]

        gate.set()
        assert {response.status_code for response in await asyncio.gather(*pending)} == {200}


@pytest.mark.asyncio
async def test_async_spread_holds_a_stream_until_its_response_is_closed():
    gate: Final = asyncio.Event()
    gate.set()
    connections: Final = _AsyncConnections(gate)
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections)) as client:
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
    async with httpx.AsyncClient(transport=AsyncHTTP2ConnectionSpread(connections)) as client:
        for _ in range(BURST):
            with pytest.raises(httpx.ConnectError):
                await client.get(URL)
    assert connections.opened == 1


def test_sync_spread_opens_another_connection_only_once_streams_run_out():
    gate: Final = threading.Event()
    connections: Final = _Connections(gate)
    with (
        httpx.Client(transport=HTTP2ConnectionSpread(connections)) as client,
        ThreadPoolExecutor(max_workers=BURST) as pool,
    ):
        pending: Final = tuple(pool.submit(client.get, URL) for _ in range(BURST))
        _wait_until(lambda: connections.in_flight() == BURST)
        assert [connection.peak for connection in connections.opened] == [
            HTTP2_STREAMS_PER_CONNECTION,
            BURST - HTTP2_STREAMS_PER_CONNECTION,
        ]

        gate.set()
        assert {future.result().status_code for future in pending} == {200}
        assert client.get(URL).text == "ok"
        assert [connection.served for connection in connections.opened] == [
            HTTP2_STREAMS_PER_CONNECTION + 1,
            BURST - HTTP2_STREAMS_PER_CONNECTION,
        ]

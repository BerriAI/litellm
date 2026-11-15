"""
Tests for EvictedClientCloser.

An evicted client must stay open long enough for a request that already holds it
to finish, and must then actually be closed, otherwise its connection pool is
retained until a generational collection runs. A client the caller supplied is
never closed, because litellm does not own its lifecycle.
"""

import asyncio

import httpx
import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


class FakeClock:
    """Hand-advanced monotonic clock, so grace windows need no real waiting."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_closer(clock: FakeClock, grace_seconds: float = 60.0) -> EvictedClientCloser:
    return EvictedClientCloser(grace_seconds=grace_seconds, clock=clock)


async def _trickling_upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serves a chunked body slowly, so a request stays on the wire long enough to observe."""
    await reader.read(4096)
    writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
    await writer.drain()
    for _ in range(6):
        writer.write(b"5\r\nhello\r\n")
        await writer.drain()
        await asyncio.sleep(0.1)
    writer.write(b"0\r\n\r\n")
    await writer.drain()


@pytest.mark.asyncio
async def test_a_client_serving_a_request_is_not_closed_when_its_grace_window_ends():
    """The grace window on its own cannot promise that a request has finished.

    ``litellm.request_timeout`` defaults to 6000 seconds and a streaming response
    is bounded only by how long the upstream keeps sending, so a client past its
    deadline is closed only once its own pool reports nothing in flight.
    """
    server = await asyncio.start_server(_trickling_upstream, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    clock = FakeClock()
    closer = make_closer(clock)
    client = httpx.AsyncClient()

    closer.mark_owned(client)
    closer.schedule(client)

    async def read_the_stream() -> int:
        received = 0
        async with client.stream("GET", f"http://127.0.0.1:{port}/") as response:
            async for chunk in response.aiter_bytes():
                received += len(chunk)
        return received

    streaming = asyncio.create_task(read_the_stream())
    await asyncio.sleep(0.25)  # the request is on the wire
    clock.advance(3600.0)  # and its grace window is long gone
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.is_closed is False, "closed a client that was serving a request"
    assert await streaming > 0, "the in-flight request did not survive the reap"

    clock.advance(3600.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.is_closed is True, "an idle client past its grace window must be closed"
    assert closer.pending_count == 0
    server.close()


@pytest.mark.asyncio
async def test_the_aiohttp_backed_handler_is_not_closed_mid_request():
    """The default async path is aiohttp-backed, whose pool accounts for its own leases."""
    server = await asyncio.start_server(_trickling_upstream, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    clock = FakeClock()
    closer = make_closer(clock)
    handler = AsyncHTTPHandler()
    held_client = handler.client

    closer.mark_owned(handler)
    closer.schedule(handler)

    request = asyncio.create_task(handler.get(f"http://127.0.0.1:{port}/"))
    await asyncio.sleep(0.25)
    clock.advance(3600.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert held_client.is_closed is False, "closed a handler that was serving a request"
    assert (await request).status_code == 200

    clock.advance(3600.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert held_client.is_closed is True
    assert handler.client.is_closed is False, "a held handler must self-heal after its evicted client is closed"
    server.close()

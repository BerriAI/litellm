"""
Tests for EvictedClientCloser.

An evicted client must stay open long enough for a request that already holds it
to finish, and must then actually be closed, otherwise its connection pool is
retained until a generational collection runs. A client the caller supplied is
never closed, because litellm does not own its lifecycle.
"""

import asyncio
import gc
import weakref

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


class AsyncClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class SyncClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class CountingDeadline(float):
    """A clock reading that tallies every deadline comparison made against it.

    Deadline comparisons are the work a reap does, so counting them says whether
    that work tracks the entries that are due or the size of the whole queue.
    """

    comparisons = 0

    def __add__(self, other: float) -> "CountingDeadline":
        return CountingDeadline(float(self) + other)

    def __le__(self, other: float) -> bool:
        CountingDeadline.comparisons += 1
        return float(self) <= float(other)

    def __gt__(self, other: float) -> bool:
        CountingDeadline.comparisons += 1
        return float(self) > float(other)


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
async def test_owned_client_is_closed_once_the_grace_window_elapses():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is True
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_owned_client_stays_open_inside_the_grace_window():
    """A request handed the client just before eviction is still using it."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(59.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 1


@pytest.mark.asyncio
async def test_caller_supplied_client_is_never_closed():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.schedule(client)
    clock.advance(3600.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_sync_client_is_closed_once_the_grace_window_elapses():
    clock = FakeClock()
    closer = make_closer(clock)
    client = SyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(61.0)
    closer.reap()

    assert client.closed is True


@pytest.mark.asyncio
async def test_a_failing_close_does_not_propagate_or_block_the_others():
    class ExplodingClient:
        async def close(self) -> None:
            raise RuntimeError("connection already gone")

    clock = FakeClock()
    closer = make_closer(clock)
    exploding, healthy = ExplodingClient(), AsyncClient()

    for client in (exploding, healthy):
        closer.mark_owned(client)
        closer.schedule(client)
    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert healthy.closed is True


@pytest.mark.asyncio
async def test_an_unhashable_cached_value_does_not_break_eviction():
    """The cache holds arbitrary values; an ownership test must never raise on one."""

    class Unhashable:
        __hash__ = None  # pyright: ignore[reportAssignmentType]  # unhashable by construction

    clock = FakeClock()
    closer = make_closer(clock)

    closer.mark_owned(Unhashable())
    closer.schedule(Unhashable())

    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_values_with_nothing_to_close_are_never_queued():
    """The cache holds plain values too; those have nothing to reclaim."""

    class NotAClient:
        pass

    clock = FakeClock()
    closer = make_closer(clock)
    value = NotAClient()

    closer.mark_owned(value)
    closer.schedule(value)

    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_a_queued_client_is_not_kept_alive_by_the_queue():
    """Waiting out a grace window must not retain what the collector would free first."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    gone = weakref.ref(client)

    closer.mark_owned(client)
    closer.schedule(client)
    del client
    gc.collect()

    assert gone() is None, "the pending queue is holding the client alive"

    clock.advance(61.0)
    closer.reap()
    assert closer.pending_count == 0


def test_sync_client_evicted_outside_an_event_loop_is_still_closed():
    """The sync httpx handler is cached and evicted from call sites with no loop."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = SyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    assert closer.pending_count == 1

    clock.advance(61.0)
    closer.reap()

    assert client.closed is True
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_an_async_client_waits_for_a_loop_rather_than_being_dropped():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    closer.mark_owned(client)

    def schedule_outside_a_loop() -> None:
        closer.schedule(client)
        clock.advance(61.0)
        closer.reap()

    await asyncio.to_thread(schedule_outside_a_loop)
    assert client.closed is False, "no loop was running, so it could not have been closed"
    assert closer.pending_count == 1

    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is True


@pytest.mark.asyncio
async def test_a_client_evicted_on_another_event_loop_is_left_alone():
    """Closing a client bound to a different loop would schedule work on that loop."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    closer.mark_owned(client)

    def schedule_on_its_own_loop() -> None:
        asyncio.run(_schedule())

    async def _schedule() -> None:
        closer.schedule(client)

    await asyncio.to_thread(schedule_on_its_own_loop)
    assert closer.pending_count == 1

    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 1


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


def test_the_pending_queue_cannot_grow_past_its_bound():
    """A caller that churns the client cache must not be able to grow this queue."""
    clock = FakeClock()
    closer = EvictedClientCloser(grace_seconds=60.0, max_pending=8, clock=clock)
    clients = tuple(SyncClient() for _ in range(50))

    for client in clients:
        closer.mark_owned(client)
        closer.schedule(client)

    assert closer.pending_count == 8, "the queue grew past max_pending"

    clock.advance(61.0)
    closer.reap()

    assert closer.pending_count == 0
    assert sum(client.closed for client in clients) == 8, "everything queued should have been closed"


def test_a_reap_looks_at_what_is_due_rather_than_at_the_whole_queue():
    """Sustained churn evicts a client per request, and every read of the cache reaps.

    So the cost of a reap has to track the entries that are due, not the length of
    the queue; a reap that filters the whole queue makes the pair quadratic. Each
    bucket is ordered by deadline, so an up-to-date reap compares one entry per
    bucket and stops. Counting the comparisons measures that directly, where a
    wall-clock budget would only measure the machine.
    """
    evictions = 1_000
    clock = FakeClock()
    closer = EvictedClientCloser(
        grace_seconds=60.0,
        max_pending=evictions,
        clock=lambda: CountingDeadline(clock.now),
    )
    clients = tuple(SyncClient() for _ in range(evictions))
    for client in clients:
        closer.mark_owned(client)

    CountingDeadline.comparisons = 0
    for client in clients:
        closer.schedule(client)
        closer.reap()  # nothing is due yet, which is the hot path
    clock.advance(61.0)
    closer.reap()

    assert closer.pending_count == 0
    assert all(client.closed for client in clients)
    assert CountingDeadline.comparisons < 10 * evictions, (
        f"{CountingDeadline.comparisons} deadline comparisons for {evictions} evictions; "
        "a reap is walking the whole queue"
    )

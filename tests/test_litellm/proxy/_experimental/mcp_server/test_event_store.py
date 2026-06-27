import pytest

pytest.importorskip("mcp")

from mcp.server.streamable_http import EventMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from litellm.proxy._experimental.mcp_server.event_store import InMemoryMCPEventStore


def _message(seq: int) -> JSONRPCMessage:
    return JSONRPCMessage(
        JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/message",
            params={"seq": seq},
        )
    )


def _collector():
    received: list = []

    async def send(event: EventMessage) -> None:
        received.append(event)

    return received, send


@pytest.mark.asyncio
async def test_replay_returns_events_after_last_event_id_in_order():
    store = InMemoryMCPEventStore()
    ids = [await store.store_event("stream-1", _message(i)) for i in range(5)]

    received, send = _collector()
    stream_id = await store.replay_events_after(ids[1], send)

    assert stream_id == "stream-1"
    assert [e.event_id for e in received] == ids[2:]
    assert [e.message.root.params["seq"] for e in received] == [2, 3, 4]


@pytest.mark.asyncio
async def test_replay_unknown_event_id_returns_none_and_sends_nothing():
    store = InMemoryMCPEventStore()
    await store.store_event("stream-1", _message(0))

    received, send = _collector()
    assert await store.replay_events_after("nonexistent", send) is None
    assert received == []


@pytest.mark.asyncio
async def test_replay_is_scoped_to_the_stream_of_the_last_event():
    store = InMemoryMCPEventStore()
    first = await store.store_event("stream-1", _message(1))
    await store.store_event("stream-2", _message(100))
    await store.store_event("stream-1", _message(2))

    received, send = _collector()
    stream_id = await store.replay_events_after(first, send)

    assert stream_id == "stream-1"
    assert [e.message.root.params["seq"] for e in received] == [2]


@pytest.mark.asyncio
async def test_priming_events_are_not_replayed():
    store = InMemoryMCPEventStore()
    first = await store.store_event("stream-1", _message(1))
    await store.store_event("stream-1", None)
    await store.store_event("stream-1", _message(2))

    received, send = _collector()
    await store.replay_events_after(first, send)

    assert [e.message.root.params["seq"] for e in received] == [2]


@pytest.mark.asyncio
async def test_per_stream_bound_evicts_oldest_events():
    store = InMemoryMCPEventStore(max_events_per_stream=3)
    ids = [await store.store_event("stream-1", _message(i)) for i in range(5)]

    received, send = _collector()
    assert await store.replay_events_after(ids[0], send) is None

    received, send = _collector()
    stream_id = await store.replay_events_after(ids[2], send)
    assert stream_id == "stream-1"
    assert [e.message.root.params["seq"] for e in received] == [3, 4]


@pytest.mark.asyncio
async def test_stream_count_bound_evicts_least_recently_used_stream():
    store = InMemoryMCPEventStore(max_streams=2)
    old_id = await store.store_event("stream-old", _message(0))
    mid_id = await store.store_event("stream-mid", _message(1))
    # Touch stream-old so stream-mid becomes the LRU candidate.
    kept_id = await store.store_event("stream-old", _message(2))
    await store.store_event("stream-new", _message(3))

    received, send = _collector()
    assert await store.replay_events_after(mid_id, send) is None
    assert received == []

    old_received, send_old = _collector()
    assert await store.replay_events_after(old_id, send_old) == "stream-old"
    assert [e.event_id for e in old_received] == [kept_id]

from typing import Final

import anyio
import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.events.upstream import Overflow, StreamFrame, StreamRoute, call, list_all, routing_headers, stream
from litellm.types.mcp_server.events import EventType

from .upstream_server import EventsUpstream, events_client

REPO: Final = EventType(
    name="build.failed",
    delivery=("poll",),
    input_schema={"type": "object", "properties": {"repo": {"type": "string", "x-mcp-header": "Repo"}}},
)


@pytest.mark.asyncio
async def test_a_unary_call_declares_the_extension_and_carries_routing_headers() -> None:
    upstream: Final = EventsUpstream()
    async with events_client(upstream) as client:
        result: Final = await call(
            client, "events/poll", {"name": "build.failed", "arguments": {"repo": "acme/web"}}, routing_headers("build.failed", {"repo": "acme/web"}, REPO), 5.0
        )
    assert result["cursor"] == "c1"
    method, headers, params = upstream.seen[-1]
    assert (method, headers["mcp-name"], headers["mcp-param-repo"]) == ("events/poll", "build.failed", "acme/web")
    capabilities: Final = params["_meta"]["io.modelcontextprotocol/clientCapabilities"]  # pyright: ignore[reportIndexIssue]  # wire dict
    assert capabilities["extensions"] == {"io.modelcontextprotocol/events": {}}  # pyright: ignore[reportIndexIssue]  # wire dict


@pytest.mark.asyncio
async def test_upstream_errors_surface_as_mcp_errors() -> None:
    async with events_client(EventsUpstream()) as client:
        with pytest.raises(MCPError) as rejected:
            await call(client, "events/poll", {"name": "nope"}, {}, 5.0)
    assert rejected.value.error.code == -32023


@pytest.mark.asyncio
async def test_a_hung_upstream_times_out_instead_of_hanging_the_poll() -> None:
    async with events_client(EventsUpstream(hang=True), timeout=0.2) as client:
        with anyio.fail_after(5), pytest.raises((MCPError, TimeoutError)):
            await call(client, "events/poll", {"name": "build.failed"}, {}, 0.2)


@pytest.mark.asyncio
async def test_listing_follows_upstream_pages() -> None:
    upstream: Final = EventsUpstream(pages=(("a.one",), ("b.two", "c.three")))
    async with events_client(upstream) as client:
        listed: Final = await list_all(client, 10, 5.0)
    assert [item.name for item in listed] == ["a.one", "b.two", "c.three"]


@pytest.mark.asyncio
async def test_a_stream_delivers_its_own_notifications_in_wire_order() -> None:
    received: Final[list[str]] = []  # mutable-ok: collects frames

    async def consume(frames: object) -> None:
        async for frame in frames:  # pyright: ignore[reportGeneralTypeIssues]  # memory receive stream
            assert isinstance(frame, StreamFrame)
            received.append(frame.method)

    async with events_client(EventsUpstream()) as client:
        with anyio.fail_after(5):
            await stream(client, {"name": "build.failed"}, {}, consume)  # pyright: ignore[reportArgumentType]  # test consumer
    assert received == [
        "notifications/events/active",
        "notifications/events/event",
        "notifications/events/heartbeat",
    ]


def _frame(cursor: str) -> StreamFrame:
    return StreamFrame("notifications/events/event", {"cursor": cursor})


def _drain(route: StreamRoute) -> tuple[StreamFrame | Overflow, ...]:
    def items():  # noqa: ANN202  # local generator
        while True:
            try:
                yield route.receive.receive_nowait()
            except (anyio.EndOfStream, anyio.WouldBlock):
                return

    return tuple(items())


def test_a_full_route_reports_dropped_frames_as_one_overflow_before_the_next_frame() -> None:
    route: Final = StreamRoute()
    for index in range(259):
        route.deliver(_frame(str(index)))
    assert route.receive.receive_nowait() == _frame("0")
    route.deliver(_frame("next"))
    drained: Final = _drain(route)
    assert drained[-1] == Overflow(cursor="258")
    assert _frame("next") not in drained and _frame("257") not in drained
    route.deliver(_frame("after"))
    assert _drain(route) == (Overflow(cursor="next"), _frame("after"))


def test_an_overflow_still_pending_at_the_end_is_flushed_when_the_route_finishes() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    route.receive.receive_nowait()
    route.finish(None)
    assert _drain(route)[-1] == Overflow(cursor="256")

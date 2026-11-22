from typing import Final

import anyio
import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.events.upstream import (
    EventsClientSession,
    call,
    list_all,
    routing_headers,
)
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
            client,
            "events/poll",
            {"name": "build.failed", "arguments": {"repo": "acme/web"}},
            routing_headers("build.failed", {"repo": "acme/web"}, REPO),
            5.0,
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
async def test_an_events_call_does_not_change_the_session_later_tool_calls_use() -> None:
    from mcp.client.session import ClientSession

    seen: Final[list[type]] = []  # mutable-ok: records the session type of each call

    async with events_client(EventsUpstream()) as client:
        await call(client, "events/list", {}, {}, 5.0)

        async def probe(session: ClientSession) -> dict[str, object]:
            seen.append(type(session))
            return {}

        await client.run_with_session(probe, quiet_on_error=True)
    assert seen and seen[0] is not EventsClientSession

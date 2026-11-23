import math
from typing import Final

import anyio
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolRequestParams, CallToolResult, ClientCapabilities, TextContent
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.mcp_server.events.upstream import (
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
@pytest.mark.parametrize("cursor", ("1", ""))
async def test_listing_follows_upstream_pages(cursor: str) -> None:
    upstream: Final = EventsUpstream(pages=(("a.one",), ("b.two", "c.three")), second_page_cursor=cursor)
    async with events_client(upstream) as client:
        listed: Final = await list_all(client, 10, 5.0)
    assert [item.name for item in listed] == ["a.one", "b.two", "c.three"]


@pytest.mark.asyncio
async def test_an_events_call_does_not_change_the_session_later_tool_calls_use() -> None:
    upstream: Final = EventsUpstream()
    async with events_client(upstream) as client:
        await call(client, "events/list", {}, {}, 5.0)
        result: Final = await client.call_tool(CallToolRequestParams(name="example", arguments={}))
    assert result == CallToolResult(
        _meta={"io.modelcontextprotocol/serverInfo": {"name": "events-upstream", "version": "1.0.0"}},
        content=[TextContent(type="text", text="tool still works")],
        isError=False,
    )
    method, _, params = upstream.seen[-1]
    assert method == "tools/call"
    metadata: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(params["_meta"])
    capabilities: Final = ClientCapabilities.model_validate(metadata["io.modelcontextprotocol/clientCapabilities"])
    assert "io.modelcontextprotocol/events" not in (capabilities.extensions or {})


@pytest.mark.asyncio
async def test_catalog_larger_than_python_stack_depth_is_fully_returned() -> None:
    names: Final = tuple(f"build.event_{index}" for index in range(1100))
    upstream: Final = EventsUpstream(pages=tuple((name,) for name in names))
    async with events_client(upstream, timeout=60.0) as client:
        listed: Final = await list_all(client, 1100, 60.0)
    assert tuple(item.name for item in listed) == names


@pytest.mark.asyncio
async def test_catalog_exceeding_page_cap_is_rejected() -> None:
    upstream: Final = EventsUpstream(pages=(("a.one",), ("b.two",)))
    async with events_client(upstream) as client:
        with pytest.raises(MCPError) as rejected:
            await list_all(client, 1, 5.0)
    assert rejected.value.error.code == -32603
    assert len(upstream.seen) == 1


@pytest.mark.asyncio
async def test_catalog_pages_share_one_finite_deadline() -> None:
    upstream: Final = EventsUpstream(pages=(("a.one",), ("b.two",)))
    async with events_client(upstream) as client:
        await list_all(client, 10, 5.0)
    first, second = upstream.catalog_deadlines
    assert math.isfinite(first)
    assert second == first


@pytest.mark.asyncio
async def test_a_server_without_the_events_extension_receives_no_events_request() -> None:
    upstream: Final = EventsUpstream(advertises_events=False)
    async with events_client(upstream) as client:
        with pytest.raises(MCPError) as rejected:
            await list_all(client, 10, 5.0)
    assert rejected.value.error.code == -32601
    assert not upstream.seen

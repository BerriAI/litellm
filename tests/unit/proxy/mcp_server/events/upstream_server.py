from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Final

import anyio
import httpx2
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from mcp.types import RequestParams
from pydantic import BaseModel, ConfigDict, JsonValue

from litellm.mcp_client.client import MCPClient

SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"


class _Params(RequestParams):
    model_config = ConfigDict(extra="allow", frozen=True)


class _Note(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    params: dict[str, JsonValue]


@dataclass
class EventsUpstream:
    pages: tuple[tuple[str, ...], ...] = (("build.failed",),)
    advertises_events: bool = True
    second_page_cursor: str = "1"
    hang: bool = False
    catalog_deadlines: list[float] = field(default_factory=list)  # mutable-ok: captures per-page deadline evidence
    seen: list[tuple[str, Mapping[str, str], dict[str, object]]] = field(
        default_factory=list
    )  # mutable-ok: records requests

    def record(self, ctx: ServerRequestContext[object, object], params: _Params) -> None:
        request: Final = getattr(ctx, "request", None)
        headers: Final = (
            {key: value for key, value in request.headers.items() if key.startswith("mcp-")} if request else {}
        )
        self.seen.append((ctx.method, headers, params.model_dump(by_alias=True)))

    async def list_events(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        token: Final = (params.model_extra or {}).get("cursor")
        cursor: Final = 1 if token == self.second_page_cursor else int(str(token or 0))
        names: Final = self.pages[cursor]
        page: Final = {"events": [{"name": name, "delivery": ["poll", "push"]} for name in names]}
        return (
            {**page, "nextCursor": self.second_page_cursor if cursor == 0 else str(cursor + 1)}
            if cursor + 1 < len(self.pages)
            else page
        )

    async def poll(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        if self.hang:
            await anyio.sleep_forever()
        if (params.model_extra or {}).get("name") != "build.failed":
            raise MCPError(code=-32023, message="NotFound", data={"kind": "event"})
        return {"events": [], "cursor": "c1", "truncated": False, "hasMore": False, "nextPollMs": 1000}

    async def call_tool(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        return {"content": [{"type": "text", "text": "tool still works"}], "isError": False, "resultType": "complete"}

    async def list_tools(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        return {
            "tools": [{"name": "example", "inputSchema": {"type": "object"}}],
            "cacheScope": "private",
            "resultType": "complete",
            "ttlMs": 0,
        }

    async def stream(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        meta: Final = {SUBSCRIPTION_ID: ctx.request_id}
        for method, payload in (
            ("notifications/events/active", {"cursor": "a", "truncated": False, "heartbeatMs": 5000}),
            (
                "notifications/events/event",
                {"eventId": "e1", "name": "build.failed", "timestamp": "t", "data": {}, "cursor": "b"},
            ),
            ("notifications/events/heartbeat", {"cursor": "c"}),
        ):
            await ctx.session.send_notification(
                _Note(method=method, params={**payload, "_meta": meta}), related_request_id=ctx.request_id
            )  # pyright: ignore[reportArgumentType]  # custom notification model
        return {"_meta": meta}


class _CatalogTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport: httpx2.ASGITransport, upstream: EventsUpstream) -> None:
        self._transport = transport
        self._upstream = upstream

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and b'"events/list"' in request.content:
            self._upstream.catalog_deadlines.append(anyio.current_effective_deadline())
        return await self._transport.handle_async_request(request)


@asynccontextmanager
async def events_client(upstream: EventsUpstream, timeout: float = 5.0) -> AsyncGenerator[MCPClient, None]:
    server = Server[object]("events-upstream", version="1.0.0")
    if upstream.advertises_events:
        server.extensions["io.modelcontextprotocol/events"] = {}
    server.add_request_handler("events/list", _Params, upstream.list_events)
    server.add_request_handler("events/poll", _Params, upstream.poll)
    server.add_request_handler("events/stream", _Params, upstream.stream)
    server.add_request_handler("tools/call", _Params, upstream.call_tool)
    server.add_request_handler("tools/list", _Params, upstream.list_tools)
    manager = StreamableHTTPSessionManager(server, stateless=True)
    async with manager.run():
        yield MCPClient(
            server_url="http://upstream.test/mcp",
            transport_type="http",
            protocol_version="2026-07-28",
            timeout=timeout,
            upstream_transport=_CatalogTransport(httpx2.ASGITransport(app=manager.handle_request), upstream),
        )

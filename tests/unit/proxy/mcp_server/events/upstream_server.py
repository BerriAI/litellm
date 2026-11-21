from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

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
    model_config = ConfigDict(extra="allow")


class _Note(BaseModel):
    method: str
    params: dict[str, JsonValue]


@dataclass
class EventsUpstream:
    pages: tuple[tuple[str, ...], ...] = (("build.failed",),)
    hang: bool = False
    seen: list[tuple[str, Mapping[str, str], dict[str, object]]] = field(
        default_factory=list
    )  # mutable-ok: records requests

    def record(self, ctx: ServerRequestContext[object, object], params: _Params) -> None:
        request = getattr(ctx, "request", None)
        headers = {key: value for key, value in request.headers.items() if key.startswith("mcp-")} if request else {}
        self.seen.append((ctx.method, headers, params.model_dump(by_alias=True)))

    async def list_events(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        cursor = int((params.model_extra or {}).get("cursor") or 0)
        names = self.pages[cursor]
        page = {"events": [{"name": name, "delivery": ["poll", "push"]} for name in names]}
        return {**page, "nextCursor": str(cursor + 1)} if cursor + 1 < len(self.pages) else page

    async def poll(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        if self.hang:
            await anyio.sleep_forever()
        if (params.model_extra or {}).get("name") != "build.failed":
            raise MCPError(code=-32023, message="NotFound", data={"kind": "event"})
        return {"events": [], "cursor": "c1", "truncated": False, "hasMore": False, "nextPollMs": 1000}

    async def stream(self, ctx: ServerRequestContext[object, object], params: _Params) -> dict[str, object]:
        self.record(ctx, params)
        meta = {SUBSCRIPTION_ID: ctx.request_id}
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


@asynccontextmanager
async def events_client(upstream: EventsUpstream, timeout: float = 5.0) -> AsyncGenerator[MCPClient, None]:
    server = Server[object]("events-upstream")
    server.add_request_handler("events/list", _Params, upstream.list_events)
    server.add_request_handler("events/poll", _Params, upstream.poll)
    server.add_request_handler("events/stream", _Params, upstream.stream)
    manager = StreamableHTTPSessionManager(server, stateless=True)
    async with manager.run():
        yield MCPClient(
            server_url="http://upstream.test/mcp",
            transport_type="http",
            protocol_version="2026-07-28",
            timeout=timeout,
            upstream_transport=httpx2.ASGITransport(app=manager.handle_request),
        )

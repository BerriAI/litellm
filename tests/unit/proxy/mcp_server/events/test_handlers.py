import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Final

import httpx2
import pytest
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.handlers import (
    advertised_extensions,
    client_declares_events,
    events_gate,
    gated_events_method,
    register_event_handlers,
)
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.upstream import StreamFrame

from .fakes import SETTINGS, FakeUpstream, context, occurrence, runtime, server

META: Final = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {"extensions": {"io.modelcontextprotocol/events": {}}},
}


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-handler-key")


@asynccontextmanager
async def _operation(_: ServerRequestContext[object, object]) -> AsyncGenerator[OperationContext, None]:
    yield context()


class _Authorization:
    async def __call__(self) -> OperationContext:
        return context()


@asynccontextmanager
async def _authorization(_: ServerRequestContext[object, object]) -> AsyncGenerator[_Authorization, None]:
    yield _Authorization()


def _server(settings: EventsSettings | None, events_runtime: EventsRuntime | None) -> Server[object]:
    gateway: Final = Server[object]("gateway")
    if settings is not None and events_runtime is not None:
        gateway.middleware.append(events_gate(lambda: False, lambda: settings))
        register_event_handlers(gateway, _operation, _authorization, lambda: settings, events_runtime)  # pyright: ignore[reportArgumentType]  # test authorization scope
    return gateway


async def _call(
    gateway: Server[object], method: str, params: Mapping[str, object], headers: Mapping[str, str] | None = None
) -> tuple[int, list[dict[str, object]]]:
    manager: Final = StreamableHTTPSessionManager(gateway, stateless=True)
    async with manager.run(), httpx2.AsyncClient(transport=httpx2.ASGITransport(app=manager.handle_request)) as client:
        response: Final = await client.post(
            "http://gw/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": method,
                **(headers or {}),
            },
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": {**params, "_meta": META}},
        )
    text: Final = response.text
    frames: Final = [json.loads(text)] if text.lstrip().startswith("{") else [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]
    return response.status_code, frames


@pytest.mark.parametrize(
    ("method", "params"),
    (
        ("events/poll", {"name": "ci-build.failed"}),
        ("events/poll", {"name": 7}),
        ("events/list", {}),
        ("events/subscribe", {}),
    ),
)
@pytest.mark.asyncio
async def test_with_events_off_every_events_method_answers_like_a_server_without_it(
    method: str, params: Mapping[str, object]
) -> None:
    baseline: Final = await _call(_server(None, None), method, params)
    gated: Final = await _call(_server(EventsSettings(), runtime((server(),))), method, params)
    assert gated == baseline
    assert baseline[1][0]["error"] == {"code": -32601, "message": "Method not found", "data": method}


@pytest.mark.asyncio
async def test_events_list_returns_the_gateway_catalog() -> None:
    status, frames = await _call(_server(SETTINGS, runtime((server(),))), "events/list", {})
    assert status == 200
    assert [item["name"] for item in frames[0]["result"]["events"]] == ["ci-build.failed"]  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]  # wire dict


@pytest.mark.asyncio
async def test_a_routing_header_mismatch_is_http_400() -> None:
    status, frames = await _call(
        _server(SETTINGS, runtime((server(),))),
        "events/poll",
        {"name": "ci-build.failed", "arguments": {"repo": "acme/web"}},
        {"Mcp-Name": "ci-build.failed"},
    )
    assert status == 400
    assert frames[0]["error"]["code"] == -32020  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]  # wire dict


@pytest.mark.asyncio
async def test_events_stream_sends_notifications_on_the_request_and_closes_with_the_subscription_id() -> None:
    upstream: Final = FakeUpstream(
        frames=(
            StreamFrame("notifications/events/active", {"cursor": "1", "truncated": False, "heartbeatMs": 5000}),
            StreamFrame("notifications/events/event", occurrence("evt_9", "2").model_dump(by_alias=True, mode="json")),
        )
    )
    status, frames = await _call(
        _server(SETTINGS, runtime((server(),), upstream=upstream)),
        "events/stream",
        {"name": "ci-build.failed", "arguments": {"repo": "acme/web"}},
        {"Mcp-Name": "ci-build.failed", "Mcp-Param-Repo": "acme/web"},
    )
    assert status == 200
    assert [frame.get("method") for frame in frames] == [
        "notifications/events/active",
        "notifications/events/event",
        None,
    ]
    assert frames[-1]["result"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == 1  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]  # wire dict
    assert frames[1]["params"]["eventId"] == "evt_9"  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]  # wire dict


@pytest.mark.parametrize(
    ("meta", "declared"),
    (
        (META, True),
        ({"io.modelcontextprotocol/clientCapabilities": {}}, False),
        ({"io.modelcontextprotocol/clientCapabilities": {"extensions": {"other/ext": {}}}}, False),
        ({"io.modelcontextprotocol/clientCapabilities": "garbage"}, False),
        (None, False),
    ),
)
def test_client_declaration(meta: Mapping[str, object] | None, declared: bool) -> None:
    assert client_declares_events(meta) is declared


@pytest.mark.parametrize(
    ("settings", "protocol", "servers", "meta", "advertised"),
    (
        (SETTINGS, "2026-07-28", (server(),), META, True),
        (EventsSettings(), "2026-07-28", (server(),), META, False),
        (SETTINGS, "2025-11-25", (server(),), META, False),
        (SETTINGS, "2026-07-28", (), META, False),
        (SETTINGS, "2026-07-28", (server(),), {}, False),
    ),
)
@pytest.mark.asyncio
async def test_discovery_advertises_events_only_when_everything_lines_up(
    settings: EventsSettings, protocol: str, servers: tuple[object, ...], meta: Mapping[str, object], advertised: bool
) -> None:
    caller_context: Final = OperationContext(_caller=context().user_api_key_auth, protocol_version=protocol)
    extensions: Final = await advertised_extensions(caller_context, meta, lambda: settings, runtime(servers))  # pyright: ignore[reportArgumentType]  # servers are MCPServer
    assert (extensions == {"io.modelcontextprotocol/events": {}}) is advertised


@pytest.mark.parametrize(
    ("method", "settings", "protocol", "proxy", "gated"),
    (
        ("events/poll", EventsSettings(), "2026-07-28", False, True),
        ("events/poll", SETTINGS, "2026-07-28", False, False),
        ("events/poll", SETTINGS, "2025-11-25", False, True),
        ("events/poll", SETTINGS, "2026-07-28", True, True),
        ("tools/list", EventsSettings(), "2026-07-28", False, False),
    ),
)
def test_gating(method: str, settings: EventsSettings, protocol: str, proxy: bool, gated: bool) -> None:
    assert gated_events_method(method, protocol, proxy, lambda: settings) is gated

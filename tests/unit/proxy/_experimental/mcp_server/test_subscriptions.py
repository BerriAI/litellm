import asyncio
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from mcp import MCPError
from mcp.server.context import ServerRequestContext
from mcp.server.session import ServerSession
from mcp.server.subscriptions import ListenHandler
from mcp.shared.subscriptions import PromptsListChanged, ToolsListChanged
from mcp.types import SubscriptionFilter, SubscriptionsListenRequestParams

from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server import subscriptions
from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.mcp import MCPTransport


def context(session: ServerSession, revision: str = "2026-07-28") -> ServerRequestContext:
    return ServerRequestContext(
        session=session,
        lifespan_context=None,
        protocol_version=revision,
        method="subscriptions/listen",
        request_id="listener",
    )


@pytest.mark.asyncio
async def test_queued_notification_rechecks_authority_before_delivery() -> None:
    bus: Final = subscriptions._ListenerBus()
    sender: Final = Mock(spec=ServerSession)
    first_delivery: Final = asyncio.Event()
    release: Final = asyncio.Event()
    revoked: Final = asyncio.Event()
    observed: Final[list[str]] = []

    async def send(notification, related_request_id=None) -> None:
        assert related_request_id == "listener"
        observed.append(notification.method)
        if notification.method == "notifications/tools/list_changed":
            first_delivery.set()
            await release.wait()

    async def authorize() -> OperationContext:
        if revoked.is_set():
            raise HTTPException(status_code=403, detail="Revoked caller")
        return OperationContext(_caller=UserAPIKeyAuth(user_id="alice"))

    sender.send_notification = AsyncMock(side_effect=send)
    handler: Final = ListenHandler(bus, max_buffered_events=2)
    request: Final = SubscriptionsListenRequestParams(
        notifications=SubscriptionFilter(tools_list_changed=True, prompts_list_changed=True)
    )
    task: Final = asyncio.create_task(handler(context(subscriptions._AuthorizedSession(sender, authorize)), request))
    try:
        await asyncio.wait_for(bus.ready.wait(), 1)
        await bus.publish(ToolsListChanged())
        await asyncio.wait_for(first_delivery.wait(), 1)
        await bus.publish(PromptsListChanged())
        revoked.set()
        release.set()
        with pytest.raises(HTTPException, match="Revoked caller"):
            await asyncio.wait_for(task, 1)
        assert observed == ["notifications/subscriptions/acknowledged", "notifications/tools/list_changed"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", ["2024-11-05", "2025-11-25"])
async def test_legacy_listener_rejected_before_upstream_work(revision: str) -> None:
    authorize: Final = AsyncMock()
    service: Final = subscriptions.GatewaySubscriptions()
    with pytest.raises(MCPError, match="modern MCP protocol"):
        await service.listen(
            context(Mock(spec=ServerSession), revision),
            SubscriptionsListenRequestParams(notifications=SubscriptionFilter(tools_list_changed=True)),
            authorize,
            "alice",
        )
    authorize.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "multiple-resources", "denied"])
async def test_unsupported_targets_do_not_open_upstream_listeners(mode: str) -> None:
    client: Final = MCPClient(server_url="http://127.0.0.1:1/mcp", transport_type=MCPTransport.http)
    server: Final = MCPServer(
        server_id="upstream",
        name="upstream",
        url="http://localhost/mcp",
        transport="http",
        protocol_version="2025-11-25" if mode == "legacy" else "2026-07-28",
    )
    target: Final = subscriptions.SubscriptionTarget(server, client, "synthetic-fingerprint")
    targets: Final = () if mode == "denied" else (target, target) if mode == "multiple-resources" else (target,)

    async def resolve(_: OperationContext) -> tuple[subscriptions.SubscriptionTarget, ...]:
        return targets

    service: Final = subscriptions.GatewaySubscriptions(resolve_targets=resolve)
    request: Final = SubscriptionsListenRequestParams(
        notifications=SubscriptionFilter(resource_subscriptions=["test://value"])
    )
    error: Final = HTTPException if mode == "denied" else MCPError
    expected: Final = {
        "legacy": "Select modern upstream",
        "multiple-resources": "exactly one allowed server",
        "denied": "No authorized MCP subscription targets",
    }[mode]
    with pytest.raises(error, match=expected):
        await service.listen(
            context(Mock(spec=ServerSession)), request, AsyncMock(return_value=OperationContext(_caller=None)), "alice"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("second_owner", ["alice", "bob"])
async def test_listener_limits_release_capacity_after_cancellation(second_owner: str) -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def prepare(_: OperationContext) -> tuple[subscriptions.SubscriptionTarget, ...]:
        entered.set()
        await release.wait()
        return ()

    prepare_mock: Final = AsyncMock(side_effect=prepare)
    service: Final = subscriptions.GatewaySubscriptions(max_listeners=1, max_per_caller=1, resolve_targets=prepare_mock)
    request: Final = SubscriptionsListenRequestParams(notifications=SubscriptionFilter(tools_list_changed=True))
    authorize: Final = AsyncMock(return_value=OperationContext(_caller=None))
    ctx: Final = context(Mock(spec=ServerSession))
    first: Final = asyncio.create_task(service.listen(ctx, request, authorize, "alice"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(MCPError, match=r"[Ss]ubscription limit reached"):
            await service.listen(ctx, request, authorize, second_owner)
        assert prepare_mock.await_count == 1
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    release.set()
    with pytest.raises(HTTPException, match="No authorized MCP subscription targets"):
        await service.listen(ctx, request, authorize, second_owner)
    assert prepare_mock.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_mode", [False, True])
async def test_subscription_requires_a_supported_http_gateway_route(proxy_mode: bool) -> None:
    from litellm.proxy._experimental.mcp_server import server

    token = server._mcp_proxy_mode.set(proxy_mode)
    try:
        error = MCPError if proxy_mode else HTTPException
        with pytest.raises(error) as rejected:
            await server.listen_subscriptions(
                context(Mock(spec=ServerSession)),
                SubscriptionsListenRequestParams(notifications=SubscriptionFilter(tools_list_changed=True)),
            )
        if proxy_mode:
            assert rejected.value.code == -32601
            assert "/mcp/proxy" in str(rejected.value)
        else:
            assert rejected.value.status_code == 400
            assert "HTTP request" in rejected.value.detail
    finally:
        server._mcp_proxy_mode.reset(token)

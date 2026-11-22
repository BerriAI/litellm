import asyncio
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from mcp import MCPError
from mcp.server.context import ServerRequestContext
from mcp.server.session import ServerSession
from mcp.server.subscriptions import ListenHandler
from mcp.shared.subscriptions import PromptsListChanged, ToolsListChanged
from mcp.types import ListToolsRequest, ListToolsResult, SubscriptionFilter, SubscriptionsListenRequestParams

from litellm.mcp_client.client import MCPClient
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server import subscriptions
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def allow_one_request(monkeypatch: pytest.MonkeyPatch, user_id: str, server_id: str) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

    windows: Final = {(user_id, server_id): asyncio.Semaphore(1)}

    async def enforce(user: UserAPIKeyAuth | None, server: MCPServer) -> None:
        window: Final = windows[(user.user_id if user is not None else None, server.server_id)]
        if window.locked():
            raise ProxyRateLimitError(detail=f"{server_id} RPM exceeded")
        await window.acquire()

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(enforce_mcp_server_rate_limits=enforce))


def context(session: ServerSession, revision: str = "2026-07-28") -> ServerRequestContext:
    return ServerRequestContext(
        session=session,
        lifespan_context=None,
        protocol_version=revision,
        method="subscriptions/listen",
        request_id="listener",
    )


class _ListingGateway:
    def __init__(self, *, result: ListToolsResult | None = None, error: MCPError | None = None) -> None:
        self._result: ListToolsResult | None = result
        self._error: MCPError | None = error

    async def execute(self, _operation: ListToolsRequest, _context: OperationContext) -> ListToolsResult:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


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
@pytest.mark.parametrize("mode", ["stdio", "multiple-resources", "denied"])
async def test_unsupported_targets_do_not_open_upstream_listeners(mode: str) -> None:
    client: Final = MCPClient(server_url="http://127.0.0.1:1/mcp", transport_type=MCPTransport.http)
    server: Final = MCPServer(
        server_id="upstream",
        name="upstream",
        url="http://localhost/mcp",
        transport="stdio" if mode == "stdio" else "http",
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
        "stdio": "Modern upstream subscriptions require HTTP",
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
    from litellm.proxy.mcp_server import server

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


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [None, "user_id", "team_id", "org_id", "end_user_id", "api_key"])
async def test_subscription_authorization_preserves_original_caller(changed: str | None) -> None:
    from dataclasses import replace

    from litellm.proxy.mcp_server.server import _owner_fingerprint_for

    initial = OperationContext(_caller=UserAPIKeyAuth(api_key="first", user_id="alice", team_id="team"))
    current = replace(
        initial,
        _caller=initial.user_api_key_auth.model_copy(update={changed: "different"})
        if changed
        else initial.user_api_key_auth,
    )

    async def refresh() -> OperationContext:
        return current

    authorization = subscriptions.SubscriptionAuthorization(initial, refresh, _owner_fingerprint_for)
    if changed:
        with pytest.raises(HTTPException, match="Subscription caller changed"):
            await authorization()
    else:
        assert await authorization() is current
        assert authorization.owner == _owner_fingerprint_for(initial.user_api_key_auth)


@pytest.mark.asyncio
async def test_legacy_events_filter_resource_updates_and_unsubscribe_on_exit() -> None:
    from mcp.shared.subscriptions import ResourceUpdated
    from mcp.types import (
        ResourcesCapability,
        ResourceUpdatedNotification,
        ResourceUpdatedNotificationParams,
        ServerCapabilities,
        ToolListChangedNotification,
        ToolsCapability,
    )

    session = Mock()
    session.server_capabilities = ServerCapabilities(
        tools=ToolsCapability(list_changed=True), resources=ResourcesCapability(subscribe=True)
    )
    session.send_request = AsyncMock()
    events = subscriptions.LegacyNotifications()
    requested = SubscriptionFilter(
        tools_list_changed=True, prompts_list_changed=True, resource_subscriptions=["test://value"]
    )
    async with events.listen(session, requested) as listener:
        assert listener.honored == SubscriptionFilter(tools_list_changed=True, resource_subscriptions=["test://value"])
        await events.receive(ResourceUpdatedNotification(params=ResourceUpdatedNotificationParams(uri="test://other")))
        await events.receive(ToolListChangedNotification())
        await events.receive(ResourceUpdatedNotification(params=ResourceUpdatedNotificationParams(uri="test://value")))
        assert await anext(listener) == ToolsListChanged()
        assert await anext(listener) == ResourceUpdated(uri="test://value")
    assert [(call.args[0].method, str(call.args[0].params.uri)) for call in session.send_request.await_args_list] == [
        ("resources/subscribe", "test://value"),
        ("resources/unsubscribe", "test://value"),
    ]


@pytest.mark.asyncio
async def test_legacy_notification_overflow_closes_instead_of_silently_losing_changes() -> None:
    from mcp.types import ServerCapabilities, ToolListChangedNotification, ToolsCapability

    session = Mock()
    session.server_capabilities = ServerCapabilities(tools=ToolsCapability(list_changed=True))
    events = subscriptions.LegacyNotifications(max_buffered_events=1)
    async with events.listen(session, SubscriptionFilter(tools_list_changed=True)) as listener:
        await events.receive(ToolListChangedNotification())
        await events.receive(ToolListChangedNotification())
        with pytest.raises(MCPError, match="overflow"):
            await anext(listener)


@pytest.mark.asyncio
async def test_partial_legacy_resource_setup_only_unsubscribes_accepted_resources() -> None:
    from mcp.types import ResourcesCapability, ServerCapabilities

    session = Mock()
    session.server_capabilities = ServerCapabilities(resources=ResourcesCapability(subscribe=True))
    session.send_request = AsyncMock(side_effect=[None, MCPError(code=-32602, message="Denied resource"), None, None])
    events = subscriptions.LegacyNotifications()
    with pytest.raises(MCPError, match="Denied resource"):
        async with events.listen(
            session, SubscriptionFilter(resource_subscriptions=["test://allowed", "test://denied"])
        ):
            pytest.fail("Rejected setup must not acknowledge a listener")
    assert [(call.args[0].method, str(call.args[0].params.uri)) for call in session.send_request.await_args_list] == [
        ("resources/subscribe", "test://allowed"),
        ("resources/subscribe", "test://denied"),
        ("resources/unsubscribe", "test://allowed"),
    ]


@pytest.mark.asyncio
async def test_legacy_initialization_waiters_are_session_local_and_released_on_failure() -> None:
    import anyio

    service = subscriptions.GatewaySubscriptions()
    first = context(Mock(spec=ServerSession), "2025-11-25")
    second = context(Mock(spec=ServerSession), "2025-11-25")
    async with service.initializing(first) as started:
        assert started
        async with service.initializing(first) as duplicate:
            assert not duplicate
        waiter = asyncio.create_task(service.wait_initialized(first))
        await anyio.lowlevel.checkpoint()
        assert not waiter.done()
        await service.wait_initialized(second)
        assert not waiter.done()
    await waiter
    async with service.initializing(first) as restarted:
        assert restarted


@pytest.mark.asyncio
async def test_sessionless_legacy_resource_subscription_is_rejected() -> None:
    session = Mock(spec=ServerSession)
    session.client_params = None
    service = subscriptions.GatewaySubscriptions()
    with pytest.raises(MCPError, match="live legacy session"):
        await service.legacy_resource(context(session, "2025-11-25"), "test://value", subscribe=True)


@pytest.mark.asyncio
async def test_legacy_upstream_without_capabilities_does_not_receive_subscribe() -> None:
    session = Mock()
    session.server_capabilities = None
    session.send_request = AsyncMock()
    events = subscriptions.LegacyNotifications()
    async with events.listen(
        session, SubscriptionFilter(tools_list_changed=True, resource_subscriptions=["test://value"])
    ) as listener:
        assert listener.honored == SubscriptionFilter()
    session.send_request.assert_not_awaited()


@pytest.mark.parametrize("proxy_mode", [False, True])
def test_legacy_notification_capabilities_require_a_notification_route(proxy_mode: bool) -> None:
    from litellm.proxy.mcp_server import server

    token = server._mcp_proxy_mode.set(proxy_mode)
    try:
        capabilities = server.server.create_initialization_options().capabilities
        assert capabilities.tools.list_changed is (not proxy_mode)
        if proxy_mode:
            assert capabilities.resources is None
    finally:
        server._mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
async def test_idle_legacy_listener_detects_closed_upstream_without_restarting_work(monkeypatch) -> None:
    import anyio

    session = Mock()
    session.server_capabilities = None
    session.send_ping = AsyncMock(side_effect=OSError("Upstream closed"))
    deadline = anyio.move_on_after
    monkeypatch.setattr(subscriptions.anyio, "move_on_after", lambda seconds, **kw: deadline(0.001, **kw))
    events = subscriptions.LegacyNotifications()
    with pytest.raises(OSError, match="Upstream closed"):
        async with events.listen(session, SubscriptionFilter()) as listener:
            await anext(listener)
    session.send_ping.assert_awaited_once()
    session.send_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["subscribe_resource", "unsubscribe_resource"])
async def test_proxy_route_rejects_legacy_resource_subscription_before_setup(method: str) -> None:
    from mcp.types import SubscribeRequestParams, UnsubscribeRequestParams

    from litellm.proxy.mcp_server import server

    params = (SubscribeRequestParams if method == "subscribe_resource" else UnsubscribeRequestParams)(uri="test://value")
    token = server._mcp_proxy_mode.set(True)
    try:
        with pytest.raises(MCPError, match="/mcp/proxy"):
            await getattr(server, method)(context(Mock(spec=ServerSession)), params)
    finally:
        server._mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_requests", "tools_list_admitted"),
    [((), True), (("subscriptions/listen",), False), (("resources/subscribe",), False)],
)
async def test_only_client_initiated_listeners_consume_server_rate_limit(
    monkeypatch: pytest.MonkeyPatch, client_requests: tuple[str, ...], tools_list_admitted: bool
) -> None:
    from litellm.proxy.mcp_server import operations

    allow_one_request(monkeypatch, "alice", "upstream")
    opened: Final = asyncio.Queue[None]()

    async def hold_upstream(*_: object, **__: object) -> None:
        opened.put_nowait(None)
        await asyncio.Event().wait()

    client: Final = Mock(spec=MCPClient)
    client.run_with_session = AsyncMock(side_effect=hold_upstream)
    server: Final = MCPServer(server_id="upstream", name="upstream", url="http://localhost/mcp", transport="http")
    target: Final = subscriptions.SubscriptionTarget(server, client, "synthetic-fingerprint")

    async def resolve(_: OperationContext) -> tuple[subscriptions.SubscriptionTarget, ...]:
        return (target,)

    service: Final = subscriptions.GatewaySubscriptions(resolve_targets=resolve)
    caller: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="alice"))
    authorize: Final = AsyncMock(return_value=caller)
    session: Final = Mock(spec=ServerSession)
    session.client_params = Mock()
    legacy: Final = context(session, "2025-11-25")
    requests: Final = {
        "subscriptions/listen": lambda: service.listen(
            context(Mock(spec=ServerSession)),
            SubscriptionsListenRequestParams(notifications=SubscriptionFilter(tools_list_changed=True)),
            authorize,
            "alice",
        ),
        "resources/subscribe": lambda: service.legacy_resource(legacy, "test://value", subscribe=True),
    }
    authorization: Final = subscriptions.SubscriptionAuthorization(caller, authorize, lambda *_: "alice")
    starts: Final = (lambda: service.legacy_session(legacy, authorization), *(requests[r] for r in client_requests))
    tasks: Final = tuple(asyncio.create_task(start()) for start in starts)
    try:
        for _ in tasks:
            await asyncio.wait_for(opened.get(), 1)
        rejection: Final = await operations._mcp_server_rate_limit_rejection(server, caller.user_api_key_auth)
        assert (rejection is None) is tools_list_admitted
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_listener_catalog_refreshes_do_not_consume_server_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import ListToolsResult, ServerCapabilities, Tool, ToolsCapability

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import catalog, operations
    from litellm.proxy.mcp_server.faults.list_outcomes import ServerListOk

    server: Final = MCPServer(server_id="list-changed", name="list-changed", transport=MCPTransport.http)
    caller: Final = UserAPIKeyAuth(api_key="list-changed-key", user_id="alice")
    allow_one_request(monkeypatch, "alice", "list-changed")
    listed: Final = asyncio.Queue[None]()

    async def fetch_tools(_: MCPServer, **__: object) -> tuple[ListToolsResult, ServerListOk]:
        listed.put_nowait(None)
        return ListToolsResult(tools=[Tool(name="add", inputSchema={"type": "object"})]), ServerListOk(tool_count=1)

    monkeypatch.setenv("LITELLM_SALT_KEY", "list-changed-test-key")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations.global_mcp_server_manager, "registry", {server.server_id: server})
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server]))
    monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch_tools)

    upstream: Final = Mock()
    upstream.server_capabilities = ServerCapabilities(tools=ToolsCapability(list_changed=True))
    upstream.protocol_version = "2025-11-25"
    upstream.send_ping = AsyncMock()

    async def run_with_session(connected, **_: object) -> None:
        await connected(upstream)

    client: Final = Mock(spec=MCPClient)
    client.run_with_session = AsyncMock(side_effect=run_with_session)
    target: Final = subscriptions.SubscriptionTarget(server, client, "synthetic-fingerprint")

    async def resolve(_: OperationContext) -> tuple[subscriptions.SubscriptionTarget, ...]:
        return (target,)

    initial: Final = operations.prepare_context(user_api_key_auth=caller, mcp_servers=[server.server_id])
    authorize: Final = AsyncMock(return_value=initial)
    session: Final = Mock(spec=ServerSession)
    session.client_params = Mock()
    service: Final = subscriptions.GatewaySubscriptions(resolve_targets=resolve)
    listener: Final = asyncio.create_task(
        service.legacy_session(
            context(session, "2025-11-25"),
            subscriptions.SubscriptionAuthorization(initial, authorize, lambda *_: "alice"),
        )
    )
    try:
        for _ in range(2):
            await asyncio.wait_for(listed.get(), 5)
        assert await operations._mcp_server_rate_limit_rejection(server, caller) is None
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


@pytest.mark.asyncio
async def test_all_failed_tools_listing_keeps_empty_catalog_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.mcp_server.catalog import ToolListingFailedOutright

    listener_context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="listener"))
    empty_gateway: Final = _ListingGateway(result=ListToolsResult(tools=[]))
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.operations.GatewayOperations", lambda: empty_gateway
    )
    empty_digest: Final = await subscriptions._listing(ToolsListChanged(), listener_context)

    failed_gateway: Final = _ListingGateway(
        error=ToolListingFailedOutright(code=-32603, message="Every MCP server in scope failed")
    )
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.operations.GatewayOperations", lambda: failed_gateway
    )
    failed_digest: Final = await subscriptions._listing(ToolsListChanged(), listener_context)

    assert failed_digest == empty_digest


@pytest.mark.asyncio
async def test_tools_listing_propagates_other_mcp_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = MCPError(code=-32602, message="Bad cursor")
    gateway: Final = _ListingGateway(error=failure)
    monkeypatch.setattr("litellm.proxy.mcp_server.operations.GatewayOperations", lambda: gateway)
    listener_context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="listener"))

    expected_error: Final = pytest.raises(MCPError)
    with expected_error as exc_info:
        await subscriptions._listing(ToolsListChanged(), listener_context)

    assert exc_info.value is failure


@pytest.mark.asyncio
async def test_toolset_subscription_authorization_preserves_scope_and_dashboard_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request as StarletteRequest

    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.mcp_server import operations, server, ui_session_utils

    granted_toolsets: Final = {"ts-1"}
    toolset_perms: Final = {"srv-portal": ["stages_update_stage_forecast"]}
    dashboard_user: Final = UserAPIKeyAuth(
        user_id="user-1",
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="user-op",
            mcp_servers=["srv-portal", "srv-stalled"],
        ),
    )

    async def fake_granted(_user: UserAPIKeyAuth) -> frozenset[str]:
        return frozenset(granted_toolsets)

    async def fake_acting(user: UserAPIKeyAuth) -> UserAPIKeyAuth:
        return user.model_copy(update={"user_id": "user-1", "team_id": None})

    async def fake_extract(_scope: object, _path: str) -> tuple[
        UserAPIKeyAuth,
        None,
        None,
        None,
        None,
        object,
    ]:
        return (dashboard_user.model_copy(deep=True), None, None, None, None, {})

    monkeypatch.setattr(ui_session_utils, "acting_user_auth", fake_acting)
    monkeypatch.setattr(ui_session_utils, "granted_toolset_ids", fake_granted)
    monkeypatch.setattr(
        operations.global_mcp_server_manager,
        "resolve_toolset_tool_permissions",
        AsyncMock(return_value=toolset_perms),
    )
    monkeypatch.setattr(server, "extract_mcp_auth_context", fake_extract)
    monkeypatch.setattr(
        server,
        "apply_toolset_scope",
        lambda auth, ts_id: server._apply_toolset_scope(
            auth, ts_id, acting_user=fake_acting, granted=fake_granted
        ),
    )

    scoped_initial: Final = await server.apply_toolset_scope(dashboard_user, "ts-1")
    session: Final = Mock(spec=ServerSession)
    session.client_params = Mock()
    http_request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    ctx: Final = ServerRequestContext(
        session=session,
        lifespan_context=None,
        protocol_version="2025-11-25",
        method="notifications/initialized",
        request_id=None,
        request=http_request,
    )

    auth_token: Final = server.auth_context_var.set(
        server.MCPAuthenticatedUser(
            user_api_key_auth=scoped_initial,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={},
            client_ip="127.0.0.1",
        )
    )
    try:
        async with server._subscription_authorization(ctx) as authorization:
            refreshed: Final = await authorization()
            assert refreshed.user_api_key_auth is not None
            assert refreshed.user_api_key_auth.mcp_toolset_id == "ts-1"
            assert refreshed.user_api_key_auth.object_permission is not None
            assert refreshed.user_api_key_auth.object_permission.mcp_servers == ["srv-portal"]
            assert refreshed.user_api_key_auth.object_permission.mcp_tool_permissions == toolset_perms

            granted_toolsets.clear()
            with pytest.raises(HTTPException) as denied:
                await authorization()
            assert denied.value.status_code == 403
    finally:
        server.auth_context_var.reset(auth_token)


@pytest.mark.asyncio
async def test_toolset_legacy_tools_list_ignores_unrelated_stalled_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request as StarletteRequest

    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.mcp_server import catalog, operations, server
    from litellm.proxy.mcp_server.faults.list_outcomes import ServerListOk

    portal: Final = MCPServer(
        server_id="srv-portal",
        name="portal",
        url="http://portal.local/mcp",
        transport=MCPTransport.http,
    )
    stalled: Final = MCPServer(
        server_id="srv-stalled",
        name="stalled",
        url="http://stalled.local/mcp",
        transport=MCPTransport.http,
    )
    toolset_perms: Final = {portal.server_id: ["stages_update_stage_forecast"]}
    base_auth: Final = UserAPIKeyAuth(
        api_key="sk-toolset-test",
        user_id="user-1",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-toolset",
            mcp_servers=[portal.server_id, stalled.server_id],
            mcp_toolsets=["ts-1"],
        ),
    )
    connects: Final[list[str]] = []  # mutable-ok: append-only log of upstream connection order
    release_stalled: Final = asyncio.Event()

    class _FakeUpstreamClient:
        def __init__(self, srv: MCPServer) -> None:
            self._server = srv

        async def subscription_auth_fingerprint(self) -> str:
            return "fp"

        async def run_with_session(
            self,
            operation,
            on_notification: object = None,
        ) -> None:
            from mcp.types import ServerCapabilities, ToolsCapability

            _ = on_notification
            connects.append(self._server.name)
            if self._server.server_id == stalled.server_id:
                await release_stalled.wait()
                return
            upstream: Final = SimpleNamespace(
                server_capabilities=ServerCapabilities(tools=ToolsCapability(list_changed=True)),
                protocol_version="2025-11-25",
                send_ping=AsyncMock(),
            )
            await operation(upstream)

    async def fake_allowed(
        user_api_key_auth: UserAPIKeyAuth | None = None,
        mcp_servers: object = None,
        client_ip: str | None = None,
    ) -> tuple[MCPServer, ...]:
        _ = (mcp_servers, client_ip)
        if user_api_key_auth and user_api_key_auth.object_permission and user_api_key_auth.object_permission.mcp_servers:
            allowed_ids: Final = frozenset(user_api_key_auth.object_permission.mcp_servers)
            return tuple(s for s in (portal, stalled) if s.server_id in allowed_ids)
        return (portal, stalled)

    async def fake_extract(_scope: object, _path: str) -> tuple[
        UserAPIKeyAuth,
        None,
        None,
        None,
        None,
        object,
    ]:
        return (base_auth.model_copy(deep=True), None, None, None, None, {})

    async def fake_tools(
        _srv: MCPServer,
        **_kwargs: object,  # kwargs-ok: matches catalog.get_filtered_server_tools mock contract
    ) -> tuple[ListToolsResult, ServerListOk]:
        from mcp.types import Tool

        return (
            ListToolsResult(tools=[Tool(name="stages_update_stage_forecast", inputSchema={"type": "object"})]),
            ServerListOk(tool_count=1),
        )

    monkeypatch.setattr(operations.global_mcp_server_manager, "registry", {portal.server_id: portal, stalled.server_id: stalled})
    monkeypatch.setattr(
        operations.global_mcp_server_manager,
        "resolve_toolset_tool_permissions",
        AsyncMock(return_value=toolset_perms),
    )
    monkeypatch.setattr(
        operations.global_mcp_server_manager,
        "create_mcp_client",
        AsyncMock(side_effect=lambda srv, **__: _FakeUpstreamClient(srv)),
    )
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", fake_allowed)
    monkeypatch.setattr(catalog, "get_filtered_server_tools", fake_tools)
    monkeypatch.setattr(server, "extract_mcp_auth_context", fake_extract)

    scoped_auth: Final = await server.apply_toolset_scope(base_auth, "ts-1")
    session: Final = Mock(spec=ServerSession)
    session.client_params = Mock()
    http_request: Final = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    init_ctx: Final = ServerRequestContext(
        session=session,
        lifespan_context=None,
        protocol_version="2025-11-25",
        method="notifications/initialized",
        request_id=None,
        request=http_request,
    )
    list_ctx: Final = ServerRequestContext(
        session=session,
        lifespan_context=None,
        protocol_version="2025-11-25",
        method="tools/list",
        request_id="req-list",
        request=http_request,
    )

    from mcp.types import NotificationParams, PaginatedRequestParams

    auth_token: Final = server.auth_context_var.set(
        server.MCPAuthenticatedUser(
            user_api_key_auth=scoped_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={},
            client_ip="127.0.0.1",
        )
    )
    init_task: Final = asyncio.create_task(server.legacy_notifications(init_ctx, NotificationParams()))
    try:
        result: Final = await asyncio.wait_for(
            server.handle_list_tools(list_ctx, PaginatedRequestParams()),
            timeout=2.0,
        )
        assert [tool.name for tool in result.tools] == ["stages_update_stage_forecast"]
        assert connects == ["portal"]
    finally:
        release_stalled.set()
        init_task.cancel()
        await asyncio.gather(init_task, return_exceptions=True)
        server.auth_context_var.reset(auth_token)

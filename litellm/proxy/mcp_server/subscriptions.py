import asyncio
import hashlib
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from itertools import chain
from typing import Final, NoReturn

import anyio
from anyio.abc import TaskGroup, TaskStatus
from fastapi import HTTPException
from mcp import ClientSession, MCPError
from mcp.client.subscriptions import listen
from mcp.server.context import ServerRequestContext
from mcp.server.session import ServerSession
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler
from mcp.shared.subscriptions import (
    PromptsListChanged,
    ResourcesListChanged,
    ResourceUpdated,
    ServerEvent,
    ToolsListChanged,
    event_from_wire,
    event_matches,
    event_to_notification,
)
from mcp.types import (
    EmptyResult,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    PromptsCapability,
    RequestId,
    ResourcesCapability,
    ServerCapabilities,
    ServerNotification,
    SubscribeRequest,
    SubscribeRequestParams,
    SubscriptionFilter,
    SubscriptionsListenRequestParams,
    SubscriptionsListenResult,
    ToolsCapability,
    UnsubscribeRequest,
    UnsubscribeRequestParams,
)

from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES, MCP_TOOL_LISTING_TIMEOUT
from litellm.mcp_client.client import MCPClient
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server.catalog import (
    ToolListingFailedOutright,
)
from litellm.proxy.mcp_server.catalog import (
    _configuration_identity as configuration_identity,  # pyright: ignore[reportPrivateUsage]  # reuse catalog-owned configuration identity without treating discovered metadata as configuration
)
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.types.mcp import MCP_LEGACY_VERSIONS
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class SubscriptionAuthorization:
    def __init__(
        self,
        initial: OperationContext,
        refresh: Callable[[], Awaitable[OperationContext]],
        fingerprint: Callable[[UserAPIKeyAuth | None, Mapping[str, str] | None, str | None], str],
    ) -> None:
        self._refresh = refresh
        self._fingerprint = fingerprint
        self.owner = self._owner(initial)
        self._identity = self._caller(initial)

    def _owner(self, context: OperationContext) -> str:
        return self._fingerprint(context.user_api_key_auth, dict(context.oauth2_headers or {}), context.client_ip)

    @staticmethod
    def _caller(context: OperationContext) -> tuple[str | None, str | None, str | None, str | None] | None:
        caller: Final = context.user_api_key_auth
        return (caller.user_id, caller.team_id, caller.org_id, caller.end_user_id) if caller is not None else None

    async def __call__(self) -> OperationContext:
        fresh: Final = await self._refresh()
        if self._caller(fresh) != self._identity or self._owner(fresh) != self.owner:
            raise HTTPException(status_code=403, detail="Subscription caller changed")
        return fresh


def _supported_notifications(requested: SubscriptionFilter, capabilities: ServerCapabilities) -> SubscriptionFilter:
    return SubscriptionFilter(
        tools_list_changed=bool(requested.tools_list_changed and capabilities.tools and capabilities.tools.list_changed)
        or None,
        prompts_list_changed=bool(
            requested.prompts_list_changed and capabilities.prompts and capabilities.prompts.list_changed
        )
        or None,
        resources_list_changed=bool(
            requested.resources_list_changed and capabilities.resources and capabilities.resources.list_changed
        )
        or None,
        resource_subscriptions=list(requested.resource_subscriptions or ())
        if capabilities.resources and capabilities.resources.subscribe
        else None,
    )


class LegacyNotifications:
    def __init__(self, *, max_buffered_events: int = 128) -> None:
        self.honored = SubscriptionFilter()
        self._session: ClientSession | None = None
        self._events: Final[asyncio.Queue[ServerEvent | MCPError]] = asyncio.Queue(max_buffered_events)

    async def receive(self, notification: ServerNotification) -> None:
        event: Final = event_from_wire(
            notification.method,
            notification.params.model_dump(by_alias=True) if notification.params is not None else None,
        )
        if event is None or not event_matches(
            self.honored, frozenset(self.honored.resource_subscriptions or ()), event
        ):
            return
        self.publish(event)

    def publish(self, event: ServerEvent) -> None:
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            self._events.get_nowait()
            self._events.put_nowait(MCPError(code=-32001, message="Legacy notification buffer overflow; refresh"))

    def __aiter__(self) -> "LegacyNotifications":
        return self

    async def __anext__(self) -> ServerEvent:
        while True:
            with anyio.move_on_after(15 if self._session is not None else float("inf")) as idle:
                event = await self._events.get()
                if isinstance(event, MCPError):
                    raise event
                return event
            if idle.cancel_called and self._session is not None:
                await self._session.send_ping()

    @asynccontextmanager
    async def listen(
        self, session: ClientSession, requested: SubscriptionFilter
    ) -> AsyncGenerator["LegacyNotifications"]:
        self._session = session
        capabilities: Final = session.server_capabilities or ServerCapabilities()
        self.honored = _supported_notifications(requested, capabilities)
        cleanup: Final = AsyncExitStack()
        try:
            for uri in self.honored.resource_subscriptions or ():
                await session.send_request(SubscribeRequest(params=SubscribeRequestParams(uri=uri)), EmptyResult)
                cleanup.push_async_callback(
                    session.send_request, UnsubscribeRequest(params=UnsubscribeRequestParams(uri=uri)), EmptyResult
                )
            yield self
        finally:
            with anyio.move_on_after(5, shield=True):
                await cleanup.aclose()
            self._session = None


@dataclass(frozen=True)
class SubscriptionTarget:
    server: MCPServer
    client: MCPClient
    credential: str


class _AuthorizedSession(ServerSession):
    def __init__(self, session: ServerSession, authorize: Callable[[], Awaitable[OperationContext]]) -> None:
        self._session = session
        self._authorize = authorize

    async def send_notification(
        self, notification: ServerNotification, related_request_id: RequestId | None = None
    ) -> None:
        await self._authorize()
        await self._session.send_notification(notification, related_request_id)


class _ListenerBus(InMemorySubscriptionBus):
    def __init__(self) -> None:
        super().__init__()
        self.ready = asyncio.Event()

    def subscribe(self, listener: Callable[[ServerEvent], None]) -> Callable[[], None]:
        unsubscribe: Final = super().subscribe(listener)
        self.ready.set()
        return unsubscribe


async def _targets(context: OperationContext) -> tuple[SubscriptionTarget, ...]:
    manager: Final = operations.global_mcp_server_manager
    auth, token, servers, server_headers, oauth_headers, headers, ip = context.legacy_auth()
    async with manager.catalog.operation():
        allowed: Final = await operations._get_allowed_mcp_servers(auth, servers, ip)

        async def prepare(server: MCPServer) -> SubscriptionTarget:
            server_auth, extra = operations._prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=server_headers,
                mcp_auth_header=token,
                oauth2_headers=oauth_headers,
                raw_headers=headers,
                user_api_key_auth=auth,
                scope_servers=list(allowed),
            )
            client: Final = await manager.create_mcp_client(
                server,
                mcp_auth_header=server_auth,
                extra_headers={**(extra or {}), **(server.static_headers or {})},
                stdio_env=manager.build_stdio_env(server, headers),
                subject_token=manager._obo_subject_token(server, headers, auth),  # pyright: ignore[reportPrivateUsage]  # reuse the manager-owned OBO policy
                user_api_key_auth=auth,
                raw_headers=headers,
                client_ip=ip,
            )
            return SubscriptionTarget(server, client, await client.subscription_auth_fingerprint())

        return tuple([await prepare(server) for server in allowed])


async def subscription_capabilities(context: OperationContext) -> tuple[frozenset[str], ServerCapabilities]:
    try:
        targets: Final = await _targets(context)
    except Exception:  # noqa: BLE001  # missing optional upstream credentials must not erase usable catalog results
        return frozenset(MCP_LEGACY_VERSIONS), ServerCapabilities()
    configured: Final = frozenset(target.server.protocol_version for target in targets)
    versions: Final = (
        configured | frozenset(MCP_LEGACY_VERSIONS) if not configured or "auto" in configured else configured
    )
    if not targets or any(
        target.server.protocol_version == "2026-07-28" and target.server.transport != "http" for target in targets
    ):
        return versions, ServerCapabilities()

    async def probe(target: SubscriptionTarget) -> ServerCapabilities:
        async def read(session: ClientSession) -> ServerCapabilities:
            return session.server_capabilities or ServerCapabilities()

        try:
            if await operations._mcp_server_rate_limit_rejection(target.server, context.user_api_key_auth) is not None:  # pyright: ignore[reportPrivateUsage]  # reuse request-scoped catalog admission
                return ServerCapabilities()
            with anyio.fail_after(MCP_TOOL_LISTING_TIMEOUT):
                return await target.client.run_with_session(read, quiet_on_error=True)
        except Exception:  # noqa: BLE001  # an optional notification probe must not disable an otherwise usable catalog
            return ServerCapabilities()

    declared: Final = tuple(await asyncio.gather(*(probe(target) for target in targets)))
    return versions, ServerCapabilities(
        tools=ToolsCapability(list_changed=True)
        if all(item.tools and item.tools.list_changed for item in declared)
        else None,
        prompts=PromptsCapability(list_changed=True)
        if all(item.prompts and item.prompts.list_changed for item in declared)
        else None,
        resources=ResourcesCapability(
            list_changed=all(item.resources and item.resources.list_changed for item in declared) or None,
            subscribe=(len(targets) == 1 and any(item.resources and item.resources.subscribe for item in declared))
            or None,
        )
        if any(item.resources for item in declared)
        else None,
    )


async def _listing(event: ServerEvent, context: OperationContext) -> str | None:
    if isinstance(event, ResourceUpdated):
        return None
    operation: Final = (
        ListToolsRequest()
        if isinstance(event, ToolsListChanged)
        else ListPromptsRequest()
        if isinstance(event, PromptsListChanged)
        else ListResourcesRequest()
    )
    gateway: Final = operations.GatewayOperations()
    digest: Final = hashlib.sha256()
    cursor: str | None = None  # rebind-ok: advance the signed gateway cursor for each catalog page
    with anyio.fail_after(MCP_TOOL_LISTING_TIMEOUT):
        for _ in range(MCP_TOOL_LISTING_MAX_PAGES):
            try:
                result = await gateway.execute(
                    operation.model_copy(update={"params": PaginatedRequestParams(cursor=cursor)}), context
                )
            except ToolListingFailedOutright:
                return digest.hexdigest()
            items = (
                result.tools
                if isinstance(result, ListToolsResult)
                else result.prompts
                if isinstance(result, ListPromptsResult)
                else result.resources
            )
            for item in items:
                digest.update(item.model_dump_json().encode())
            cursor = result.next_cursor
            if cursor is None:
                return digest.hexdigest()
    raise MCPError(code=-32603, message="Subscription catalog exceeds the pagination limit")


async def _enforce_rate_limit(context: OperationContext, server: MCPServer) -> None:
    await operations._enforce_mcp_server_rate_limit(context.user_api_key_auth, server)  # pyright: ignore[reportPrivateUsage]  # same admission policy as other gateway operations


async def _unmetered(_: OperationContext, __: MCPServer) -> None:
    return None


async def _admitted() -> ProxyRateLimitError | None:
    return None


@asynccontextmanager
async def _listener_admission(targets: tuple[SubscriptionTarget, ...]) -> AsyncGenerator[None]:
    scope: Final = operations._mcp_server_admission_memo.set(  # pyright: ignore[reportPrivateUsage]  # the listener's own catalog refreshes reuse its admission
        {target.server.server_id: asyncio.create_task(_admitted()) for target in targets}
    )
    try:
        yield
    finally:
        operations._mcp_server_admission_memo.reset(scope)  # pyright: ignore[reportPrivateUsage]  # restore the caller's admission scope


async def _recheck_authorization(check: Callable[[], Awaitable[OperationContext]]) -> None:
    while True:
        await asyncio.sleep(15)
        await check()


async def _deliver_legacy(
    ctx: ServerRequestContext,
    bus: _ListenerBus,
    check: Callable[[], Awaitable[OperationContext]],
    notifications: SubscriptionFilter,
    ready: Callable[[SubscriptionFilter], None] | None,
) -> NoReturn:
    events: Final = LegacyNotifications()
    unsubscribe: Final = bus.subscribe(events.publish)
    try:
        if ready is not None:
            ready(notifications)
        while True:
            event = await anext(events)
            await check()
            await ctx.session.send_notification(event_to_notification(event, {}))
    finally:
        unsubscribe()


@dataclass
class LegacySubscriptionSession:
    group: TaskGroup
    authorization: SubscriptionAuthorization
    resources: Mapping[str, anyio.CancelScope]
    lock: asyncio.Lock


class GatewaySubscriptions:
    def __init__(
        self,
        *,
        max_listeners: int = 128,
        max_per_caller: int = 8,
        resolve_targets: Callable[[OperationContext], Awaitable[tuple[SubscriptionTarget, ...]]] = _targets,
    ) -> None:
        self._resolve_targets = resolve_targets
        self._slots = anyio.CapacityLimiter(max_listeners)
        self._upstream_slots = anyio.CapacityLimiter(max_listeners)
        self._max_per_caller = max_per_caller
        self._owners: tuple[str, ...] = ()
        self._legacy: Mapping[int, LegacySubscriptionSession] = {}
        self._legacy_started = asyncio.Event()
        self._initializing: Mapping[int, asyncio.Event] = {}

    async def listen(
        self,
        ctx: ServerRequestContext,
        params: SubscriptionsListenRequestParams,
        authorize: Callable[[], Awaitable[OperationContext]],
        owner: str,
    ) -> SubscriptionsListenResult:
        if ctx.protocol_version != "2026-07-28":
            raise MCPError(code=-32601, message="Subscriptions require the modern MCP protocol")
        async with self._capacity(owner):
            return await self._listen(ctx, params, authorize, _enforce_rate_limit)

    @asynccontextmanager
    async def _capacity(self, owner: str) -> AsyncGenerator[None]:
        if self._owners.count(owner) >= self._max_per_caller:
            raise MCPError(code=-32603, message="Caller subscription limit reached")
        try:
            self._slots.acquire_nowait()
        except anyio.WouldBlock as error:
            raise MCPError(code=-32603, message="Subscription limit reached") from error
        self._owners += (owner,)
        try:
            yield
        finally:
            position: Final = self._owners.index(owner)
            self._owners = self._owners[:position] + self._owners[position + 1 :]
            self._slots.release()

    @asynccontextmanager
    async def initializing(self, ctx: ServerRequestContext) -> AsyncGenerator[bool]:
        key: Final = id(ctx.session.client_params)
        if key in self._initializing:
            yield False
            return
        ready: Final = asyncio.Event()
        self._initializing = {**self._initializing, key: ready}
        try:
            yield True
        finally:
            ready.set()
            self._initializing = {
                registered: event for registered, event in self._initializing.items() if registered != key
            }

    async def wait_initialized(self, ctx: ServerRequestContext) -> None:
        ready: Final = self._initializing.get(id(ctx.session.client_params))
        if ready is not None:
            await ready.wait()

    async def legacy_session(self, ctx: ServerRequestContext, authorization: SubscriptionAuthorization) -> None:
        key: Final = id(ctx.session.client_params)
        if ctx.session.client_params is None or key in self._legacy:
            return
        async with anyio.create_task_group() as group:
            state: Final = LegacySubscriptionSession(group, authorization, {}, asyncio.Lock())
            self._legacy = {**self._legacy, key: state}
            self._legacy_started.set()
            try:
                await group.start(
                    self._legacy_listener,
                    ctx,
                    SubscriptionFilter(tools_list_changed=True, prompts_list_changed=True, resources_list_changed=True),
                    authorization,
                    _unmetered,
                )
                if (ready := self._initializing.get(key)) is not None:
                    ready.set()
                await anyio.sleep_forever()
            finally:
                self._legacy = {registered: state for registered, state in self._legacy.items() if registered != key}
                group.cancel_scope.cancel()

    async def _legacy_listener(
        self,
        ctx: ServerRequestContext,
        notifications: SubscriptionFilter,
        authorization: SubscriptionAuthorization,
        admit: Callable[[OperationContext, MCPServer], Awaitable[None]],
        scope: anyio.CancelScope | None = None,
        *,
        task_status: TaskStatus[None],
    ) -> None:
        with scope if scope is not None else anyio.CancelScope():

            def ready(honored: SubscriptionFilter) -> None:
                if (
                    notifications.resource_subscriptions
                    and honored.resource_subscriptions != notifications.resource_subscriptions
                ):
                    raise MCPError(code=-32602, message="Upstream does not support this resource subscription")
                task_status.started()

            async with self._capacity(authorization.owner):
                await self._listen(
                    ctx, SubscriptionsListenRequestParams(notifications=notifications), authorization, admit, ready
                )

    async def legacy_resource(self, ctx: ServerRequestContext, uri: str, *, subscribe: bool) -> None:
        if ctx.session.client_params is None:
            raise MCPError(code=-32602, message="Resource subscriptions require a live legacy session")
        key: Final = id(ctx.session.client_params)
        with anyio.fail_after(10):
            while key not in self._legacy:
                self._legacy_started.clear()
                await self._legacy_started.wait()
        state: Final = self._legacy[key]
        async with state.lock:
            await state.authorization()
            if subscribe and uri not in state.resources:
                scope: Final = anyio.CancelScope()
                await state.group.start(
                    self._legacy_listener,
                    ctx,
                    SubscriptionFilter(resource_subscriptions=[uri]),
                    state.authorization,
                    _enforce_rate_limit,
                    scope,
                )
                state.resources = {**state.resources, uri: scope}
            elif not subscribe and uri in state.resources:
                state.resources[uri].cancel()
                state.resources = {resource: scope for resource, scope in state.resources.items() if resource != uri}

    async def _listen(
        self,
        ctx: ServerRequestContext,
        params: SubscriptionsListenRequestParams,
        authorize: Callable[[], Awaitable[OperationContext]],
        admit: Callable[[OperationContext, MCPServer], Awaitable[None]],
        ready: Callable[[SubscriptionFilter], None] | None = None,
    ) -> SubscriptionsListenResult:
        context: Final = await authorize()
        targets: Final = await self._resolve_targets(context)
        if any(
            target.server.protocol_version == "2026-07-28" and target.server.transport != "http" for target in targets
        ):
            raise MCPError(code=-32602, message="Modern upstream subscriptions require HTTP")
        if not targets:
            raise HTTPException(status_code=403, detail="No authorized MCP subscription targets")
        if params.notifications.resource_subscriptions and len(targets) != 1:
            raise MCPError(code=-32602, message="Resource subscriptions require exactly one allowed server")
        bus: Final = _ListenerBus()
        handler: Final = ListenHandler(bus, max_subscriptions=1, max_buffered_events=128)

        async def check() -> OperationContext:
            fresh: Final = await authorize()
            current: Final = await self._resolve_targets(fresh)
            if tuple((configuration_identity(item.server), item.credential) for item in current) != tuple(
                (configuration_identity(item.server), item.credential) for item in targets
            ):
                raise MCPError(code=-32001, message="Subscription authorization or upstream configuration changed")
            return fresh

        async def relay(target: SubscriptionTarget, *, task_status: TaskStatus[SubscriptionFilter]) -> None:
            await admit(context, target.server)

            legacy: Final = LegacyNotifications()

            async def connected(session: ClientSession) -> None:
                requested: Final = params.notifications
                if ctx.protocol_version != "2026-07-28" and not any(
                    _supported_notifications(requested, session.server_capabilities or ServerCapabilities())
                    .model_dump()
                    .values()
                ):
                    task_status.started(SubscriptionFilter())
                    return
                requested_events: Final = tuple(
                    event
                    for event in (ToolsListChanged(), PromptsListChanged(), ResourcesListChanged())
                    if event_matches(requested, frozenset(), event)
                )
                baseline: Final = {event: await _listing(event, context) for event in requested_events}
                stream: Final = (
                    listen(
                        session,
                        tools_list_changed=bool(requested.tools_list_changed),
                        prompts_list_changed=bool(requested.prompts_list_changed),
                        resources_list_changed=bool(requested.resources_list_changed),
                        resource_subscriptions=requested.resource_subscriptions or (),
                    )
                    if session.protocol_version == "2026-07-28"
                    else legacy.listen(session, requested)
                )
                async with stream as subscription:
                    events: Final = tuple(
                        event for event in requested_events if event_matches(subscription.honored, frozenset(), event)
                    )
                    setup_context: Final = await check()
                    catalog: Final = operations.global_mcp_server_manager
                    catalog.invalidate_discovery_lists(target.server.server_id)
                    catalog._drop_listed_tools(target.server.server_id)  # pyright: ignore[reportPrivateUsage]  # refresh the setup snapshot through the same catalog owner as event delivery
                    previous: Final = {event: await _listing(event, setup_context) for event in events}
                    setup_changes: Final = tuple(event for event in events if baseline[event] != previous[event])
                    task_status.started(subscription.honored)
                    await bus.ready.wait()
                    for event in setup_changes:
                        await bus.publish(event)
                    async for event in subscription:
                        fresh = await check()
                        manager = operations.global_mcp_server_manager
                        manager.invalidate_discovery_lists(target.server.server_id)
                        manager._drop_listed_tools(target.server.server_id)  # pyright: ignore[reportPrivateUsage]  # evict served catalogs without invalidating OAuth metadata
                        listing = await _listing(event, fresh)
                        if not isinstance(event, ResourceUpdated):
                            if previous.get(event) == listing:
                                continue
                            previous[event] = listing
                        await bus.publish(event)
                    raise MCPError(code=-32001, message="Upstream subscription ended; listen again and refresh")

            try:
                self._upstream_slots.acquire_nowait()
            except anyio.WouldBlock as error:
                raise MCPError(code=-32603, message="Upstream subscription limit reached") from error
            try:
                await target.client.run_with_session(connected, on_notification=legacy.receive)
            finally:
                self._upstream_slots.release()

        async with _listener_admission(targets), anyio.create_task_group() as group:
            honored: Final = tuple(
                [SubscriptionFilter.model_validate(await group.start(relay, target)) for target in targets]
            )
            notifications: Final = SubscriptionFilter(
                tools_list_changed=any(item.tools_list_changed for item in honored) or None,
                prompts_list_changed=any(item.prompts_list_changed for item in honored) or None,
                resources_list_changed=any(item.resources_list_changed for item in honored) or None,
                resource_subscriptions=list(
                    dict.fromkeys(chain.from_iterable(item.resource_subscriptions or () for item in honored))
                )
                or None,
            )
            group.start_soon(_recheck_authorization, check)
            try:
                if ctx.protocol_version != "2026-07-28":
                    await _deliver_legacy(ctx, bus, check, notifications, ready)
                return await handler(
                    replace(ctx, session=_AuthorizedSession(ctx.session, check)),
                    params.model_copy(update={"notifications": notifications}),
                )
            finally:
                group.cancel_scope.cancel()

        raise MCPError(code=-32001, message="Subscription ended before completion")

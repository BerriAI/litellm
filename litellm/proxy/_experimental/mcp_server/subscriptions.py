import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from itertools import chain
from typing import Final

import anyio
from anyio.abc import TaskStatus
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
    event_matches,
)
from mcp.types import (
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    RequestId,
    ServerNotification,
    SubscriptionFilter,
    SubscriptionsListenRequestParams,
    SubscriptionsListenResult,
)

from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES, MCP_TOOL_LISTING_TIMEOUT
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server import operations
from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.types.mcp_server.mcp_server_manager import MCPServer


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
            return SubscriptionTarget(server, client, await client.discovery_auth_fingerprint())

        return tuple([await prepare(server) for server in allowed])


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
            result = await gateway.execute(
                operation.model_copy(update={"params": PaginatedRequestParams(cursor=cursor)}), context
            )
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

    async def listen(
        self,
        ctx: ServerRequestContext,
        params: SubscriptionsListenRequestParams,
        authorize: Callable[[], Awaitable[OperationContext]],
        owner: str,
    ) -> SubscriptionsListenResult:
        if self._owners.count(owner) >= self._max_per_caller:
            raise MCPError(code=-32603, message="Caller subscription limit reached")
        if ctx.protocol_version != "2026-07-28":
            raise MCPError(code=-32601, message="Subscriptions require the modern MCP protocol")
        try:
            self._slots.acquire_nowait()
        except anyio.WouldBlock as error:
            raise MCPError(code=-32603, message="Subscription limit reached") from error
        self._owners += (owner,)
        try:
            return await self._listen(ctx, params, authorize)
        finally:
            position: Final = self._owners.index(owner)
            self._owners = self._owners[:position] + self._owners[position + 1 :]
            self._slots.release()

    async def _listen(
        self,
        ctx: ServerRequestContext,
        params: SubscriptionsListenRequestParams,
        authorize: Callable[[], Awaitable[OperationContext]],
    ) -> SubscriptionsListenResult:
        context: Final = await authorize()
        targets: Final = await self._resolve_targets(context)
        if any(target.server.protocol_version != "2026-07-28" for target in targets):
            raise MCPError(code=-32602, message="Select modern upstream servers for subscriptions")
        if not targets:
            raise HTTPException(status_code=403, detail="No authorized MCP subscription targets")
        if params.notifications.resource_subscriptions and len(targets) != 1:
            raise MCPError(code=-32602, message="Resource subscriptions require exactly one allowed server")
        bus: Final = _ListenerBus()
        handler: Final = ListenHandler(bus, max_subscriptions=1, max_buffered_events=128)

        async def check() -> OperationContext:
            fresh: Final = await authorize()
            current: Final = await self._resolve_targets(fresh)
            if tuple((item.server, item.credential) for item in current) != tuple(
                (item.server, item.credential) for item in targets
            ):
                raise MCPError(code=-32001, message="Subscription authorization or upstream configuration changed")
            return fresh

        async def recheck() -> None:
            while True:
                await asyncio.sleep(15)
                await check()

        async def relay(target: SubscriptionTarget, *, task_status: TaskStatus[SubscriptionFilter]) -> None:
            await operations._enforce_mcp_server_rate_limit(context.user_api_key_auth, target.server)  # pyright: ignore[reportPrivateUsage]  # same admission policy as other gateway operations

            async def connected(session: ClientSession) -> None:
                requested: Final = params.notifications
                async with listen(
                    session,
                    tools_list_changed=bool(requested.tools_list_changed),
                    prompts_list_changed=bool(requested.prompts_list_changed),
                    resources_list_changed=bool(requested.resources_list_changed),
                    resource_subscriptions=requested.resource_subscriptions or (),
                ) as subscription:
                    scoped: Final = replace(context, mcp_servers=(target.server.server_id,))
                    events: Final = tuple(
                        event
                        for event in (ToolsListChanged(), PromptsListChanged(), ResourcesListChanged())
                        if event_matches(subscription.honored, frozenset(), event)
                    )
                    previous: Final = {event: await _listing(event, scoped) for event in events}
                    task_status.started(subscription.honored)
                    await bus.ready.wait()
                    async for event in subscription:
                        fresh = await check()
                        manager = operations.global_mcp_server_manager
                        manager.invalidate_discovery_lists(target.server.server_id)
                        manager._drop_listed_tools(target.server.server_id)  # pyright: ignore[reportPrivateUsage]  # evict served catalogs without invalidating OAuth metadata
                        listing = await _listing(event, replace(fresh, mcp_servers=(target.server.server_id,)))
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
                await target.client.run_with_session(connected)
            finally:
                self._upstream_slots.release()

        async with anyio.create_task_group() as group:
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
            group.start_soon(recheck)
            try:
                return await handler(
                    replace(ctx, session=_AuthorizedSession(ctx.session, check)),
                    params.model_copy(update={"notifications": notifications}),
                )
            finally:
                group.cancel_scope.cancel()

        raise MCPError(code=-32001, message="Subscription ended before completion")

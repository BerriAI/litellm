from dataclasses import dataclass
from itertools import chain
from typing import Final

from mcp.shared.inbound import find_invalid_x_mcp_header

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant, grant_allows, holds_webhooks
from litellm.proxy.mcp_server.events.config import EventsSettings, server_allows_event, server_events
from litellm.proxy.mcp_server.events.errors import EventsFailure, event_not_found
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.proxy.mcp_server.utils import (
    add_server_prefix_to_name,
    get_server_prefix,
    iter_known_server_prefixes,
    match_known_server_prefix,
)
from litellm.types.mcp_server.events import DeliveryMode, EventType
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig


@dataclass(frozen=True, slots=True)
class EventServer:
    server: MCPServer
    config: MCPServerEventsConfig


@dataclass(frozen=True, slots=True)
class ResolvedEvent:
    target: EventServer
    upstream_name: str
    gateway_name: str


@dataclass(frozen=True, slots=True)
class PreparedEvent:
    resolved: ResolvedEvent
    upstream: SubscriptionTarget
    descriptor: EventType
    grant: EventGrant


def event_servers(allowed: tuple[MCPServer, ...]) -> tuple[EventServer, ...]:
    return tuple(EventServer(server, config) for server in allowed if (config := server_events(server)) is not None)


def gateway_name(server: MCPServer, upstream_name: str) -> str:
    return add_server_prefix_to_name(upstream_name, get_server_prefix(server))


def resolve(name: str, servers: tuple[EventServer, ...]) -> ResolvedEvent | None:
    matches: Final = tuple(
        (len(match[0]), ResolvedEvent(target=server, upstream_name=match[1], gateway_name=name))
        for server in servers
        if (match := match_known_server_prefix(name, iter_known_server_prefixes(server.server))) is not None
    )
    return max(matches, key=lambda item: item[0])[1] if matches else None


def offered_modes(
    descriptor: EventType, settings: EventsSettings, context: OperationContext
) -> tuple[DeliveryMode, ...]:
    webhook_ready: Final = settings.public_base_url is not None and holds_webhooks(context.user_api_key_auth)
    return tuple(mode for mode in descriptor.delivery if mode != "webhook" or webhook_ready)


def publishable(descriptor: EventType, config: MCPServerEventsConfig, grant: EventGrant) -> bool:
    return (
        server_allows_event(config, descriptor.name)
        and grant_allows(grant, descriptor.name)
        and find_invalid_x_mcp_header(dict(descriptor.input_schema)) is None
    )


def _published(
    descriptor: EventType, server: MCPServer, settings: EventsSettings, context: OperationContext
) -> EventType | None:
    modes: Final = offered_modes(descriptor, settings, context)
    if not modes:
        return None
    return descriptor.model_copy(update={"name": gateway_name(server, descriptor.name), "delivery": modes})


async def server_catalog(
    context: OperationContext,
    server: EventServer,
    allowed: tuple[MCPServer, ...],
    settings: EventsSettings,
    runtime: EventsRuntime,
) -> tuple[EventType, ...]:
    target: Final = await runtime.prepare_target(context, server.server, allowed)
    grant: Final = await runtime.grant(server.server.server_id, context.user_api_key_auth)
    listed: Final = tuple(
        _published(descriptor, server.server, settings, context)
        for descriptor in await runtime.descriptors(target)
        if publishable(descriptor, server.config, grant)
    )
    return tuple(item for item in listed if item is not None)


async def list_events(
    context: OperationContext, settings: EventsSettings, runtime: EventsRuntime
) -> tuple[EventType, ...]:
    allowed: Final = await runtime.allowed_servers(context)

    async def safe(server: EventServer) -> tuple[EventType, ...]:
        try:
            return await server_catalog(context, server, allowed, settings, runtime)
        except Exception as error:  # noqa: BLE001  # one failing upstream must not hide the other servers' events
            verbose_logger.warning("MCP events/list failed for server %s: %s", server.server.server_id, error)
            return ()

    pages: Final = tuple([await safe(server) for server in event_servers(allowed)])
    return tuple(sorted(chain.from_iterable(pages), key=lambda item: item.name))


async def prepare_event(
    context: OperationContext, name: str, runtime: EventsRuntime
) -> Result[PreparedEvent, EventsFailure]:
    allowed: Final = await runtime.allowed_servers(context)
    resolved: Final = resolve(name, event_servers(allowed))
    if resolved is None:
        return Error(event_not_found())
    grant: Final = await runtime.grant(resolved.target.server.server_id, context.user_api_key_auth)
    if not server_allows_event(resolved.target.config, resolved.upstream_name) or not grant_allows(
        grant, resolved.upstream_name
    ):
        return Error(event_not_found())
    target: Final = await runtime.prepare_target(context, resolved.target.server, allowed)
    descriptor: Final = next(
        (item for item in await runtime.descriptors(target) if item.name == resolved.upstream_name), None
    )
    if descriptor is None or not publishable(descriptor, resolved.target.config, grant):
        return Error(event_not_found())
    return Ok(PreparedEvent(resolved=resolved, upstream=target, descriptor=descriptor, grant=grant))

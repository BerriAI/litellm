import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from mcp.shared.inbound import find_invalid_x_mcp_header

from litellm._logging import verbose_logger
from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES, MCP_TOOL_LISTING_TIMEOUT
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant, event_grant, grant_allows, principal_of
from litellm.proxy.mcp_server.events.config import EventsSettings, server_allows_event, server_events
from litellm.proxy.mcp_server.events.errors import EventsFailure, event_not_found
from litellm.proxy.mcp_server.events.upstream import list_all
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget, prepare_target
from litellm.proxy.mcp_server.utils import (
    add_server_prefix_to_name,
    get_server_prefix,
    iter_known_server_prefixes,
    match_known_server_prefix,
)
from litellm.types.mcp_server.events import DeliveryMode, EventType
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

_DESCRIPTOR_TTL_SECONDS: Final = 60.0
_DESCRIPTOR_CACHE_SIZE: Final = 1024


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


class DescriptorCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], tuple[float, tuple[EventType, ...]]] = OrderedDict()

    async def get(
        self, target: SubscriptionTarget, load: Callable[[], Awaitable[tuple[EventType, ...]]]
    ) -> tuple[EventType, ...]:
        key: Final = (target.server.server_id, target.credential)
        cached: Final = self._entries.get(key)
        now: Final = self._clock()
        if cached is not None and cached[0] > now:
            return cached[1]
        loaded: Final = await load()
        self._entries[key] = (now + _DESCRIPTOR_TTL_SECONDS, loaded)
        self._entries.move_to_end(key)
        while len(self._entries) > _DESCRIPTOR_CACHE_SIZE:
            self._entries.popitem(last=False)
        return loaded


descriptor_cache: Final = DescriptorCache()


async def allowed_servers(context: OperationContext) -> tuple[MCPServer, ...]:
    from litellm.proxy.mcp_server import operations

    auth, _token, servers, _server_headers, _oauth_headers, _headers, ip = context.legacy_auth()
    return tuple(await operations._get_allowed_mcp_servers(auth, servers, ip))  # pyright: ignore[reportPrivateUsage]  # same server allowlist every gateway operation uses


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


def offered_modes(descriptor: EventType, settings: EventsSettings, context: OperationContext) -> tuple[DeliveryMode, ...]:
    webhook_ready: Final = settings.public_base_url is not None and principal_of(context.user_api_key_auth) is not None
    return tuple(mode for mode in descriptor.delivery if mode != "webhook" or webhook_ready)


async def descriptors(target: SubscriptionTarget) -> tuple[EventType, ...]:
    return await descriptor_cache.get(
        target, lambda: list_all(target.client, MCP_TOOL_LISTING_MAX_PAGES, MCP_TOOL_LISTING_TIMEOUT)
    )


def _publishable(descriptor: EventType, config: MCPServerEventsConfig, grant: EventGrant) -> bool:
    return (
        server_allows_event(config, descriptor.name)
        and grant_allows(grant, descriptor.name)
        and find_invalid_x_mcp_header(dict(descriptor.input_schema)) is None
    )


async def server_catalog(
    context: OperationContext, server: EventServer, allowed: tuple[MCPServer, ...], settings: EventsSettings
) -> tuple[EventType, ...]:
    target: Final = await prepare_target(context, server.server, allowed)
    grant: Final = await event_grant(server.server.server_id, context.user_api_key_auth)
    listed: Final = await descriptors(target)
    return tuple(
        descriptor.model_copy(
            update={
                "name": gateway_name(server.server, descriptor.name),
                "delivery": offered_modes(descriptor, settings, context),
            }
        )
        for descriptor in listed
        if _publishable(descriptor, server.config, grant) and offered_modes(descriptor, settings, context)
    )


async def list_events(context: OperationContext, settings: EventsSettings) -> tuple[EventType, ...]:
    allowed: Final = await allowed_servers(context)

    async def safe(server: EventServer) -> tuple[EventType, ...]:
        try:
            return await server_catalog(context, server, allowed, settings)
        except Exception as error:  # noqa: BLE001  # one failing upstream must not hide the other servers' events
            verbose_logger.warning("MCP events/list failed for server %s: %s", server.server.server_id, error)
            return ()

    servers: Final = event_servers(allowed)
    listed: Final = tuple([await safe(server) for server in servers])
    return tuple(sorted((item for page in listed for item in page), key=lambda item: item.name))


async def prepare_event(context: OperationContext, name: str) -> Result[PreparedEvent, EventsFailure]:
    allowed: Final = await allowed_servers(context)
    resolved: Final = resolve(name, event_servers(allowed))
    if resolved is None:
        return Error(event_not_found())
    grant: Final = await event_grant(resolved.target.server.server_id, context.user_api_key_auth)
    if not server_allows_event(resolved.target.config, resolved.upstream_name) or not grant_allows(
        grant, resolved.upstream_name
    ):
        return Error(event_not_found())
    target: Final = await prepare_target(context, resolved.target.server, allowed)
    descriptor: Final = next((item for item in await descriptors(target) if item.name == resolved.upstream_name), None)
    if descriptor is None or find_invalid_x_mcp_header(dict(descriptor.input_schema)) is not None:
        return Error(event_not_found())
    return Ok(PreparedEvent(resolved=resolved, upstream=target, descriptor=descriptor, grant=grant))

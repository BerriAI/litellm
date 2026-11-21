import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeAlias

from anyio.streams.memory import MemoryObjectReceiveStream

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant, event_grant
from litellm.proxy.mcp_server.events.guardrails import ScreenOutcome, screen_event
from litellm.proxy.mcp_server.events.limits import EventBudget, StreamSlots
from litellm.proxy.mcp_server.events.observability import EventActivity, record
from litellm.proxy.mcp_server.events.upstream import Overflow, StreamFrame
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import EventOccurrence, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer

Frames: TypeAlias = MemoryObjectReceiveStream[StreamFrame | Overflow]
UpstreamCall: TypeAlias = Callable[
    [SubscriptionTarget, str, JsonObject, Mapping[str, str]], Awaitable[Mapping[str, object]]
]
UpstreamStream: TypeAlias = Callable[
    [SubscriptionTarget, JsonObject, Mapping[str, str], Callable[[Frames], Awaitable[None]]], Awaitable[None]
]
Screen: TypeAlias = Callable[[EventOccurrence, str, str, str | None, UserAPIKeyAuth | None], Awaitable[ScreenOutcome]]


async def _allowed_servers(context: OperationContext) -> tuple[MCPServer, ...]:
    from litellm.proxy.mcp_server import operations

    auth, _token, servers, _server_headers, _oauth_headers, _headers, ip = context.legacy_auth()
    return tuple(await operations._get_allowed_mcp_servers(auth, servers, ip))  # pyright: ignore[reportPrivateUsage]  # same server allowlist every gateway operation uses


async def _prepare_target(
    context: OperationContext, server: MCPServer, allowed: tuple[MCPServer, ...]
) -> SubscriptionTarget:
    from litellm.proxy.mcp_server.subscriptions import prepare_target

    return await prepare_target(context, server, allowed)


async def _rate_limited(server: MCPServer, caller: UserAPIKeyAuth | None) -> bool:
    from litellm.proxy.mcp_server import operations

    return await operations._mcp_server_rate_limit_rejection(server, caller) is not None  # pyright: ignore[reportPrivateUsage]  # same per-server admission as tools/call


async def _call(
    target: SubscriptionTarget, method: str, params: JsonObject, headers: Mapping[str, str]
) -> Mapping[str, object]:
    from litellm.proxy.mcp_server.events.upstream import call

    return await call(target.client, method, params, headers, target.client.timeout)


async def _stream(
    target: SubscriptionTarget,
    params: JsonObject,
    headers: Mapping[str, str],
    consume: Callable[[Frames], Awaitable[None]],
) -> None:
    from litellm.proxy.mcp_server.events.upstream import stream

    await stream(target.client, params, headers, consume)


async def _list_upstream(target: SubscriptionTarget) -> tuple[EventType, ...]:
    from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES
    from litellm.proxy.mcp_server.events.upstream import list_all

    return await list_all(target.client, MCP_TOOL_LISTING_MAX_PAGES, target.client.timeout)


async def _screen(
    occurrence: EventOccurrence,
    server_name: str,
    gateway_name: str,
    subscription_id: str | None,
    caller: UserAPIKeyAuth | None,
) -> ScreenOutcome:
    return await screen_event(
        occurrence, server_name=server_name, gateway_name=gateway_name, subscription_id=subscription_id, caller=caller
    )


_DESCRIPTOR_TTL_SECONDS: Final = 60.0
_DESCRIPTOR_CACHE_SIZE: Final = 1024
_CacheKey: TypeAlias = tuple[str, str]
_CacheEntry: TypeAlias = tuple[float, tuple[EventType, ...]]


class DescriptorCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: OrderedDict[_CacheKey, _CacheEntry] = OrderedDict()  # mutable-ok: bounded TTL cache

    async def get(
        self, target: SubscriptionTarget, load: Callable[[SubscriptionTarget], Awaitable[tuple[EventType, ...]]]
    ) -> tuple[EventType, ...]:
        key: Final = (target.server.server_id, target.credential)
        cached: Final = self._entries.get(key)
        now: Final = self._clock()
        if cached is not None and cached[0] > now:
            return cached[1]
        loaded: Final = await load(target)
        self._entries[key] = (now + _DESCRIPTOR_TTL_SECONDS, loaded)
        self._entries.move_to_end(key)
        while len(self._entries) > _DESCRIPTOR_CACHE_SIZE:
            self._entries.popitem(last=False)
        return loaded


@dataclass(frozen=True, slots=True)
class EventsRuntime:
    clock: Callable[[], float] = time.time
    allowed_servers: Callable[[OperationContext], Awaitable[tuple[MCPServer, ...]]] = _allowed_servers
    grant: Callable[[str, UserAPIKeyAuth | None], Awaitable[EventGrant]] = event_grant
    prepare_target: Callable[[OperationContext, MCPServer, tuple[MCPServer, ...]], Awaitable[SubscriptionTarget]] = (
        _prepare_target
    )
    list_upstream: Callable[[SubscriptionTarget], Awaitable[tuple[EventType, ...]]] = _list_upstream
    rate_limited: Callable[[MCPServer, UserAPIKeyAuth | None], Awaitable[bool]] = _rate_limited
    call: UpstreamCall = _call
    stream: UpstreamStream = _stream
    screen: Screen = _screen
    record: Callable[[EventActivity], Awaitable[None]] = record
    budget: EventBudget = field(default_factory=EventBudget)
    slots: StreamSlots = field(default_factory=StreamSlots)
    descriptor_cache: DescriptorCache = field(default_factory=DescriptorCache)

    async def descriptors(self, target: SubscriptionTarget) -> tuple[EventType, ...]:
        return await self.descriptor_cache.get(target, self.list_upstream)

    def now(self) -> int:
        return int(self.clock())


DEFAULT_RUNTIME: Final = EventsRuntime()

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from litellm.mcp_client.client import MCPClient
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant
from litellm.proxy.mcp_server.events.guardrails import Blocked, Screened, ScreenOutcome
from litellm.proxy.mcp_server.events.observability import EventActivity
from litellm.proxy.mcp_server.events.runtime import EventsRuntime, Frames
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import EventOccurrence, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer

from .fixtures import (
    BUILD_FAILED,
    CALLER,
    JWT,
    SETTINGS,
    admitted_subject,
    context,
    occurrence,
    server,
)

__all__ = (
    "BUILD_FAILED",
    "CALLER",
    "JWT",
    "SETTINGS",
    "FakeUpstream",
    "admitted_subject",
    "context",
    "occurrence",
    "runtime",
    "screen_blocking",
    "screen_passthrough",
    "server",
)


@dataclass
class FakeUpstream:
    responses: Mapping[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    calls: list[tuple[str, JsonObject, Mapping[str, str]]] = field(default_factory=list)  # mutable-ok: records calls
    frames: Sequence[object] = ()
    stream_calls: list[JsonObject] = field(default_factory=list)  # mutable-ok: records streams

    async def call(
        self, target: SubscriptionTarget, method: str, params: JsonObject, headers: Mapping[str, str]
    ) -> Mapping[str, object]:
        self.calls.append((method, params, headers))
        response: Final = self.responses[method]
        if isinstance(response, Exception):
            raise response
        return response

    async def stream(
        self,
        target: SubscriptionTarget,
        params: JsonObject,
        headers: Mapping[str, str],
        consume: Callable[[Frames], Awaitable[None]],
    ) -> None:
        import anyio

        self.stream_calls.append(params)
        send, receive = anyio.create_memory_object_stream[object](len(self.frames) + 1)
        for frame in self.frames:
            send.send_nowait(frame)
        send.close()
        await consume(receive)  # pyright: ignore[reportArgumentType]  # frames are StreamFrame or Overflow in every test


def screen_passthrough() -> Callable[..., Awaitable[ScreenOutcome]]:
    async def screen(item: EventOccurrence, *_: object) -> ScreenOutcome:
        return Screened(item.data)

    return screen


def screen_blocking() -> Callable[..., Awaitable[ScreenOutcome]]:
    async def screen(item: EventOccurrence, *_: object) -> ScreenOutcome:
        return Blocked("pii")

    return screen


def runtime(
    servers: tuple[MCPServer, ...],
    descriptors: tuple[EventType, ...] = (BUILD_FAILED,),
    upstream: FakeUpstream | None = None,
    grant: EventGrant = None,
    rate_limited: bool = False,
    screen: Callable[..., Awaitable[ScreenOutcome]] | None = None,
    records: list[EventActivity] | None = None,
    clock: Callable[[], float] = lambda: 1_800_000_000.0,
) -> EventsRuntime:
    calls: Final = upstream or FakeUpstream()
    recorded: Final = records if records is not None else []

    async def allowed(_: OperationContext) -> tuple[MCPServer, ...]:
        return servers

    async def grant_for(_server_id: str, _caller: UserAPIKeyAuth | None) -> EventGrant:
        return grant

    async def prepare(_: OperationContext, item: MCPServer, __: tuple[MCPServer, ...]) -> SubscriptionTarget:
        return SubscriptionTarget(item, MCPClient(server_url="https://upstream.example/mcp"), "credential")

    async def listing(_: SubscriptionTarget) -> tuple[EventType, ...]:
        return descriptors

    async def limited(_server: MCPServer, _caller: UserAPIKeyAuth | None) -> bool:
        return rate_limited

    async def record(activity: EventActivity) -> None:
        recorded.append(activity)

    return EventsRuntime(
        clock=clock,
        allowed_servers=allowed,
        grant=grant_for,
        prepare_target=prepare,
        list_upstream=listing,
        rate_limited=limited,
        call=calls.call,
        stream=calls.stream,
        screen=screen or screen_passthrough(),
        record=record,
    )

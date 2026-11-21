from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from litellm.mcp_client.client import MCPClient
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.guardrails import Blocked, ScreenOutcome, Screened
from litellm.proxy.mcp_server.events.observability import EventActivity
from litellm.proxy.mcp_server.events.runtime import EventsRuntime, Frames
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import EventOccurrence, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

SETTINGS: Final = EventsSettings(enabled=True, public_base_url="https://gateway.example")
CALLER: Final = UserAPIKeyAuth(api_key="sk-caller-one", user_id="user-1")
BUILD_FAILED: Final = EventType(
    name="build.failed",
    delivery=("poll", "push", "webhook"),
    input_schema={"type": "object", "properties": {"repo": {"type": "string", "x-mcp-header": "Repo"}}},
)


def server(
    server_id: str = "srv-ci",
    alias: str = "ci",
    events: MCPServerEventsConfig | None = None,
    protocol_version: str = "2026-07-28",
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=alias,
        alias=alias,
        transport="http",
        url="https://upstream.example/mcp",
        protocol_version=protocol_version,
        events=events if events is not None else MCPServerEventsConfig(enabled=True),
    )


JWT: Final = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1In0.c2lnbmF0dXJlLWE"


def admitted_subject(user_id: str) -> UserAPIKeyAuth:
    subject: Final = UserAPIKeyAuth(user_id=user_id)
    subject.mcp_admitted_user_subject = True
    return subject


def context(caller: UserAPIKeyAuth | None = CALLER) -> OperationContext:
    return OperationContext(_caller=caller, protocol_version="2026-07-28")


def occurrence(event_id: str = "evt_1", cursor: str | None = "7", data: JsonObject | None = None) -> EventOccurrence:
    return EventOccurrence(
        event_id=event_id,
        name="build.failed",
        timestamp="2026-11-20T00:00:00Z",
        data=data if data is not None else {"repo": "acme/web"},
        cursor=cursor,
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

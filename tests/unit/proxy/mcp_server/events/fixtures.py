from typing import Final

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.types.mcp_server.events import EventOccurrence, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

SETTINGS: Final = EventsSettings(enabled=True, public_base_url="https://gateway.example")


def virtual_key(api_key: str, **fields: str | None) -> UserAPIKeyAuth:  # noqa: ANN003  # test builder mirrors the auth path's marker
    """A caller authenticated by a virtual key, carrying the server-only marker the key auth path sets."""
    return UserAPIKeyAuth(api_key=api_key, **fields).model_copy(update={"via_virtual_key": True})


CALLER: Final = virtual_key("sk-caller-one", user_id="user-1")
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
    return UserAPIKeyAuth(user_id=user_id).model_copy(update={"mcp_admitted_user_subject": True})


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

from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.mcp_server.events import EVENTS_PROTOCOL_VERSION
from litellm.types.mcp_server.mcp_server_manager import MCPEventCallbackBase, MCPServer, MCPServerEventsConfig

_SETTING_KEYS: Final = MappingProxyType(
    {
        "enabled": "mcp_events_enabled",
        "public_base_url": "mcp_events_public_base_url",
        "max_subscriptions_per_key": "mcp_event_max_subscriptions_per_key",
        "max_streams_per_caller": "mcp_event_max_streams_per_caller",
        "max_streams": "mcp_event_max_streams",
        "max_events_per_minute": "mcp_event_max_events_per_minute",
    }
)


class EventsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    public_base_url: MCPEventCallbackBase | None = None
    max_subscriptions_per_key: int = Field(default=50, ge=1)
    max_streams_per_caller: int = Field(default=50, ge=1)
    max_streams: int = Field(default=512, ge=1)
    max_events_per_minute: int = Field(default=600, ge=1)


def _general_settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings_view

    return general_settings_view()


def events_settings(view: Callable[[], Mapping[str, object]] = _general_settings) -> EventsSettings:
    settings: Final = view()
    return EventsSettings.model_validate(
        {field: settings[key] for field, key in _SETTING_KEYS.items() if settings.get(key) is not None}
    )


def server_events(server: MCPServer) -> MCPServerEventsConfig | None:
    config: Final = server.events
    if config is None or not config.enabled or server.protocol_version != EVENTS_PROTOCOL_VERSION:
        return None
    return config


def server_allows_event(config: MCPServerEventsConfig, upstream_name: str) -> bool:
    patterns: Final = config.allowed_events
    return patterns is None or any(fnmatchcase(upstream_name, pattern) for pattern in patterns)

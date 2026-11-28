from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings, server_allows_event, server_events
from litellm.types.mcp_server.mcp_server_manager import MCPServerEventsConfig

from .fixtures import server


def test_settings_default_to_off_with_documented_limits() -> None:
    settings: Final = events_settings(lambda: {})
    assert settings == EventsSettings()
    assert settings.enabled is False


@pytest.mark.parametrize(
    "base_url",
    (
        "",
        "gateway.example",
        "http://gateway.example",
        "https://",
        "https://gw.example?q=1",
        "https://gw.example/#section",
    ),
)
def test_callback_base_requires_an_absolute_https_url_without_query_or_fragment(base_url: str) -> None:
    with pytest.raises(ValidationError):
        events_settings(lambda: {"mcp_events_public_base_url": base_url})


def test_proxy_base_preserves_a_deployment_path_and_port() -> None:
    base: Final = "https://gw.example:1443/proxy"
    assert events_settings(lambda: {"mcp_events_public_base_url": base}).public_base_url == base


def test_settings_read_the_general_settings_keys() -> None:
    view: Final[Mapping[str, object]] = {
        "mcp_events_enabled": True,
        "mcp_events_public_base_url": "https://gw.example",
        "mcp_event_max_subscriptions_per_key": 3,
        "mcp_event_max_streams_per_caller": 4,
        "mcp_event_max_streams": 5,
        "mcp_event_max_events_per_minute": 6,
    }
    assert events_settings(lambda: view) == EventsSettings(
        enabled=True,
        public_base_url="https://gw.example",
        max_subscriptions_per_key=3,
        max_streams_per_caller=4,
        max_streams=5,
        max_events_per_minute=6,
    )


@pytest.mark.parametrize(
    ("events", "protocol", "served"),
    (
        (MCPServerEventsConfig(enabled=True), "2026-07-28", True),
        (MCPServerEventsConfig(enabled=False), "2026-07-28", False),
        (MCPServerEventsConfig(enabled=True), "2025-11-25", False),
        (MCPServerEventsConfig(enabled=True), "auto", False),
    ),
)
def test_a_server_serves_events_only_when_enabled_on_the_modern_protocol(
    events: MCPServerEventsConfig, protocol: str, served: bool
) -> None:
    assert (server_events(server(events=events, protocol_version=protocol)) is not None) is served


@pytest.mark.parametrize(
    ("patterns", "name", "allowed"),
    (
        (None, "anything", True),
        (("build.*",), "build.failed", True),
        (("build.*",), "deploy.done", False),
        (("deploy.done", "build.failed"), "build.failed", True),
    ),
)
def test_allowed_events_globs(patterns: tuple[str, ...] | None, name: str, allowed: bool) -> None:
    assert server_allows_event(MCPServerEventsConfig(enabled=True, allowed_events=patterns), name) is allowed


def test_an_empty_allowlist_is_rejected_so_it_cannot_be_read_as_allow_all_later() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MCPServerEventsConfig(enabled=True, allowed_events=())

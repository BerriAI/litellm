"""Regression tests for UI-registered embed plugins.

Covers three bugs:
1. `general_settings.plugins` was not a field on ConfigGeneralSettings, so the
   admin UI's POST /config/field/update with field_name="plugins" was rejected
   with "Invalid field=plugins passed in."
2. The in-memory plugin registry only refreshed at startup, so a plugin added
   via the UI did not appear in /api/plugins until a restart.
3. Plugins persisted to DB general_settings were not loaded on startup (the
   registry only initialised from the YAML config), so UI-added plugins vanished
   after a restart.
"""

import asyncio
from unittest.mock import MagicMock

from litellm.proxy._types import (
    ConfigGeneralSettings,
    LitellmUserRoles,
    PluginConfig,
    UserAPIKeyAuth,
)
from litellm.proxy.plugin_routes import list_plugins, register_plugins_from_config


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER)


def test_plugins_is_a_valid_general_setting() -> None:
    """The config-update endpoint gates on this exact membership check."""
    assert "plugins" in ConfigGeneralSettings.model_fields


def test_config_general_settings_parses_plugin_list() -> None:
    """A list of plugin dicts (what the UI sends) coerces into PluginConfig."""
    settings = ConfigGeneralSettings.model_validate(
        {
            "plugins": [
                {
                    "name": "chat-ui",
                    "display_name": "Chat UI",
                    "url": "http://localhost:3300",
                },
                {
                    "name": "agent-builder",
                    "url": "http://127.0.0.1:4010",
                    "plugin_key": "sk-secret",
                },
            ]
        }
    )
    plugins = settings.plugins
    assert plugins is not None
    assert [p.name for p in plugins] == ["chat-ui", "agent-builder"]
    assert isinstance(plugins[0], PluginConfig)
    assert plugins[1].display_name is None
    assert plugins[1].plugin_key == "sk-secret"


def test_registered_plugins_appear_in_list_without_restart() -> None:
    """register_plugins_from_config makes UI-added plugins visible immediately,
    and replaces (not merges) so removed plugins disappear."""
    register_plugins_from_config(
        {
            "plugins": [
                {
                    "name": "chat-ui",
                    "display_name": "Chat UI",
                    "url": "http://localhost:3300",
                }
            ]
        }
    )
    names = [p["name"] for p in asyncio.run(list_plugins(user_api_key_dict=_admin()))]
    assert names == ["chat-ui"]

    register_plugins_from_config(
        {
            "plugins": [
                {
                    "name": "chat-ui",
                    "display_name": "Chat UI",
                    "url": "http://localhost:3300",
                },
                {
                    "name": "agent-builder",
                    "display_name": "Agent Builder",
                    "url": "http://127.0.0.1:4010",
                },
            ]
        }
    )
    names = sorted(
        p["name"] for p in asyncio.run(list_plugins(user_api_key_dict=_admin()))
    )
    assert names == ["agent-builder", "chat-ui"]

    # Removing a plugin from config drops it from the live list.
    register_plugins_from_config({})
    assert asyncio.run(list_plugins(user_api_key_dict=_admin())) == []


def test_plugin_key_is_never_returned_to_the_browser() -> None:
    """plugin_key is a credential the UI never needs; /api/plugins must omit it
    for every caller, admin included, so it never lands in browser state."""
    register_plugins_from_config(
        {
            "plugins": [
                {
                    "name": "p",
                    "display_name": "P",
                    "url": "http://localhost:9",
                    "plugin_key": "sk-secret",
                }
            ]
        }
    )

    admin_entry = asyncio.run(list_plugins(user_api_key_dict=_admin()))[0]
    user_entry = asyncio.run(list_plugins(user_api_key_dict=_non_admin()))[0]

    assert "plugin_key" not in admin_entry
    assert "plugin_key" not in user_entry
    assert admin_entry["url"] == "http://localhost:9"

    register_plugins_from_config({})


def test_db_persisted_plugins_load_on_startup() -> None:
    """Plugins saved to DB general_settings must register when the DB config is
    merged at startup, not just when present in the YAML file."""
    from litellm.proxy.proxy_server import ProxyConfig

    register_plugins_from_config({})  # start empty (as if YAML had no plugins)

    ProxyConfig()._add_general_settings_from_db_config(
        config_data={
            "general_settings": {
                "plugins": [
                    {
                        "name": "db-plugin",
                        "display_name": "DB Plugin",
                        "url": "http://localhost:5000",
                    }
                ]
            }
        },
        general_settings={},
        proxy_logging_obj=MagicMock(),
    )

    names = [p["name"] for p in asyncio.run(list_plugins(user_api_key_dict=_admin()))]
    assert names == ["db-plugin"]

    register_plugins_from_config({})


def test_safe_response_headers_sandbox_and_strips_wire_headers() -> None:
    """Proxied plugin responses must be inert and shed wire/cookie headers."""
    from litellm.proxy.plugin_routes import _safe_response_headers

    out = _safe_response_headers(
        {
            "content-type": "text/html",
            "content-encoding": "gzip",
            "content-length": "123",
            "set-cookie": "session=abc",
            "content-security-policy": "default-src *",
        }
    )

    assert out["content-security-policy"] == "sandbox"
    assert out["x-content-type-options"] == "nosniff"
    assert out["content-type"] == "text/html"
    for stripped in ("content-encoding", "content-length", "set-cookie"):
        assert stripped not in out


def test_litellm_credential_header_names_covers_every_auth_header() -> None:
    """The canonical strip set must list every header user_api_key_auth accepts
    as a litellm key, so a new auth header can't silently start leaking."""
    from litellm.proxy._types import SpecialHeaders

    assert SpecialHeaders.litellm_credential_header_names() == {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "ocp-apim-subscription-key",
        "x-litellm-api-key",
    }


def test_every_litellm_auth_header_is_stripped_before_forwarding() -> None:
    """A plugin must never receive any header that authenticates against litellm,
    only the hop-by-hop set and benign headers are forwarded."""
    from litellm.proxy.plugin_routes import _request_strip_headers

    strip = _request_strip_headers()
    incoming = {
        "Authorization": "Bearer sk-litellm",
        "API-Key": "sk-litellm",
        "X-Api-Key": "sk-litellm",
        "X-Goog-Api-Key": "sk-litellm",
        "Ocp-Apim-Subscription-Key": "sk-litellm",
        "X-Litellm-Api-Key": "sk-litellm",
        "Cookie": "litellm_session=abc",
        "Accept": "application/json",
        "X-Trace-Id": "t-1",
    }
    forwarded = {k: v for k, v in incoming.items() if k.lower() not in strip}

    assert forwarded == {"Accept": "application/json", "X-Trace-Id": "t-1"}


def test_configured_custom_key_header_is_stripped() -> None:
    """A custom general_settings.litellm_key_header_name must also be stripped,
    read live so config changes are honoured without a restart."""
    from litellm.proxy import proxy_server
    from litellm.proxy.plugin_routes import _request_strip_headers

    original = getattr(proxy_server, "general_settings", None)
    proxy_server.general_settings = {"litellm_key_header_name": "X-My-Tenant-Key"}
    try:
        assert "x-my-tenant-key" in _request_strip_headers()
    finally:
        proxy_server.general_settings = original

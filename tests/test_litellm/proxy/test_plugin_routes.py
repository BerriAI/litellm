"""Regression tests for UI-registered embed plugins.

Covers two bugs:
1. `general_settings.plugins` was not a field on ConfigGeneralSettings, so the
   admin UI's POST /config/field/update with field_name="plugins" was rejected
   with "Invalid field=plugins passed in."
2. The in-memory plugin registry only refreshed at startup, so a plugin added
   via the UI did not appear in /api/plugins until a restart.
"""

import asyncio

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


def test_plugin_key_is_exposed_only_to_admins() -> None:
    """plugin_key is a credential; non-admin callers must never receive it."""
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

    assert admin_entry.get("plugin_key") == "sk-secret"
    assert "plugin_key" not in user_entry

    register_plugins_from_config({})

"""Regenerate generated/db_settings.json: how the Python proxy layers LiteLLM_Config rows over a config.

    uv run python litellm/rust/crates/config/scripts/generate_db_settings_fixtures.py \
        > litellm/rust/crates/config/generated/db_settings.json

Run it from the repository root. Each case drives the real ProxyConfig._update_config_from_db with
in-memory rows in place of the database, then _config_with_resolved_settings as get_config does, and
records the merged config and the keys each SettingsStore reports as shadowed. A case's `ui_settings` row is
the LiteLLM_UISettings column as Prisma hands it back, which the real sync_ui_settings_to_general_settings reads
through a stand-in client after the LiteLLM_Config rows, as the proxy does at startup.

Two inputs are narrowed before Python sees them, so the fixture holds what the proxy acts on rather
than what get_config returns. The litellm_settings row keeps only the keys a stored row changes
(STORED_LITELLM_SETTINGS below): get_config passes every stored key through, but the proxy only sets
LITELLM_SETTINGS_SAFE_DB_OVERRIDES on litellm, registers the callback lists and reads `responses`.
Stored values are plain text, which decrypt_value_helper passes through, since the Rust overlay takes
rows its caller already decrypted.
"""

import asyncio
import copy
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("LITELLM_SALT_KEY", "sk-fixture-salt")

from litellm.constants import LITELLM_SETTINGS_SAFE_DB_OVERRIDES  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402
from litellm.proxy.config_resolvers.settings_rules import _UI_SETTINGS_FIELDS  # noqa: E402
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import sync_ui_settings_to_general_settings  # noqa: E402

STORED_LITELLM_SETTINGS = sorted(
    {*LITELLM_SETTINGS_SAFE_DB_OVERRIDES, *proxy_server._DB_CONFIG_CALLBACK_EVENT_TYPES, "responses"}
)

CASES = [
    {
        "name": "general_settings_file_wins_and_db_fills",
        "file": {
            "general_settings": {"max_parallel_requests": 10, "completion_model": None, "store_model_in_db": True},
        },
        "rows": {
            "general_settings": {
                "max_parallel_requests": 20,
                "completion_model": "db-model",
                "store_model_in_db": True,
                "global_max_parallel_requests": 100,
                "health_check_interval": None,
                "not_a_general_setting": 1,
                "disable_custom_api_keys": True,
            },
        },
    },
    {
        "name": "router_and_litellm_settings",
        "file": {
            "router_settings": {"routing_strategy": "simple-shuffle"},
            "litellm_settings": {"turn_off_message_logging": False, "callbacks": ["file_callback"]},
        },
        "rows": {
            "router_settings": {"routing_strategy": "latency-based-routing", "num_retries": 3},
            "litellm_settings": {
                "turn_off_message_logging": True,
                "callbacks": ["db_callback"],
                "success_callback": ["db_success"],
                "default_team_params": {"max_budget": 10},
                "drop_params": True,
                "cache": True,
            },
        },
    },
    {
        "name": "pass_through_endpoints_union",
        "file": {
            "general_settings": {
                "pass_through_endpoints": [
                    {"path": "/a", "target": "https://file.example/a"},
                    {"path": "/b", "target": "https://file.example/b"},
                    {"target": "https://file.example/no-path"},
                ]
            }
        },
        "rows": {
            "general_settings": {
                "pass_through_endpoints": [
                    {"path": "/b", "target": "https://db.example/b"},
                    {"path": "/c", "target": "https://db.example/c"},
                ]
            }
        },
    },
    {
        "name": "db_only_sections",
        "file": {"model_list": [{"model_name": "m", "litellm_params": {"model": "openai/m"}}], "router_settings": None},
        "rows": {
            "general_settings": {
                "pass_through_endpoints": [{"path": "/x", "target": "https://db.example/x"}],
                "max_parallel_requests": 5,
            },
            "router_settings": {"num_retries": 2},
            "environment_variables": {"DB_ONLY": "db"},
        },
    },
    {
        "name": "environment_variables",
        "file": {
            "environment_variables": {"FILE_ONLY": "file", "SHARED": "file", "DECLARED_NULL": None},
        },
        "rows": {
            "environment_variables": {
                "SHARED": "db",
                "DECLARED_NULL": "db",
                "lower_name": "lower",
                "REFERENCE": "os.environ/SOMEWHERE_ELSE",
                "LITELLM_ENABLE_MCP_STDIO": "true",
                "litellm_enable_mcp_stdio": "true",
            },
        },
    },
    {
        "name": "ui_settings_object",
        "file": {"general_settings": {"forward_llm_provider_auth_headers": False, "master_key": "sk-file"}},
        "rows": {
            "general_settings": {
                "forward_client_headers_to_llm_api": False,
                "disable_custom_api_keys": True,
                "max_parallel_requests": 5,
            },
            "ui_settings": {
                "forward_client_headers_to_llm_api": True,
                "forward_llm_provider_auth_headers": True,
                "disable_model_add_for_internal_users": None,
                "team_admin_editable_team_fields": ["team_alias"],
                "enable_chat_ui": True,
                "usage_default_time_range": "7d",
            },
        },
    },
    {
        "name": "ui_settings_json_text",
        "file": {"model_list": []},
        "rows": {"ui_settings": json.dumps({"forward_client_headers_to_llm_api": True, "disable_aria": True})},
    },
    {
        "name": "ui_settings_unreadable",
        "file": {"general_settings": {"master_key": "sk-file"}},
        "rows": {"ui_settings": "not json"},
    },
    {
        "name": "ui_settings_not_an_object",
        "file": {"general_settings": {"master_key": "sk-file"}},
        "rows": {"ui_settings": ["forward_client_headers_to_llm_api"]},
    },
    {
        "name": "no_rows",
        "file": {"general_settings": {"master_key": "sk-file"}, "litellm_settings": None},
        "rows": {},
    },
]


def narrowed(rows):
    litellm_settings = rows.get("litellm_settings")
    if not isinstance(litellm_settings, dict):
        return rows
    return {
        **rows,
        "litellm_settings": {key: value for key, value in litellm_settings.items() if key in STORED_LITELLM_SETTINGS},
    }


async def layered(case):
    rows = narrowed(case["rows"])

    async def get_config_param(_client, section):
        if section not in rows:
            return None
        return SimpleNamespace(param_name=section, param_value=copy.deepcopy(rows[section]), reload_revision=None)

    async def find_unique(where):
        if "ui_settings" not in rows or where != {"id": "ui_settings"}:
            return None
        return SimpleNamespace(id="ui_settings", ui_settings=copy.deepcopy(rows["ui_settings"]))

    ui_settings_client = SimpleNamespace(
        db=SimpleNamespace(litellm_uisettings=SimpleNamespace(find_unique=find_unique))
    )

    proxy_server.get_config_param = get_config_param
    config = proxy_server.ProxyConfig()
    proxy_server._bind_general_settings_store(config.settings)
    file = copy.deepcopy(case["file"])
    config._load_yaml_settings_stores(file)
    environment = dict(os.environ)
    try:
        merged = await config._update_config_from_db(prisma_client=object(), config=file, store_model_in_db=True)
        await sync_ui_settings_to_general_settings(ui_settings_client)
        merged = config._config_with_resolved_settings(merged)
    finally:
        os.environ.clear()
        os.environ.update(environment)
    shadowed = sorted(
        [section, key] for section, store in config._settings_stores.items() for key in store.shadowed_db_keys()
    )
    return {**case, "merged": merged, "shadowed": shadowed}


async def main():
    fixture = {
        "tables": {
            "managed_general_settings": sorted(proxy_server._db_managed_general_settings_keys()),
            "ui_settings": sorted(_UI_SETTINGS_FIELDS),
            "stored_litellm_settings": STORED_LITELLM_SETTINGS,
        },
        "cases": [await layered(case) for case in CASES],
    }
    json.dump(fixture, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


asyncio.run(main())

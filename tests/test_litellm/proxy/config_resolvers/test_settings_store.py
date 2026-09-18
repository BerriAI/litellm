from __future__ import annotations

from typing import Final

import pytest

from litellm.proxy.config_resolvers.settings_rules import JsonValue
from litellm.proxy.config_resolvers.settings_store import SettingsStore


def test_settings_store_matches_plain_dict_mapping_operations() -> None:
    store: Final = SettingsStore("general_settings")

    store["none"] = None
    store["false"] = False
    store["zero"] = 0
    store["empty_list"] = []
    store["empty_string"] = ""
    store.update({"updated": "value"})
    defaulted: Final = store.setdefault("defaulted", "default")
    existing: Final = store.setdefault("updated", "other")
    popped: Final = store.pop("updated")

    assert defaulted == "default"
    assert existing == "value"
    assert popped == "value"
    assert store.get("missing") is None
    assert store["none"] is None
    assert "false" in store
    assert tuple(store) == ("none", "false", "zero", "empty_list", "empty_string", "defaulted")
    assert len(store) == 6
    assert dict(store) == {
        "none": None,
        "false": False,
        "zero": 0,
        "empty_list": [],
        "empty_string": "",
        "defaulted": "default",
    }


@pytest.mark.parametrize("operation", ("set", "update", "setdefault", "pop", "delete"))
@pytest.mark.parametrize("initial_value", (None, False, 0, [], ""))
def test_settings_store_mapping_operations_match_a_plain_dict(operation: str, initial_value: JsonValue) -> None:
    expected: dict[str, JsonValue] = {"value": initial_value}
    store: Final = SettingsStore("general_settings")
    store["value"] = initial_value

    match operation:
        case "set":
            expected["value"] = "replacement"
            store["value"] = "replacement"
        case "update":
            expected.update({"value": "replacement", "other": initial_value})
            store.update({"value": "replacement", "other": initial_value})
        case "setdefault":
            assert store.setdefault("value", "replacement") == expected.setdefault("value", "replacement")
            assert store.setdefault("other", initial_value) == expected.setdefault("other", initial_value)
        case "pop":
            assert store.pop("value") == expected.pop("value")
        case "delete":
            del expected["value"]
            del store["value"]
        case _:
            raise AssertionError(f"unexpected operation: {operation}")

    assert dict(store) == expected
    assert tuple(store) == tuple(expected)
    assert len(store) == len(expected)
    assert ("value" in store) is ("value" in expected)


def test_settings_store_keeps_unaffected_runtime_values_on_a_db_row_refresh() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"template": "os.environ/SETTING"})
    store.apply_runtime_values({"template": "resolved", "changed": "resolved-runtime"})

    store.apply_db_row("general_settings", {"changed": "database"})

    assert store["template"] == "resolved"
    assert store["changed"] == "database"
    assert store.source("changed") == "db"


def test_settings_store_keeps_a_config_owned_key_when_a_db_row_disagrees() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"changed": "config"})
    store.apply_runtime_values({"changed": "resolved-config"})

    store.apply_db_row("general_settings", {"changed": "database"})

    assert store["changed"] == "config"
    assert store.source("changed") == "config"


def test_settings_store_removes_only_runtime_values_affected_by_a_cleared_db_row() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"template": "os.environ/SETTING"})
    store.apply_db_row("ui_settings", {"allow_public_health_readiness_details": True})
    store.apply_runtime_values({"template": "resolved", "allow_public_health_readiness_details": True})

    store.apply_db_row("ui_settings", {})

    assert store["template"] == "resolved"
    assert "allow_public_health_readiness_details" not in store


def test_settings_store_preserves_falsy_config_values_and_provenance() -> None:
    store: Final = SettingsStore("general_settings")
    yaml_values: Final = {"none": None, "false": False, "zero": 0, "empty_list": [], "empty_string": ""}

    store.load_yaml(yaml_values)

    assert dict(store) == yaml_values
    assert tuple(store.source(key) for key in yaml_values) == ("config",) * len(yaml_values)


@pytest.mark.parametrize(
    ("yaml_value", "db_value", "expected_value", "expected_source"),
    (
        ("from-config", "from-db", "from-config", "config"),
        ("from-config", None, "from-config", "config"),
        (None, "from-db", None, "config"),
        (None, None, None, "config"),
    ),
)
def test_settings_store_resolves_a_db_row_with_provenance(
    yaml_value: object,
    db_value: object,
    expected_value: object,
    expected_source: str,
) -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"ordinary": yaml_value})
    store.apply_db_row("general_settings", {"ordinary": db_value})

    assert store["ordinary"] == expected_value
    assert store.source("ordinary") == expected_source


def test_settings_store_gives_every_config_declared_key_to_the_config_file() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"max_file_size_mb": 7, "max_parallel_requests": 3})
    store.apply_db_row("general_settings", {"max_file_size_mb": 9, "max_parallel_requests": 11})

    assert dict(store) == {"max_file_size_mb": 7, "max_parallel_requests": 3}
    assert store.source("max_file_size_mb") == "config"
    assert store.source("max_parallel_requests") == "config"


def test_settings_store_gives_a_key_the_config_file_omits_to_the_database() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"max_file_size_mb": 7})
    store.apply_db_row("general_settings", {"max_file_size_mb": 9, "max_parallel_requests": 11})

    assert dict(store) == {"max_file_size_mb": 7, "max_parallel_requests": 11}
    assert store.source("max_parallel_requests") == "db"


def test_settings_store_refuses_a_runtime_write_to_a_config_owned_key() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"max_parallel_requests": 3})

    store["max_parallel_requests"] = 11
    del store["max_parallel_requests"]

    assert store["max_parallel_requests"] == 3
    assert store.source("max_parallel_requests") == "config"


def test_settings_store_reports_the_config_owned_keys_a_write_would_change() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"max_parallel_requests": 3, "ui_access_mode": "admin_only"})

    rejected: Final = store.rejected_writes(
        {"max_parallel_requests": 11, "ui_access_mode": "admin_only", "global_max_parallel_requests": 5}
    )

    assert rejected == ("max_parallel_requests",)
    assert store.config_owned_keys() == frozenset({"max_parallel_requests", "ui_access_mode"})


def test_settings_store_resolved_view_is_read_only() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"configured": "value"})
    resolved: Final = store.resolved()

    with pytest.raises(TypeError):
        resolved["configured"] = "changed"

    assert store["configured"] == "value"


def test_settings_store_omits_a_null_database_overlay_value() -> None:
    store: Final = SettingsStore("router_settings")
    store.apply_db_row("router_settings", {"fallbacks": None})

    assert "fallbacks" not in store
    assert dict(store) == {}
    assert store.source("fallbacks") == "unset"


def test_settings_store_keeps_an_empty_database_list_without_a_config_value() -> None:
    store: Final = SettingsStore("router_settings")
    store.apply_db_row("router_settings", {"fallbacks": []})

    assert store["fallbacks"] == []
    assert store.source("fallbacks") == "db"


@pytest.mark.asyncio
async def test_load_config_returns_and_binds_the_general_settings_store(tmp_path, monkeypatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import ProxyConfig

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_list: []\ngeneral_settings:\n  max_file_size_mb: 5\n")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    proxy_config: Final = ProxyConfig()
    _router, _models, returned_store = await proxy_config.load_config(router=None, config_file_path=str(config_path))

    config_state: Final = proxy_config.get_config_state()

    assert returned_store is proxy_config.settings
    assert proxy_server.general_settings is proxy_config.settings
    assert isinstance(config_state["general_settings"], dict)
    assert config_state["general_settings"]["max_file_size_mb"] == 5


def test_settings_store_starts_with_an_unset_source() -> None:
    store: Final = SettingsStore("general_settings")

    assert store.source("unknown") == "unset"

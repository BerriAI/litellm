from __future__ import annotations

from typing import Final

import pytest

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


def test_settings_store_preserves_falsy_config_values_and_provenance() -> None:
    store: Final = SettingsStore("general_settings")
    yaml_values: Final = {"none": None, "false": False, "zero": 0, "empty_list": [], "empty_string": ""}

    store.load_yaml(yaml_values)

    assert dict(store) == yaml_values
    assert tuple(store.source(key) for key in yaml_values) == ("config",) * len(yaml_values)


@pytest.mark.parametrize(
    ("yaml_value", "db_value", "expected_value", "expected_source"),
    (
        ("from-config", "from-db", "from-db", "db"),
        ("from-config", None, "from-config", "config"),
        (None, "from-db", "from-db", "db"),
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


def test_settings_store_applies_the_registered_config_precedence_rule() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"max_file_size_mb": 7, "max_parallel_requests": 3})
    store.apply_db_row("general_settings", {"max_file_size_mb": 9, "max_parallel_requests": 11})

    assert dict(store) == {"max_file_size_mb": 7, "max_parallel_requests": 11}
    assert store.source("max_file_size_mb") == "config"
    assert store.source("max_parallel_requests") == "db"


def test_settings_store_resolved_view_is_read_only() -> None:
    store: Final = SettingsStore("general_settings")
    store.load_yaml({"configured": "value"})
    resolved: Final = store.resolved()

    with pytest.raises(TypeError):
        resolved["configured"] = "changed"

    assert store["configured"] == "value"


def test_settings_store_starts_with_an_unset_source() -> None:
    store: Final = SettingsStore("general_settings")

    assert store.source("unknown") == "unset"

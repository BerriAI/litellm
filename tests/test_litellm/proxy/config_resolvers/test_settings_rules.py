from __future__ import annotations

import itertools
from typing import Final

import pytest

from litellm.proxy.config_resolvers.settings_rules import (
    ABSENT,
    DUAL_SOURCE_KEYS,
    Absent,
    JsonValue,
    Section,
    SettingValue,
    is_absent,
    resolve,
    rule_for,
)
from litellm.proxy.config_resolvers.settings_store import SettingsStore

_SECTIONS: Final[tuple[Section, ...]] = (
    "general_settings",
    "router_settings",
    "litellm_settings",
    "environment_variables",
)

_ROUTES: Final[tuple[tuple[Section, str], ...]] = (
    ("general_settings", "max_parallel_requests"),
    ("general_settings", "max_file_size_mb"),
    ("general_settings", "alerting"),
    ("general_settings", "pass_through_endpoints"),
    ("general_settings", "forward_client_headers_to_llm_api"),
    ("router_settings", "fallbacks"),
    ("litellm_settings", "drop_params"),
    ("general_settings", "an_unregistered_key"),
)

_CONFIG_VALUES: Final[tuple[SettingValue, ...]] = (
    ABSENT,
    None,
    False,
    0,
    "",
    [],
    {},
    "config-value",
    ["config-value"],
    {"config": "value"},
    [{"path": "/shared", "target": "config"}],
)

_DB_VALUES: Final[tuple[SettingValue, ...]] = (
    ABSENT,
    None,
    False,
    0,
    "",
    [],
    {},
    "db-value",
    ["db-value"],
    {"db": "value"},
    [{"path": "/shared", "target": "db"}],
)

_MATRIX: Final = tuple(
    (section, key, config_value, db_value)
    for (section, key), config_value, db_value in itertools.product(_ROUTES, _CONFIG_VALUES, _DB_VALUES)
)

_PREVIOUSLY_DB_WINS: Final[tuple[str, ...]] = (
    "max_parallel_requests",
    "global_max_parallel_requests",
    "alerting_args",
    "ui_access_mode",
    "disable_auto_add_proxy_admin_to_teams",
    "store_model_in_db",
    "maximum_spend_logs_retention_period",
    "maximum_autorouter_session_retention_period",
    "maximum_health_check_retention_period",
    "maximum_daily_tag_spend_retention_period",
    "maximum_spend_logs_cleanup_batch_size",
    "maximum_spend_logs_cleanup_max_batches",
    "maximum_spend_logs_cleanup_run_budget",
    "maximum_spend_logs_cleanup_batch_timeout",
    "user_url_validation",
    "user_url_allowed_hosts",
    "provider_url_destination_allowed_hosts",
    "alerting",
    "pass_through_endpoints",
)


def _store_for(section: Section, key: str, config_value: SettingValue, db_value: SettingValue) -> SettingsStore:
    store: Final = SettingsStore(section)
    store.load_yaml({} if is_absent(config_value) else {key: config_value})
    if not is_absent(db_value):
        store.apply_db_row(rule_for(section, key).db_row, {key: db_value})
    return store


@pytest.mark.parametrize(("section", "key", "config_value", "db_value"), _MATRIX)
def test_the_store_resolves_every_config_and_stored_value_combination(
    section: Section, key: str, config_value: SettingValue, db_value: SettingValue
) -> None:
    store: Final = _store_for(section, key, config_value, db_value)

    if not is_absent(config_value):
        assert store[key] == config_value
        assert store.source(key) == "config"
    elif is_absent(db_value) or db_value is None:
        assert key not in store
        assert store.source(key) == "unset"
    else:
        assert store[key] == db_value
        assert store.source(key) == "db"


@pytest.mark.parametrize(("section", "key", "config_value", "db_value"), _MATRIX)
def test_the_store_and_the_resolver_never_disagree(
    section: Section, key: str, config_value: SettingValue, db_value: SettingValue
) -> None:
    resolved: Final = resolve(config_value, db_value)
    store: Final = _store_for(section, key, config_value, db_value)

    assert store.source(key) == resolved.source
    if isinstance(resolved.value, Absent):
        assert key not in store
    else:
        assert store[key] == resolved.value


@pytest.mark.parametrize(("section", "key"), _ROUTES)
def test_a_stored_row_the_key_does_not_belong_to_never_reaches_it(section: Section, key: str) -> None:
    other_row: Final = "ui_settings" if rule_for(section, key).db_row != "ui_settings" else "general_settings"
    store: Final = SettingsStore(section)
    store.load_yaml({})
    store.apply_db_row(other_row, {key: "from-the-wrong-row"})

    assert key not in store
    assert store.source(key) == "unset"


@pytest.mark.parametrize("key", _PREVIOUSLY_DB_WINS)
def test_keys_the_database_used_to_win_now_resolve_to_the_config_value(key: str) -> None:
    store: Final = _store_for("general_settings", key, "from-config", "from-db")

    assert store[key] == "from-config"
    assert store.source(key) == "config"


@pytest.mark.parametrize("key", _PREVIOUSLY_DB_WINS)
def test_a_falsy_stored_value_cannot_erase_a_config_value(key: str) -> None:
    falsy: Final[tuple[JsonValue, ...]] = (None, False, 0, "", [], {})

    stores: Final = tuple(_store_for("general_settings", key, "from-config", value) for value in falsy)

    assert {store[key] for store in stores} == {"from-config"}
    assert {store.source(key) for store in stores} == {"config"}


@pytest.mark.parametrize(
    ("key", "expected_row"),
    (
        ("forward_client_headers_to_llm_api", "ui_settings"),
        ("team_admin_editable_team_fields", "ui_settings"),
        ("disable_key_generate_for_org_admin", "ui_settings"),
        ("max_parallel_requests", "general_settings"),
        ("an_unregistered_key", "general_settings"),
    ),
)
def test_a_key_reads_from_the_row_that_carries_it(key: str, expected_row: str) -> None:
    assert rule_for("general_settings", key).db_row == expected_row


def test_every_registered_rule_routes_to_a_known_row() -> None:
    rows: Final = {rule.db_row for rule in DUAL_SOURCE_KEYS.values()}

    assert rows <= {*_SECTIONS, "ui_settings"}


def test_a_config_value_of_none_is_still_config_owned() -> None:
    resolved: Final = resolve(None, "from-db")

    assert resolved.value is None
    assert resolved.source == "config"

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

_SECTIONS: Final[tuple[Section, ...]] = (
    "general_settings",
    "router_settings",
    "litellm_settings",
    "environment_variables",
)

# One route per shape the resolver has to serve: a key that used to be database-owned,
# one that was already config-owned, the collection keys that used to merge, a key
# carried by a different stored row, another section, and a key with no rule at all.
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

_CONFIG_OWNED_MATRIX: Final = tuple(
    (section, key, config_value, db_value)
    for (section, key), config_value, db_value in itertools.product(_ROUTES, _CONFIG_VALUES, _DB_VALUES)
    if not is_absent(config_value)
)
_DB_FALLBACK_MATRIX: Final = tuple(
    (section, key, db_value) for (section, key), db_value in itertools.product(_ROUTES, _DB_VALUES)
)

# Keys the database used to win outright. The flip is the breaking change this PR ships,
# so each one is named rather than generated: a revert has to fail here.
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


@pytest.mark.parametrize(("section", "key", "config_value", "db_value"), _CONFIG_OWNED_MATRIX)
def test_a_key_the_config_file_declares_always_resolves_to_the_config_value(
    section: Section, key: str, config_value: SettingValue, db_value: SettingValue
) -> None:
    resolved: Final = resolve(rule_for(section, key), config_value, db_value)

    assert resolved.value == config_value
    assert resolved.source == "config"


@pytest.mark.parametrize(("section", "key", "db_value"), _DB_FALLBACK_MATRIX)
def test_a_key_the_config_file_omits_falls_back_to_the_stored_value(
    section: Section, key: str, db_value: SettingValue
) -> None:
    resolved: Final = resolve(rule_for(section, key), ABSENT, db_value)

    if is_absent(db_value) or db_value is None:
        assert isinstance(resolved.value, Absent)
        assert resolved.source == "unset"
    else:
        assert resolved.value == db_value
        assert resolved.source == "db"


@pytest.mark.parametrize("key", _PREVIOUSLY_DB_WINS)
def test_keys_the_database_used_to_win_now_resolve_to_the_config_value(key: str) -> None:
    resolved: Final = resolve(rule_for("general_settings", key), "from-config", "from-db")

    assert resolved.value == "from-config"
    assert resolved.source == "config"


@pytest.mark.parametrize("key", _PREVIOUSLY_DB_WINS)
def test_a_falsy_stored_value_cannot_erase_a_config_value(key: str) -> None:
    falsy: Final[tuple[JsonValue, ...]] = (None, False, 0, "", [], {})

    resolved: Final = tuple(resolve(rule_for("general_settings", key), "from-config", value) for value in falsy)

    assert {entry.value for entry in resolved} == {"from-config"}
    assert {entry.source for entry in resolved} == {"config"}


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
    resolved: Final = resolve(rule_for("general_settings", "ui_access_mode"), None, "from-db")

    assert resolved.value is None
    assert resolved.source == "config"

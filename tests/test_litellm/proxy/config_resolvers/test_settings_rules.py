from __future__ import annotations

from typing import Final

import pytest

from litellm.proxy.config_resolvers._descriptors import FieldSource
from litellm.proxy.config_resolvers.settings_rules import ABSENT, DUAL_SOURCE_KEYS, KeyRule, Resolved, resolve


@pytest.mark.parametrize(
    ("rule", "yaml_value", "db_value", "expected"),
    (
        (
            KeyRule(db_row="general_settings", kind="db_wins"),
            "from-config",
            "from-db",
            Resolved(value="from-db", source="db"),
        ),
        (
            KeyRule(db_row="general_settings", kind="config_wins"),
            "from-config",
            "from-db",
            Resolved(value="from-config", source="config"),
        ),
        (
            KeyRule(db_row="general_settings", kind="db_fallback_to_config"),
            "from-config",
            None,
            Resolved(value="from-config", source="config"),
        ),
        (
            KeyRule(db_row="general_settings", kind="list_union"),
            ["config", "shared"],
            ["db", "shared"],
            Resolved(value=["config", "shared", "db"], source="db"),
        ),
        (
            KeyRule(db_row="general_settings", kind="merge_by_path"),
            [{"path": "/config"}, {"path": "/shared", "source": "config"}],
            [{"path": "/db"}, {"path": "/shared", "source": "db"}],
            Resolved(
                value=[
                    {"path": "/db"},
                    {"path": "/shared", "source": "db"},
                    {"path": "/config"},
                ],
                source="db",
            ),
        ),
        (
            KeyRule(db_row="router_settings", kind="db_overlay"),
            {"config": 1, "nested": {"config": True, "shared": "config"}, "fallbacks": ["config"]},
            {"db": 2, "nested": {"shared": "db", "db": True}, "fallbacks": []},
            Resolved(
                value={
                    "config": 1,
                    "db": 2,
                    "nested": {"config": True, "shared": "db", "db": True},
                    "fallbacks": ["config"],
                },
                source="db",
            ),
        ),
    ),
)
def test_resolve_matches_the_config_and_db_precedence_rules(
    rule: KeyRule,
    yaml_value: object,
    db_value: object,
    expected: Resolved,
) -> None:
    assert resolve(rule, yaml_value, db_value) == expected


@pytest.mark.parametrize("rule", tuple(DUAL_SOURCE_KEYS.values()))
def test_resolve_treats_none_from_the_database_as_absent(rule: KeyRule) -> None:
    resolved: Final = resolve(rule, "from-config", None)

    assert resolved == Resolved(value="from-config", source="config")


def test_resolve_distinguishes_an_absent_config_value_from_a_configured_null() -> None:
    absent: Final = resolve(KeyRule(db_row="general_settings", kind="db_wins"), ABSENT, None)
    configured_null: Final = resolve(KeyRule(db_row="general_settings", kind="db_wins"), None, None)

    assert absent == Resolved(value=ABSENT, source="unset")
    assert configured_null == Resolved(value=None, source="config")


def test_resolve_reports_config_db_and_unset_sources() -> None:
    rule: Final = KeyRule(db_row="general_settings", kind="db_wins")
    sources: Final[tuple[FieldSource, ...]] = (
        resolve(rule, "from-config", None).source,
        resolve(rule, "from-config", "from-db").source,
        resolve(rule, ABSENT, None).source,
    )

    assert sources == ("config", "db", "unset")

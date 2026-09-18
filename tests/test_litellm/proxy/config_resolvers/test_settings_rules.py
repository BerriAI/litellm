from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast

import pytest

from litellm.proxy.config_resolvers._descriptors import FieldSource
from litellm.proxy.config_resolvers.settings_rules import (
    ABSENT,
    DUAL_SOURCE_KEYS,
    Absent,
    JsonValue,
    KeyRule,
    Resolved,
    Section,
    SettingValue,
    _build_dual_source_keys,
    resolve,
    rule_for,
)


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


_PRECEDENCE_MATRIX_PATH: Final = Path(__file__).parent / "fixtures" / "precedence_matrix.json"


def _load_precedence_matrix() -> tuple[dict[str, object], ...]:
    raw: Final[object] = json.loads(_PRECEDENCE_MATRIX_PATH.read_text())
    assert isinstance(raw, dict)
    cases: Final[object] = raw.get("cases")
    assert isinstance(cases, list)
    assert all(isinstance(case, dict) for case in cases)
    return tuple(cast(dict[str, object], case) for case in cases)


def _matrix_value(case: Mapping[str, object], source: Literal["config", "db"]) -> SettingValue:
    raw_value: Final[object] = case[source]
    assert isinstance(raw_value, Mapping)
    present: Final[object] = raw_value.get("present")
    assert isinstance(present, bool)
    if not present:
        return ABSENT
    return cast(JsonValue, raw_value["value"])


def test_dual_source_key_registry_matches_the_golden_precedence_matrix() -> None:
    registry: Final = _build_dual_source_keys()

    for case in _load_precedence_matrix():
        section: Final[object] = case["section"]
        key: Final[object] = case["key"]
        rule_kind: Final[object] = case["rule"]
        db_row: Final[object] = case["db_row"]
        assert isinstance(section, str)
        assert isinstance(key, str)
        assert isinstance(rule_kind, str)
        assert isinstance(db_row, str)
        resolved_rule: Final = registry.get((cast(Section, section), key), registry[(cast(Section, section), "*")])
        assert resolved_rule.kind == rule_kind
        assert resolved_rule.db_row == db_row


@pytest.mark.parametrize("case", _load_precedence_matrix())
def test_resolve_matches_the_golden_precedence_matrix(case: dict[str, object]) -> None:
    section: Final[object] = case["section"]
    key: Final[object] = case["key"]
    rule_kind: Final[object] = case["rule"]
    expected: Final[object] = case["expected"]
    assert isinstance(section, str)
    assert isinstance(key, str)
    assert isinstance(rule_kind, str)
    assert isinstance(expected, Mapping)

    resolved: Final = resolve(
        rule_for(cast(Section, section), key),
        _matrix_value(case, "config"),
        _matrix_value(case, "db"),
    )

    expected_present: Final[object] = expected["present"]
    assert isinstance(expected_present, bool)
    assert rule_for(cast(Section, section), key).kind == rule_kind
    assert not isinstance(resolved.value, Absent) is expected_present
    if expected_present:
        assert resolved.value == expected["value"]
    assert resolved.source == expected["source"]

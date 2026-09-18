from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.proxy.config_resolvers._descriptors import FieldSource

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
Section: TypeAlias = Literal[
    "general_settings",
    "router_settings",
    "litellm_settings",
    "environment_variables",
    "ui_settings",
]
DbRow: TypeAlias = Section


@dataclass(frozen=True, slots=True)
class Absent:
    pass


ABSENT: Final = Absent()
SettingValue: TypeAlias = JsonValue | Absent


@dataclass(frozen=True, slots=True)
class KeyRule:
    """Which stored row carries this key. Precedence no longer varies per key."""

    db_row: DbRow


@dataclass(frozen=True, slots=True)
class Resolved:
    value: SettingValue
    source: FieldSource


_UI_SETTINGS_FIELDS: Final[tuple[str, ...]] = (
    "allow_public_health_readiness_details",
    "forward_client_headers_to_llm_api",
    "forward_llm_provider_auth_headers",
    "disable_agents_for_internal_users",
    "allow_agents_for_team_admins",
    "disable_vector_stores_for_internal_users",
    "allow_vector_stores_for_team_admins",
    "disable_key_generate_for_org_admin",
    "team_admin_editable_team_fields",
)


def _rules_for(
    section: Section, keys: tuple[str, ...], db_row: DbRow
) -> tuple[tuple[tuple[Section, str], KeyRule], ...]:
    return tuple(((section, key), KeyRule(db_row=db_row)) for key in keys)


def _build_dual_source_keys() -> Mapping[tuple[Section, str], KeyRule]:
    """Maps a key to the stored row that carries it, for the keys whose row is not their own section."""
    return MappingProxyType(
        dict(
            (
                *_rules_for("general_settings", _UI_SETTINGS_FIELDS, "ui_settings"),
                *(
                    ((section, "*"), KeyRule(db_row=section))
                    for section in ("general_settings", "router_settings", "litellm_settings", "environment_variables")
                ),
            )
        )
    )


DUAL_SOURCE_KEYS: Final[Mapping[tuple[Section, str], KeyRule]] = _build_dual_source_keys()


def rule_for(section: Section, key: str) -> KeyRule:
    return DUAL_SOURCE_KEYS.get((section, key), DUAL_SOURCE_KEYS[(section, "*")])


def coerce_bool(value: JsonValue) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def resolve(rule: KeyRule, yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    """Config wins. A key the config file declares is config-owned, whatever the database holds.

    ``rule`` only selects which stored row the database value came from; it no longer
    varies the precedence. A stored ``null`` still counts as absent, so clearing a row
    does not erase a value the file never declared.
    """
    del rule
    if yaml_value is not ABSENT:
        return Resolved(value=yaml_value, source="config")
    if _db_is_present(db_value):
        return Resolved(value=db_value, source="db")
    return Resolved(value=ABSENT, source="unset")








def is_absent(value: SettingValue) -> bool:
    return value is ABSENT


def _db_is_present(value: SettingValue) -> bool:
    return not is_absent(value) and value is not None



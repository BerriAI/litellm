from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from typing_extensions import (
    assert_never,
)

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
RuleKind: TypeAlias = Literal[
    "db_wins",
    "config_wins",
    "db_fallback_to_config",
    "list_union",
    "merge_by_path",
    "db_overlay",
]


@dataclass(frozen=True, slots=True)
class Absent:
    pass


ABSENT: Final = Absent()
SettingValue: TypeAlias = JsonValue | Absent


@dataclass(frozen=True, slots=True)
class KeyRule:
    db_row: DbRow
    kind: RuleKind


@dataclass(frozen=True, slots=True)
class Resolved:
    value: SettingValue
    source: FieldSource


_DB_GENERAL_SETTINGS: Final[tuple[str, ...]] = (
    "max_parallel_requests",
    "global_max_parallel_requests",
    "alerting_args",
    "ui_access_mode",
    "disable_auto_add_proxy_admin_to_teams",
    "store_model_in_db",
    "maximum_spend_logs_retention_period",
    "maximum_autorouter_session_retention_period",
    "maximum_health_check_retention_period",
    "user_url_validation",
    "user_url_allowed_hosts",
    "provider_url_destination_allowed_hosts",
)
_CONFIG_GENERAL_SETTINGS: Final[tuple[str, ...]] = (
    "max_batch_file_size_mb",
    "max_file_size_mb",
    "allowed_file_extensions",
    "blocked_file_extensions",
    "store_prompts_in_spend_logs",
    "apply_user_budget_to_team_keys",
    "enable_openai_websocket_passthrough",
    "user_api_key_cache_max_size",
)
_CLEANUP_BOUNDS: Final[tuple[str, ...]] = (
    "maximum_spend_logs_cleanup_batch_size",
    "maximum_spend_logs_cleanup_max_batches",
    "maximum_spend_logs_cleanup_run_budget",
    "maximum_spend_logs_cleanup_batch_timeout",
)
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
    section: Section, keys: tuple[str, ...], db_row: DbRow, kind: RuleKind
) -> tuple[tuple[tuple[Section, str], KeyRule], ...]:
    return tuple(((section, key), KeyRule(db_row=db_row, kind=kind)) for key in keys)


def _build_dual_source_keys() -> Mapping[tuple[Section, str], KeyRule]:
    return MappingProxyType(
        dict(
            (
                *_rules_for("general_settings", _DB_GENERAL_SETTINGS, "general_settings", "db_wins"),
                *_rules_for("general_settings", _CONFIG_GENERAL_SETTINGS, "general_settings", "config_wins"),
                *_rules_for("general_settings", _CLEANUP_BOUNDS, "general_settings", "db_fallback_to_config"),
                (("general_settings", "alerting"), KeyRule(db_row="general_settings", kind="list_union")),
                (
                    ("general_settings", "pass_through_endpoints"),
                    KeyRule(db_row="general_settings", kind="merge_by_path"),
                ),
                (("general_settings", "*"), KeyRule(db_row="general_settings", kind="db_overlay")),
                (("router_settings", "*"), KeyRule(db_row="router_settings", kind="db_overlay")),
                (("litellm_settings", "*"), KeyRule(db_row="litellm_settings", kind="db_overlay")),
                (("environment_variables", "*"), KeyRule(db_row="environment_variables", kind="db_overlay")),
                *_rules_for("general_settings", _UI_SETTINGS_FIELDS, "ui_settings", "db_wins"),
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
    match rule.kind:
        case "db_wins" | "db_fallback_to_config":
            return _db_wins(yaml_value, db_value)
        case "config_wins":
            return _config_wins(yaml_value, db_value)
        case "list_union":
            return _list_union(yaml_value, db_value)
        case "merge_by_path":
            return _merge_by_path(yaml_value, db_value)
        case "db_overlay":
            return _db_overlay(yaml_value, db_value)
        case _:
            assert_never(rule.kind)


def _db_wins(yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    if _db_is_present(db_value):
        return Resolved(value=db_value, source="db")
    if yaml_value is not ABSENT:
        return Resolved(value=yaml_value, source="config")
    return Resolved(value=ABSENT, source="unset")


def _config_wins(yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    if yaml_value is not ABSENT:
        return Resolved(value=yaml_value, source="config")
    if _db_is_present(db_value):
        return Resolved(value=db_value, source="db")
    return Resolved(value=ABSENT, source="unset")


def _list_union(yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    if not _db_is_present(db_value):
        return _db_wins(yaml_value, db_value)
    if not isinstance(yaml_value, list) or not isinstance(db_value, list):
        return _db_wins(yaml_value, db_value)
    merged: Final[list[JsonValue]] = [  # mutable-ok: resolved config values retain the legacy JSON-list contract
        *yaml_value,
        *(value for value in db_value if value not in yaml_value),
    ]
    return Resolved(value=merged, source="db")


def _merge_by_path(yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    if not _db_is_present(db_value):
        return _db_wins(yaml_value, db_value)
    if not isinstance(yaml_value, list) or not isinstance(db_value, list):
        return _db_wins(yaml_value, db_value)
    db_paths: Final = frozenset(_endpoint_path(value) for value in db_value if _endpoint_path(value) is not None)
    merged: Final[list[JsonValue]] = [  # mutable-ok: resolved config values retain the legacy JSON-list contract
        *db_value,
        *(value for value in yaml_value if _endpoint_path(value) not in db_paths),
    ]
    return Resolved(value=merged, source="db")


def _db_overlay(yaml_value: SettingValue, db_value: SettingValue) -> Resolved:
    if not _db_is_present(db_value):
        return _db_wins(yaml_value, db_value)
    if isinstance(db_value, list) and not db_value and yaml_value is not ABSENT:
        return Resolved(value=yaml_value, source="config")
    if not isinstance(yaml_value, dict) or not isinstance(db_value, dict):
        return _db_wins(yaml_value, db_value)
    overlay: Final = _overlay_mapping(yaml_value, db_value)
    source: Final[FieldSource] = "db" if overlay != yaml_value else "config"
    return Resolved(value=overlay, source=source)


def _overlay_mapping(yaml_value: dict[str, JsonValue], db_value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return dict(  # mutable-ok: resolved config values retain the legacy JSON-object contract
        (
            *(
                (key, _overlay_value(value, db_value[key]) if key in db_value else value)
                for key, value in yaml_value.items()
            ),
            *(
                (key, value)
                for key, value in db_value.items()
                if key not in yaml_value and not _db_overlay_defers(value)
            ),
        )
    )


def _overlay_value(yaml_value: JsonValue, db_value: JsonValue) -> JsonValue:
    if isinstance(yaml_value, dict) and isinstance(db_value, dict):
        return _overlay_mapping(yaml_value, db_value)
    return yaml_value if _db_overlay_defers(db_value) else db_value


def _db_overlay_defers(value: JsonValue) -> bool:
    return value is None or (isinstance(value, list) and not value)


def is_absent(value: SettingValue) -> bool:
    return value is ABSENT


def _db_is_present(value: SettingValue) -> bool:
    return not is_absent(value) and value is not None


def _endpoint_path(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    path: Final = value.get("path")
    return path if isinstance(path, str) else None

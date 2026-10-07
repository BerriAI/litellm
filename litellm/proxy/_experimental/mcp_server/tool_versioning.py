import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final

from pydantic import TypeAdapter

from litellm.types.mcp_server.mcp_server_manager import (
    MCPToolChange,
    MCPToolChangeKind,
    MCPToolVersion,
    PinnedMCPTool,
)


@dataclass(frozen=True, slots=True)
class PlannedToolVersion:
    tool_name: str
    version: int
    tool: PinnedMCPTool
    change_kind: MCPToolChangeKind
    changes: tuple[MCPToolChange, ...]


def _properties(input_schema: Mapping[str, object]) -> Mapping[str, object]:
    properties: Final[object] = input_schema.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _required(input_schema: Mapping[str, object]) -> frozenset[str]:
    required: Final[object] = input_schema.get("required")
    if not isinstance(required, list):
        return frozenset()
    return frozenset(value for value in required if isinstance(value, str))


_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_ARRAY: Final = TypeAdapter(tuple[object, ...])


def _mapping_without_documentation(schema: Mapping[str, object]) -> Mapping[str, object]:
    mapping_keys: Final = frozenset(("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"))
    list_keys: Final = frozenset(("items", "prefixItems", "anyOf", "oneOf", "allOf"))
    schema_keys: Final = frozenset(
        (
            "items",
            "additionalProperties",
            "unevaluatedProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
        )
    )
    return {
        key: (
            {
                child_key: _schema_without_documentation(child)
                for child_key, child in _OBJECT_MAPPING.validate_python(item).items()
            }
            if key in mapping_keys and isinstance(item, Mapping)
            else [_schema_without_documentation(child) for child in _OBJECT_ARRAY.validate_python(item)]
            if key in list_keys and isinstance(item, list)
            else _schema_without_documentation(item)
            if key in schema_keys and isinstance(item, Mapping)
            else item
        )
        for key, item in schema.items()
        if key not in ("description", "title")
    }


def _schema_without_documentation(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    return _mapping_without_documentation(_OBJECT_MAPPING.validate_python(value))


def _type_label(parameter_schema: object) -> str:
    if not isinstance(parameter_schema, Mapping) or "type" not in parameter_schema:
        return "any"
    return json.dumps(parameter_schema["type"])


def _classify_parameter(
    name: str,
    previous_properties: Mapping[str, object],
    current_properties: Mapping[str, object],
    previous_required: frozenset[str],
    current_required: frozenset[str],
) -> tuple[MCPToolChange, ...]:
    in_previous: Final = name in previous_properties
    in_current: Final = name in current_properties
    required_change: Final[tuple[MCPToolChange, ...]] = (
        (MCPToolChange(breaking=True, summary=f'Parameter "{name}" is now required'),)
        if name not in previous_required and name in current_required
        else (
            (MCPToolChange(breaking=False, summary=f'Parameter "{name}" is now optional'),)
            if name in previous_required and name not in current_required
            else ()
        )
    )
    if not in_previous and not in_current:
        return required_change
    if in_previous and not in_current:
        return (MCPToolChange(breaking=True, summary=f'Removed parameter "{name}"'),)
    if in_current and not in_previous:
        is_required: Final = name in current_required
        summary: Final = f'Added required parameter "{name}"' if is_required else f'Added optional parameter "{name}"'
        return (MCPToolChange(breaking=is_required, summary=summary),)

    previous_parameter: Final = previous_properties[name]
    current_parameter: Final = current_properties[name]
    previous_type: Final = _type_label(previous_parameter)
    current_type: Final = _type_label(current_parameter)
    schema_change: Final[tuple[MCPToolChange, ...]] = (
        (
            MCPToolChange(
                breaking=True,
                summary=f'Parameter "{name}" type changed from {previous_type} to {current_type}',
            ),
        )
        if previous_type != current_type
        else (
            (MCPToolChange(breaking=True, summary=f'Parameter "{name}" schema changed'),)
            if _schema_without_documentation(previous_parameter) != _schema_without_documentation(current_parameter)
            else (
                (MCPToolChange(breaking=False, summary=f'Parameter "{name}" description changed'),)
                if previous_parameter != current_parameter
                else ()
            )
        )
    )
    return schema_change + required_change


def classify_tool_change(previous: PinnedMCPTool, current: PinnedMCPTool) -> tuple[MCPToolChange, ...]:
    previous_properties: Final = _properties(previous.input_schema)
    current_properties: Final = _properties(current.input_schema)
    previous_required: Final = _required(previous.input_schema)
    current_required: Final = _required(current.input_schema)
    parameter_names: Final = sorted(
        set(previous_properties) | set(current_properties) | previous_required | current_required
    )
    parameter_changes: Final = tuple(
        chain.from_iterable(
            _classify_parameter(
                name,
                previous_properties,
                current_properties,
                previous_required,
                current_required,
            )
            for name in parameter_names
        )
    )

    excluded_top_level_keys: Final = frozenset(("properties", "required", "description", "title"))
    previous_top_level_raw: Final = {
        key: value for key, value in previous.input_schema.items() if key not in excluded_top_level_keys
    }
    current_top_level_raw: Final = {
        key: value for key, value in current.input_schema.items() if key not in excluded_top_level_keys
    }
    previous_top_level: Final = {
        key: value
        for key, value in _mapping_without_documentation(previous.input_schema).items()
        if key not in excluded_top_level_keys
    }
    current_top_level: Final = {
        key: value
        for key, value in _mapping_without_documentation(current.input_schema).items()
        if key not in excluded_top_level_keys
    }
    description_changes: Final = (
        (MCPToolChange(breaking=False, summary="Description changed"),)
        if previous.description != current.description
        else ()
    )
    input_schema_description_changes: Final = (
        (MCPToolChange(breaking=False, summary="Input schema description changed"),)
        if previous.input_schema.get("description") != current.input_schema.get("description")
        else ()
    )
    input_schema_title_changes: Final = (
        (MCPToolChange(breaking=False, summary="Input schema title changed"),)
        if previous.input_schema.get("title") != current.input_schema.get("title")
        else ()
    )
    schema_changes: Final = (
        (MCPToolChange(breaking=True, summary="Input schema changed"),)
        if previous_top_level != current_top_level
        else (
            (MCPToolChange(breaking=False, summary="Input schema documentation changed"),)
            if previous_top_level_raw != current_top_level_raw
            else ()
        )
    )
    return (
        description_changes
        + input_schema_description_changes
        + input_schema_title_changes
        + parameter_changes
        + schema_changes
    )


def _plan_tool_version(
    name: str,
    latest: Mapping[str, MCPToolVersion],
    snapshot: Mapping[str, PinnedMCPTool],
) -> tuple[PlannedToolVersion, ...]:
    has_snapshot: Final = name in snapshot
    has_latest: Final = name in latest
    if has_snapshot and not has_latest:
        return (PlannedToolVersion(name, 1, snapshot[name], "initial", ()),)
    if has_snapshot and has_latest:
        current_tool: Final = snapshot[name]
        latest_version: Final = latest[name]
        is_restored: Final = latest_version.change_kind == "removed"
        changes: Final = (
            (MCPToolChange(breaking=False, summary="Tool restored"),) if is_restored else ()
        ) + classify_tool_change(
            PinnedMCPTool(description=latest_version.description, input_schema=latest_version.input_schema),
            current_tool,
        )
        if not changes:
            return ()
        change_kind: Final[MCPToolChangeKind] = (
            "breaking" if any(change.breaking for change in changes) else "non_breaking"
        )
        return (PlannedToolVersion(name, latest_version.version + 1, current_tool, change_kind, changes),)
    if has_latest and latest[name].change_kind != "removed":
        previous_version: Final = latest[name]
        return (
            PlannedToolVersion(
                name,
                previous_version.version + 1,
                PinnedMCPTool(description=previous_version.description, input_schema=previous_version.input_schema),
                "removed",
                (MCPToolChange(breaking=True, summary="Tool removed"),),
            ),
        )
    return ()


def plan_tool_versions(
    latest: Mapping[str, MCPToolVersion],
    snapshot: Mapping[str, PinnedMCPTool],
) -> tuple[PlannedToolVersion, ...]:
    names: Final = sorted(set(latest) | set(snapshot))
    return tuple(chain.from_iterable(_plan_tool_version(name, latest, snapshot) for name in names))

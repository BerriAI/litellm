from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from mcp.types import Tool

_PRIMITIVE_TYPES: Final = frozenset({"string", "integer", "number", "boolean"})
_INPUT_FIELD_KEYS: Final = frozenset({"type", "title", "description"})
_OBJECT_KEYS: Final = frozenset({"type", "properties", "required", "title", "description", "additionalProperties"})


@dataclass(frozen=True, slots=True)
class WorkflowCheck:
    binding_fields: tuple[str, ...]
    first_step_inputs: tuple[str, ...]
    second_step_inputs: tuple[str, ...]


def _required_fields(schema: Mapping[str, object]) -> tuple[str, ...]:
    required: Final = schema.get("required")
    if not isinstance(required, (list, tuple)):
        return ()
    fields: Final = cast(list[object] | tuple[object, ...], required)
    if not all(isinstance(field, str) for field in fields):
        return ()
    return tuple(cast(str, field) for field in fields)


def _properties(schema: Mapping[str, object]) -> Mapping[str, object]:
    properties: Final = schema.get("properties")
    if not isinstance(properties, Mapping) or not all(
        isinstance(key, str) for key in cast(Mapping[object, object], properties)
    ):
        return {}
    return cast(Mapping[str, object], properties)


def _compatible_field(source: object, target: object) -> bool:
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return False
    source_schema: Final = cast(Mapping[str, object], source)
    target_schema: Final = cast(Mapping[str, object], target)
    if not set(source_schema).issubset(_INPUT_FIELD_KEYS) or not set(target_schema).issubset(_INPUT_FIELD_KEYS):
        return False
    source_type: Final = source_schema.get("type")
    target_type: Final = target_schema.get("type")
    return isinstance(source_type, str) and source_type in _PRIMITIVE_TYPES and source_type == target_type


def check_tool_workflow(first: Tool, second: Tool) -> WorkflowCheck:
    first_input: Final = first.input_schema
    first_output: Final = first.output_schema or {}
    second_input: Final = second.input_schema
    first_required: Final = _required_fields(first_input)
    second_required: Final = _required_fields(second_input)
    output_required: Final = frozenset(_required_fields(first_output))
    output_properties: Final = _properties(first_output)
    input_properties: Final = _properties(second_input)
    simple_schemas: Final = (
        first_output.get("type") == "object"
        and second_input.get("type") == "object"
        and set(first_output).issubset(_OBJECT_KEYS)
        and set(second_input).issubset(_OBJECT_KEYS)
    )
    bindings: Final = (
        tuple(
            field
            for field in second_required
            if field in output_required and _compatible_field(output_properties.get(field), input_properties.get(field))
        )
        if simple_schemas
        else ()
    )
    return WorkflowCheck(
        binding_fields=bindings,
        first_step_inputs=first_required,
        second_step_inputs=tuple(field for field in second_required if field not in bindings),
    )


def find_workflow_sources(catalog: Sequence[Tool], target: Tool) -> tuple[tuple[Tool, WorkflowCheck], ...]:
    candidates: Final = tuple(
        (source, check)
        for source in catalog
        if source is not target
        for check in (check_tool_workflow(source, target),)
        if check.binding_fields
    )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -len(candidate[1].binding_fields),
                len(candidate[1].first_step_inputs),
                candidate[0].name,
            ),
        )
    )

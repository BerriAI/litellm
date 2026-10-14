"""
Shape a JSON schema into what Claude's structured-output grammar accepts, for
Anthropic ``output_format`` and Bedrock Converse and Invoke alike. Mirrors the
Anthropic SDK transform:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works
"""

import json
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

# anthropic SDK ``lib/_parse/_transform.py`` SupportedStringFormats, checked 2026-10-13
_SUPPORTED_STRING_FORMATS: Final = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)

_CONSTRAINT_NOTES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "minItems": "minimum number of items: {}",
        "maxItems": "maximum number of items: {}",
        "uniqueItems": "all array items must be unique",
        "contains": "array must contain an item matching: {}",
        "minContains": "minimum number of matching items: {}",
        "maxContains": "maximum number of matching items: {}",
        "prefixItems": "leading items must match, in order: {}",
        "minimum": "minimum value: {}",
        "maximum": "maximum value: {}",
        "exclusiveMinimum": "exclusive minimum value: {}",
        "exclusiveMaximum": "exclusive maximum value: {}",
        "multipleOf": "must be a multiple of {}",
        "minLength": "minimum length: {}",
        "maxLength": "maximum length: {}",
        "minProperties": "minimum number of properties: {}",
        "maxProperties": "maximum number of properties: {}",
        "patternProperties": "properties whose names match each pattern must satisfy: {}",
        "propertyNames": "property names must satisfy: {}",
        "dependentRequired": "dependent required properties: {}",
        "dependentSchemas": "dependent schemas: {}",
        "unevaluatedProperties": "unevaluated properties must satisfy: {}",
        "if": "conditional (if): {}",
        "then": "conditional (then): {}",
        "else": "conditional (else): {}",
        "not": "must not match: {}",
    }
)

_SCHEMA_MAP_KEYS: Final = frozenset({"properties", "$defs", "definitions"})
_COMPOSITION_KEYS: Final = frozenset({"anyOf", "oneOf", "allOf", "$ref"})

_ENUM_TYPE_CHECKS: Final[Mapping[str, Callable[[object], bool]]] = MappingProxyType(
    {
        "null": lambda v: v is None,
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
)

_SCHEMA: Final = TypeAdapter(Mapping[str, object])
_ITEMS: Final = TypeAdapter(list[object])
_TYPE_NAMES: Final = TypeAdapter(list[str])


def _as_schema(value: object) -> Mapping[str, object] | None:
    try:
        return _SCHEMA.validate_python(value)
    except ValidationError:
        return None


def _as_items(value: object) -> list[object] | None:
    try:
        return _ITEMS.validate_python(value)
    except ValidationError:
        return None


def _declared_types(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset((value,))
    try:
        return frozenset(_TYPE_NAMES.validate_python(value))
    except ValidationError:
        return frozenset()


def _enum_conflicts_with_declared_type(schema: Mapping[str, object]) -> bool:
    """Whether ``schema``'s ``enum`` cannot match its declared ``type``."""
    enum_values: Final = _as_items(schema.get("enum"))
    declared_type: Final = schema.get("type")
    if enum_values is None or declared_type is None:
        return False
    if isinstance(declared_type, str):
        check: Final = _ENUM_TYPE_CHECKS.get(declared_type)
        return check is not None and not all(check(value) for value in enum_values)
    return _as_items(declared_type) is not None


def _note_value(value: object) -> object:
    return value if value is None or isinstance(value, (str, int, float)) else json.dumps(value)


def _constraint_notes(schema: Mapping[str, object]) -> Iterator[str]:
    for key, label in _CONSTRAINT_NOTES.items():
        if schema.get(key, False) is not False:
            yield label.format(_note_value(schema[key]))


def _unsupported_format(schema: Mapping[str, object]) -> str | None:
    value: Final = schema.get("format")
    if isinstance(value, str) and value not in _SUPPORTED_STRING_FORMATS:
        return value
    return None


def _description_field(existing: object, notes: tuple[str, ...]) -> Mapping[str, object]:
    if not notes:
        return {}
    note: Final = "Note: " + ", ".join(notes) + "."
    if isinstance(existing, str) and existing:
        return {"description": existing + " " + note}
    return {"description": note}


def _object_fields(declared_type: object, schema: Mapping[str, object]) -> Mapping[str, object]:
    """Claude 400s on any object node not closed with ``additionalProperties: false`` or missing ``type``."""
    if "object" in _declared_types(declared_type):
        return {"additionalProperties": False}
    if (
        declared_type is None
        and _as_schema(schema.get("properties")) is not None
        and not _COMPOSITION_KEYS & schema.keys()
    ):
        return {"type": "object", "additionalProperties": False}
    return {}


def _shape_if_schema(value: object) -> object:
    nested: Final = _as_schema(value)
    return value if nested is None else shape_claude_output_schema(nested)


def _shape_each(value: object) -> object:
    items: Final = _as_items(value)
    return value if items is None else [_shape_if_schema(item) for item in items]


def _shape_nested(key: str, value: object) -> object:
    if key in _SCHEMA_MAP_KEYS:
        members: Final = _as_schema(value)
        return value if members is None else {name: _shape_if_schema(member) for name, member in members.items()}
    if key == "items":
        return _shape_if_schema(value)
    if key == "allOf":
        return _shape_each(value)
    return value


def _any_of_field(schema: Mapping[str, object]) -> Mapping[str, object]:
    variants: Final = [*(_as_items(schema.get("anyOf")) or ()), *(_as_items(schema.get("oneOf")) or ())]
    return {"anyOf": [_shape_if_schema(variant) for variant in variants]} if variants else {}


def shape_claude_output_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Return a new schema that Claude's structured-output API accepts. Idempotent."""
    drops_type: Final = _enum_conflicts_with_declared_type(schema)
    declared_type: Final = None if drops_type else schema.get("type")
    object_fields: Final = _object_fields(declared_type, schema)
    unsupported_format: Final = _unsupported_format(schema)
    notes: Final = (
        *_constraint_notes(schema),
        *(() if unsupported_format is None else (f"format: {unsupported_format}",)),
    )
    dropped: Final = frozenset(
        (*_CONSTRAINT_NOTES, "anyOf", "oneOf")
        + (("type",) if drops_type else ())
        + (("additionalProperties",) if object_fields else ())
        + (("description",) if notes else ())
        + (("format",) if unsupported_format is not None else ())
    )
    kept: Final = {key: _shape_nested(key, value) for key, value in schema.items() if key not in dropped}
    return {**_description_field(schema.get("description"), notes), **kept, **_any_of_field(schema), **object_fields}


def shape_claude_output_format(output_format: Mapping[str, object]) -> dict[str, object]:
    """Shape the ``schema`` inside an Anthropic ``output_format`` block, leaving the rest untouched."""
    schema: Final = _as_schema(output_format.get("schema"))
    if schema is None:
        return dict(output_format)
    return {**output_format, "schema": shape_claude_output_schema(schema)}

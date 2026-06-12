"""Pure ports of v1's gemini schema munging (vertex_ai/common_utils.py).

``build_vertex_schema`` mirrors ``_build_vertex_schema`` step for step over
plain JSON, returning new values instead of mutating, and failing closed
(``unsupported``) where v1 reaches code this port does not carry: ``$defs`` /
``$ref`` unwinding (the same "$ref inlining" fallback the anthropic port
declared) and schemas nested past v1's recursion cap. One deliberate
difference is documented inline: v1 removes ``anyOf`` null members while
iterating the same list (skipping the element after each hit); this port
filters all null members, which only differs for multiple adjacent nulls.
"""

from __future__ import annotations

from collections.abc import Callable

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...errors import TranslationError
from ...ir import PlainJson

# Keep in sync with litellm.types.llms.vertex_ai.Schema (v1 derives this set
# via get_type_hints(Schema) inside _build_vertex_schema).
_VALID_SCHEMA_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "default",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "properties",
        "propertyOrdering",
        "required",
        "minProperties",
        "maxProperties",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
    }
)

_TYPE_SPECIFIC_FIELDS = (
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
)

_SchemaResult = PlainJson | TranslationError


def strip_strict(value: PlainJson) -> PlainJson:
    """v1 ``_remove_strict_from_schema``: drop every ``strict`` key."""
    if isinstance(value, dict):
        return {k: strip_strict(v) for k, v in value.items() if k != "strict"}
    if isinstance(value, list):
        return [strip_strict(item) for item in value]
    return value


def strip_additional_properties(value: PlainJson) -> PlainJson:
    """v1 ``_remove_additional_properties``: drop ``additionalProperties``
    only when it is exactly ``False``."""
    if isinstance(value, dict):
        return {
            k: strip_additional_properties(v)
            for k, v in value.items()
            if not (k == "additionalProperties" and v is False)
        }
    if isinstance(value, list):
        return [strip_additional_properties(item) for item in value]
    return value


def _contains_ref(value: PlainJson) -> bool:
    if isinstance(value, dict):
        if "$ref" in value or "$defs" in value:
            return True
        return any(_contains_ref(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False


def build_vertex_schema(
    parameters: dict[str, PlainJson], add_property_ordering: bool
) -> _SchemaResult:
    if _contains_ref(parameters):
        return TranslationError.of_unsupported(
            "schema $ref/$defs need v1's unpack_defs inlining"
        )
    converted = _convert_anyof_null(parameters, 0)
    if isinstance(converted, TranslationError) or not isinstance(converted, dict):
        return converted
    typed = _convert_types(converted, 0)
    if isinstance(typed, TranslationError) or not isinstance(typed, dict):
        return typed
    enum_fixed = _fix_enum_types(_fix_enum_empty_strings(typed))
    items_done = _process_items(enum_fixed, 0)
    if isinstance(items_done, TranslationError):
        return items_done
    objected = _add_object_type(items_done)
    filtered = _filter_fields(objected)
    if add_property_ordering and isinstance(filtered, dict):
        return _with_property_ordering(filtered, 0)
    return filtered


def _too_deep(depth: int) -> TranslationError | None:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return TranslationError.of_unsupported(
            "schema nests past v1's recursion cap; v1 raises"
        )
    return None


def _convert_anyof_members(
    anyof: list[PlainJson],
) -> list[PlainJson] | TranslationError:
    """v1 mutates the list during iteration (anyof.remove inside the for
    loop); this filter form only differs for multiple adjacent nulls."""
    empty_member: PlainJson = {"type": "object"}
    kept: list[PlainJson] = [
        (empty_member if a == {} else a)
        for a in anyof
        if not (isinstance(a, dict) and a.get("type") == "null")
    ]
    had_null = any(isinstance(a, dict) and a.get("type") == "null" for a in anyof)
    if len(kept) == 0:
        return TranslationError.of_unsupported(
            "anyOf with only null members; v1 raises ValueError"
        )
    if had_null:
        nullable: list[PlainJson] = [
            {**a, "nullable": True} if isinstance(a, dict) else a for a in kept
        ]
        return nullable
    return kept


def _convert_anyof_null(schema: PlainJson, depth: int) -> _SchemaResult:
    deep = _too_deep(depth)
    if deep is not None:
        return deep
    if not isinstance(schema, dict):
        return schema
    out: dict[str, PlainJson] = dict(schema)
    anyof = schema.get("anyOf")
    if isinstance(anyof, list):
        members = _convert_anyof_members(anyof)
        if isinstance(members, TranslationError):
            return members
        out = {**out, "anyOf": members}
    props = out.get("properties")
    if isinstance(props, dict):
        new_props: dict[str, PlainJson] = {}
        for name, value in props.items():
            converted = _convert_anyof_null(value, depth + 1)
            if isinstance(converted, TranslationError):
                return converted
            new_props = {**new_props, name: converted}
        out = {**out, "properties": new_props}
    items = out.get("items")
    if items is not None:
        converted = _convert_anyof_null(items, depth + 1)
        if isinstance(converted, TranslationError):
            return converted
        out = {**out, "items": converted}
    return out


def _split_type_array(
    schema: dict[str, PlainJson], types: list[PlainJson]
) -> dict[str, PlainJson]:
    any_of: list[PlainJson] = []
    for t in types:
        if not isinstance(t, str):
            continue
        if t == "null":
            any_of = [*any_of, {"type": "null"}]
        elif t in ("object", "array"):
            item: dict[str, PlainJson] = {"type": t}
            for field in _TYPE_SPECIFIC_FIELDS:
                if field in schema:
                    item = {**item, field: schema[field]}
            any_of = [*any_of, item]
        else:
            any_of = [*any_of, {"type": t}]
    has_container = any(t in ("object", "array") for t in types if isinstance(t, str))
    out = {
        k: v
        for k, v in schema.items()
        if k != "type" and not (has_container and k in _TYPE_SPECIFIC_FIELDS)
    }
    return {**out, "anyOf": any_of}


def _convert_types(schema: PlainJson, depth: int) -> _SchemaResult:
    deep = _too_deep(depth)
    if deep is not None:
        return deep
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    type_val = out.get("type")
    if isinstance(type_val, list) and len(type_val) > 1:
        out = _split_type_array(out, type_val)
    elif isinstance(type_val, list) and len(type_val) == 1:
        out = {**out, "type": type_val[0]}
    return _convert_types_children(out, depth)


def _convert_types_children(out: dict[str, PlainJson], depth: int) -> _SchemaResult:
    props = out.get("properties")
    if isinstance(props, dict):
        new_props: dict[str, PlainJson] = {}
        for name, value in props.items():
            converted = _convert_types(value, depth + 1)
            if isinstance(converted, TranslationError):
                return converted
            new_props = {**new_props, name: converted}
        out = {**out, "properties": new_props}
    items = out.get("items")
    if items is not None:
        converted = _convert_types(items, depth + 1)
        if isinstance(converted, TranslationError):
            return converted
        out = {**out, "items": converted}
    anyof = out.get("anyOf")
    if isinstance(anyof, list):
        members: list[PlainJson] = []
        for member in anyof:
            converted = _convert_types(member, depth + 1)
            if isinstance(converted, TranslationError):
                return converted
            members = [*members, converted]
        out = {**out, "anyOf": members}
    return out


def _map_children(
    schema: dict[str, PlainJson], transform: Callable[[PlainJson], PlainJson]
) -> dict[str, PlainJson]:
    """Apply ``transform`` to properties values and items (v1's shared
    recursion shape for the enum fixes)."""
    out: dict[str, PlainJson] = dict(schema)
    props = out.get("properties")
    if isinstance(props, dict):
        mapped: PlainJson = {name: transform(value) for name, value in props.items()}
        out = {**out, "properties": mapped}
    items = out.get("items")
    if items is not None:
        out = {**out, "items": transform(items)}
    return out


def _fix_enum_empty_strings(schema: PlainJson) -> PlainJson:
    if not isinstance(schema, dict):
        return schema
    out: dict[str, PlainJson] = dict(schema)
    enum = out.get("enum")
    if isinstance(enum, list):
        fixed: list[PlainJson] = [None if v == "" else v for v in enum]
        out = {**out, "enum": fixed}
    return _map_children(out, _fix_enum_empty_strings)


def _anyof_has_string(anyof: PlainJson) -> bool:
    if not isinstance(anyof, list):
        return False
    return any(
        isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and str(item.get("type")).lower() == "string"
        for item in anyof
    )


def _fix_enum_types(schema: PlainJson) -> PlainJson:
    if not isinstance(schema, dict):
        return schema
    out: dict[str, PlainJson] = dict(schema)
    if isinstance(out.get("enum"), list):
        schema_type = out.get("type")
        keep = (
            isinstance(schema_type, str) and schema_type.lower() == "string"
        ) or _anyof_has_string(out.get("anyOf"))
        if not keep:
            out = {k: v for k, v in out.items() if k != "enum"}
    return _map_children(out, _fix_enum_types)


def _process_items(schema: PlainJson, depth: int) -> _SchemaResult:
    deep = _too_deep(depth)
    if deep is not None:
        return deep
    if not isinstance(schema, dict):
        return schema
    out: dict[str, PlainJson] = dict(schema)
    type_val = out.get("type")
    if (
        isinstance(type_val, str)
        and type_val.lower() == "array"
        and ("items" not in out or out.get("items") == {})
    ):
        object_items: PlainJson = {"type": "object"}
        out = {**out, "items": object_items}
    new_out: dict[str, PlainJson] = {}
    for key, value in out.items():
        child = _process_items_value(value, depth)
        if isinstance(child, TranslationError):
            return child
        new_out = {**new_out, key: child}
    return new_out


def _process_items_value(value: PlainJson, depth: int) -> _SchemaResult:
    if isinstance(value, dict):
        return _process_items(value, depth + 1)
    if isinstance(value, list):
        members: list[PlainJson] = []
        for item in value:
            if isinstance(item, dict):
                child = _process_items(item, depth + 1)
                if isinstance(child, TranslationError):
                    return child
                members = [*members, child]
            else:
                members = [*members, item]
        return members
    return value


def _add_object_type(schema: PlainJson) -> PlainJson:
    if not isinstance(schema, dict):
        return schema
    out: dict[str, PlainJson] = dict(schema)
    if (
        "type" not in out
        and "anyOf" not in out
        and "oneOf" not in out
        and "allOf" not in out
    ):
        out = {**out, "type": "object"}
    props = out.get("properties")
    if props is not None and isinstance(props, dict):
        if "required" in out and out["required"] is None:
            out = {k: v for k, v in out.items() if k != "required"}
        if not props:
            out = {
                **{k: v for k, v in out.items() if k not in ("properties", "required")},
                "type": "object",
            }
        else:
            converted_props: PlainJson = {
                name: _add_object_type(value) for name, value in props.items()
            }
            out = {**out, "type": "object", "properties": converted_props}
    items = out.get("items")
    if items is not None:
        out = {**out, "items": _add_object_type(items)}
    for key in ("anyOf", "oneOf", "allOf"):
        values = out.get(key)
        if isinstance(values, list):
            converted_members: PlainJson = [
                _add_object_type(v) if isinstance(v, dict) else v for v in values
            ]
            out = {**out, key: converted_members}
    return out


def _filter_anyof_only(schema: dict[str, PlainJson]) -> dict[str, PlainJson]:
    """v1 ``_filter_anyof_fields``: when anyOf is present keep only anyOf,
    pushing title/description down into the members."""
    anyof = schema.get("anyOf")
    if not anyof:
        return schema
    title = schema.get("title")
    description = schema.get("description")
    if (
        (title or description)
        and isinstance(anyof, list)
        and all(isinstance(item, dict) for item in anyof)
    ):
        decorated: list[PlainJson] = []
        for item in anyof:
            if not isinstance(item, dict):
                continue  # the all() guard above proved dicts; narrows the type
            extra: dict[str, PlainJson] = {}
            if title:
                extra = {**extra, "title": title}
            if description:
                extra = {**extra, "description": description}
            decorated = [*decorated, {**item, **extra}]
        return {"anyOf": decorated}
    return {"anyOf": anyof}


def _filter_fields(schema: PlainJson) -> PlainJson:
    if not isinstance(schema, dict):
        return schema
    source = _filter_anyof_only(schema)
    result: dict[str, PlainJson] = {}
    for key, value in source.items():
        if key not in _VALID_SCHEMA_FIELDS:
            continue
        if key == "properties" and isinstance(value, dict):
            result = {
                **result,
                key: {name: _filter_fields(v) for name, v in value.items()},
            }
        elif key == "format":
            if value in ("enum", "date-time"):
                result = {**result, key: value}
        elif key == "items" and isinstance(value, dict):
            result = {**result, key: _filter_fields(value)}
        elif key == "anyOf" and isinstance(value, list):
            result = {**result, key: [_filter_fields(item) for item in value]}
        else:
            result = {**result, key: value}
    return result


def _with_property_ordering(schema: dict[str, PlainJson], depth: int) -> _SchemaResult:
    deep = _too_deep(depth)
    if deep is not None:
        return deep
    out: dict[str, PlainJson] = dict(schema)
    props = out.get("properties")
    if isinstance(props, dict):
        new_props: dict[str, PlainJson] = {}
        for name, value in props.items():
            if isinstance(value, dict):
                child = _with_property_ordering(value, depth + 1)
                if isinstance(child, TranslationError):
                    return child
                ordered_child: PlainJson = child if isinstance(child, dict) else value
                new_props = {**new_props, name: ordered_child}
            else:
                new_props = {**new_props, name: value}
        props_json: PlainJson = new_props
        out = {**out, "properties": props_json}
        if "propertyOrdering" not in out:
            ordering: PlainJson = list(props.keys())
            out = {**out, "propertyOrdering": ordering}
    items = out.get("items")
    if isinstance(items, dict):
        child = _with_property_ordering(items, depth + 1)
        if isinstance(child, TranslationError):
            return child
        out = {**out, "items": child}
    return out

"""Anthropic tool serialization: defs, name sanitization, schema whitelist.

Ports the exact v1 behavior for the supported surface:

- ``input_schema`` is coerced to ``type: object`` and filtered to the keys the
  Anthropic API accepts (v1 ``_map_tool_helper`` over ``AnthropicInputSchema``);
  pydantic/zod-generated extras like a top-level ``title`` are stripped.
- Tool names are sanitized to ``^[a-zA-Z0-9_-]{1,128}$`` with per-request
  collision suffixes (v1 ``_build_anthropic_tool_name_maps``). The maps are a
  deterministic pure function of the ordered tool-name list, so the response
  side recomputes the reverse map from the same ``ChatRequest`` instead of
  threading hidden state.
- ``tool_use`` ids are sanitized to the API's ``^[a-zA-Z0-9_-]+$``.
"""

from __future__ import annotations

import re
from itertools import count

from expression import Option
from expression.collections import Block

from ...ir import Body, CacheControl, PlainJson, ToolDef

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_NAME_MAX_LEN = 128

_SCHEMA_ALLOWED_KEYS = frozenset(
    {"type", "properties", "additionalProperties", "required", "$defs", "strict"}
)


def cache_json(cache: CacheControl) -> PlainJson:
    base: dict[str, PlainJson] = {"type": cache.type}
    match cache.ttl:
        case Option(tag="some", some=ttl):
            return {**base, "ttl": ttl}
        case _:
            return base


def serialize_tool(tool: ToolDef) -> Body:
    schema = _input_schema(tool.parameters.map(lambda blob: blob.value))
    base: dict[str, PlainJson] = {
        "name": tool.name,
        "input_schema": schema,
        "type": "custom",
    }
    described = (
        {**base, "description": tool.description.value}
        if tool.description.is_some()
        else base
    )
    match tool.cache:
        case Option(tag="some", some=cache):
            return {**described, "cache_control": cache_json(cache)}
        case _:
            return described


def _input_schema(parameters: Option[PlainJson]) -> PlainJson:
    raw: PlainJson
    match parameters:
        case Option(tag="some", some=value):
            raw = value
        case _:
            raw = {"type": "object", "properties": {}}
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}
    coerced = (
        raw
        if raw.get("type") == "object"
        else {**raw, "type": "object", "properties": raw.get("properties") or {}}
    )
    return {key: value for key, value in coerced.items() if key in _SCHEMA_ALLOWED_KEYS}


def dummy_tool() -> Body:
    """Fresh per call: returned bodies are mutable dicts downstream may edit."""
    return {
        "name": "dummy_tool",
        "input_schema": {"type": "object", "properties": {}},
        "type": "custom",
        "description": "This is a dummy tool call",
    }


def sanitize_tool_use_id(tool_use_id: str) -> str:
    sanitized = _INVALID_NAME_CHARS.sub("_", tool_use_id)
    return sanitized if sanitized else "tool_use_id"


def _basic_sanitize_name(name: str) -> str:
    return _INVALID_NAME_CHARS.sub("_", name)[:_NAME_MAX_LEN]


def build_name_maps(
    original_names: Block[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """(forward, reverse) maps; only rewritten names appear (v1 semantics).

    Already-valid names reserve their slot first regardless of order; invalid
    or colliding names get ``_2``-style suffixes in encounter order. Local
    accumulators are the standard build-locally pattern; nothing escapes.
    """
    valid = {name for name in original_names if _basic_sanitize_name(name) == name}
    forward: dict[str, str] = {}
    used = set(valid)
    for original in original_names:
        candidate = _basic_sanitize_name(original)
        if candidate == original or original in forward or not original:
            continue
        unique = candidate
        for n in count(2):
            if unique not in used:
                break
            suffix = f"_{n}"
            unique = f"{candidate[: _NAME_MAX_LEN - len(suffix)]}{suffix}"
        forward = {**forward, original: unique}
        used = used | {unique}
    reverse = {sanitized: original for original, sanitized in forward.items()}
    return forward, reverse


def request_name_maps(tools: Block[ToolDef]) -> tuple[dict[str, str], dict[str, str]]:
    return build_name_maps(tools.map(lambda tool: tool.name))


def response_format_tool(schema: PlainJson | None) -> Body:
    """v1 ``_create_json_tool_call_for_response_format``: note no ``type`` key."""
    if schema is None:
        input_schema: PlainJson = {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
        }
    elif isinstance(schema, dict):
        input_schema = {"type": "object", **schema}
    else:
        input_schema = {"type": "object"}
    return {"name": "json_tool_call", "input_schema": input_schema}


def filter_output_schema(schema: PlainJson) -> PlainJson:
    """Port of v1 ``filter_anthropic_output_schema``: strip constraints the
    output_format API rejects, fold them into the description, and default
    ``additionalProperties: false`` for objects."""
    if not isinstance(schema, dict):
        return schema
    labels = {
        "minItems": "minimum number of items: {}",
        "maxItems": "maximum number of items: {}",
        "minimum": "minimum value: {}",
        "maximum": "maximum value: {}",
        "exclusiveMinimum": "exclusive minimum value: {}",
        "exclusiveMaximum": "exclusive maximum value: {}",
        "minLength": "minimum length: {}",
        "maxLength": "maximum length: {}",
    }
    # Deliberately a set iterated like v1's: the note order inherits the same
    # per-process string-hash order, so the differential matches in-process.
    unsupported = {
        "maxItems",
        "minItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
    }
    notes = [labels[key].format(schema[key]) for key in unsupported if key in schema]
    description = _described(schema, notes)
    filtered = {
        key: _filter_child(key, value)
        for key, value in schema.items()
        if key not in labels and not (key == "description" and description is not None)
    }
    with_description = (
        {**filtered, "description": description}
        if description is not None
        else filtered
    )
    if with_description.get("type") == "object" and (
        "additionalProperties" not in with_description
    ):
        return {**with_description, "additionalProperties": False}
    return with_description


def _described(schema: dict[str, PlainJson], notes: list[str]) -> str | None:
    if not notes:
        return None
    note = "Note: " + ", ".join(notes) + "."
    existing = schema.get("description", "")
    if existing:
        return f"{existing} {note}"
    return note


def _filter_child(key: str, value: PlainJson) -> PlainJson:
    if key in ("properties", "$defs") and isinstance(value, dict):
        return {name: filter_output_schema(child) for name, child in value.items()}
    if key == "items" and isinstance(value, dict):
        return filter_output_schema(value)
    if key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
        return [filter_output_schema(child) for child in value]
    return value

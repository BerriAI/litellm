"""Converse tool serialization: toolSpec blocks, bedrock tool names, cachePoints.

Ports the exact v1 behavior for the supported surface:

- tool names normalize to ``[a-zA-Z][a-zA-Z0-9_-]*`` (v1
  ``make_valid_bedrock_tool_name``): prepend ``a`` when the first char is not
  a letter, replace invalid chars with ``_``. No length cap and no collision
  suffixes (unlike anthropic). The reverse map is recomputed from the request
  tools on the response side, replacing v1's process-global name cache.
- ``toolSpec.inputSchema.json`` keeps exactly type/properties/required
  (+ additionalProperties for Claude); ``required`` defaults to ``[]``
  (v1 ``BedrockToolSpec``).
- a tool's ``cache_control`` becomes a ``cachePoint`` block after its spec,
  only for type ``ephemeral``; ``ttl`` survives only on Claude >= 4.5.
"""

from __future__ import annotations

import json

from expression import Nothing, Option, Some
from expression.collections import Block

from ...errors import TranslationError
from ...ir import CacheControl, JsonSchemaSpec, PlainJson, ToolDef
from .params import is_claude_4_5_plus

_NAME_HEAD_FALLBACK = "a"


def make_valid_bedrock_name(name: str) -> str:
    if not name:
        return name
    candidate = name if name[0].isalpha() else f"{_NAME_HEAD_FALLBACK}{name}"
    return "".join(
        char if (char.isalnum() or char in ("_", "-")) else "_" for char in candidate
    )


def reverse_name_map(tools: Block[ToolDef]) -> dict[str, str]:
    """{bedrock name -> original} for names the request rewrote (last wins,
    matching v1's cache-overwrite order)."""
    pairs = (
        (make_valid_bedrock_name(_effective_name(index, tool.name)), tool.name)
        for index, tool in enumerate(tools)
    )
    return {valid: original for valid, original in pairs if valid != original}


def _effective_name(index: int, name: str) -> str:
    return name if name and name.strip() else f"litellm_unnamed_tool_{index}"


def serialize_tools(
    tools: Block[ToolDef], model: str
) -> list[PlainJson] | TranslationError:
    blocks: list[PlainJson] = []
    for index, tool in enumerate(tools):
        spec = _tool_spec(index, tool)
        if isinstance(spec, TranslationError):
            return spec
        blocks.append(spec)  # nosemgrep: translation-no-mutation
        cache_point = tool_cache_point(tool.cache, model)
        if cache_point is not None:
            blocks.append(cache_point)  # nosemgrep: translation-no-mutation
    return blocks


def _tool_spec(index: int, tool: ToolDef) -> PlainJson | TranslationError:
    parameters = tool.parameters.map(lambda blob: blob.value).default_value(None)
    schema = _input_schema_json(parameters)
    if isinstance(schema, TranslationError):
        return schema
    name = make_valid_bedrock_name(_effective_name(index, tool.name))
    description = tool.description.default_value(None)
    return {
        "toolSpec": {
            "inputSchema": {"json": schema},
            "name": name,
            "description": description if description else name,
        }
    }


def _empty_object_schema() -> PlainJson:
    return {"type": "object", "properties": {}}


def _input_schema_json(raw_parameters: PlainJson) -> PlainJson | TranslationError:
    parameters = (
        raw_parameters if raw_parameters is not None else _empty_object_schema()
    )
    if not isinstance(parameters, dict):
        return TranslationError.of_unsupported(
            "non-object tool parameters; v1 handles them"
        )
    if "$defs" in parameters or '"$ref"' in json.dumps(parameters):
        return TranslationError.of_unsupported(
            "tool schema with $defs/$ref needs v1's unpack_defs inlining"
        )
    kind = parameters.get("type")
    valid_roots = ("array", "boolean", "integer", "null", "number", "object", "string")
    schema: dict[str, PlainJson] = {
        "type": kind if kind in valid_roots else "object",
        "properties": _normalized(parameters.get("properties", {})),
        "required": parameters.get("required", []),
    }
    additional = parameters.get("additionalProperties")
    if additional is not None:
        return {**schema, "additionalProperties": _normalized(additional)}
    return schema


def _normalized(value: PlainJson) -> PlainJson:
    """v1 ``normalize_json_schema_custom_types_to_object``: nested
    ``type: "custom"`` becomes ``"object"`` (Claude Code emits it)."""
    if isinstance(value, dict):
        replaced = {key: _normalized(child) for key, child in value.items()}
        if replaced.get("type") == "custom":
            return {**replaced, "type": "object"}
        return replaced
    if isinstance(value, list):
        return [_normalized(child) for child in value]
    return value


def tool_cache_point(cache: Option[CacheControl], model: str) -> PlainJson | None:
    """v1 ``add_cache_point_tool_block``: ephemeral only; ttl on Claude >= 4.5."""
    match cache:
        case Option(tag="some", some=control):
            if control.type != "ephemeral":
                return None
            return {"cachePoint": _cache_point_json(control, model)}
        case _:
            return None


def content_cache_point(cache: Option[CacheControl]) -> PlainJson | None:
    """v1 ``_get_cache_point_block`` as the factory calls it for message
    content (no model passed, so ttl never survives) — any cache type counts."""
    if cache.is_none():
        return None
    return {"cachePoint": {"type": "default"}}


def system_cache_point(cache: Option[CacheControl], model: str) -> PlainJson | None:
    """System blocks DO pass the model, so ttl survives on Claude >= 4.5."""
    match cache:
        case Option(tag="some", some=control):
            return {"cachePoint": _cache_point_json(control, model)}
        case _:
            return None


def _cache_point_json(control: CacheControl, model: str) -> PlainJson:
    match control.ttl:
        case Option(tag="some", some=ttl):
            if ttl in ("5m", "1h") and is_claude_4_5_plus(model):
                return {"type": "default", "ttl": ttl}
            return {"type": "default"}
        case _:
            return {"type": "default"}


def response_format_tool(spec: JsonSchemaSpec) -> ToolDef | TranslationError:
    """v1 ``_create_json_tool_call_for_response_format``: the schema rides
    as-is into a ``json_tool_call`` toolSpec (description defaults to the
    name; the IR drops wire descriptions before this point)."""
    schema = spec.schema.value
    if not isinstance(schema, dict):
        return TranslationError.of_unsupported(
            "non-object response_format schema; v1 handles it"
        )
    return ToolDef(
        name="json_tool_call",
        description=Nothing,
        parameters=Some(spec.schema),
        cache=Nothing,
    )

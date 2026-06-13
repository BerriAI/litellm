"""The databricks claude tool round-trip and the thinking max-token bump.

Two databricks-only pure transforms, split out of ``serialize.py`` to keep it
under the size cap (the ollama_fold precedent).

THE CLAUDE TOOL ROUND-TRIP (``"claude" in model`` only — DB-R1). v1 runs
openai-tool -> anthropic-tool (``AnthropicConfig._map_tool_helper``) ->
databricks-tool (``convert_anthropic_tool_to_databricks_tool``). Probed
in-process at HEAD, the net effect on a function tool is:

- the FUNCTION-level ``strict`` and ``cache_control`` are DROPPED (the
  anthropic mapper reads only name/parameters/description);
- ``parameters`` is filtered to ``AnthropicInputSchema``'s allowed keys
  (``$defs``/additionalProperties/properties/required/strict/type) -> a
  ``$schema`` key inside parameters is DROPPED, ``additionalProperties`` is
  KEPT;
- absent/None parameters become ``{"type": "object", "properties": {}}``; a
  non-"object" ``type`` is coerced to "object" (+ ``properties: {}`` if
  missing);
- the description rides only when present (the empty string "" IS present, so
  it survives — repositioned AFTER parameters in the output, probed);
- the output shape is ``{"type": "function", "function": {"name",
  "parameters", ["description"]}}``.

NON-claude tools ride VERBATIM (v1's ``if "claude" not in model: return
tools``) — the shared openai_compat assembly already emits them.

The schema-key whitelist is the SAME ``AnthropicInputSchema`` symbol v1
filters against (the request gate's mirror test pins it), so a new allowed
key cannot silently drift. Legacy ``$ref``/``$defs`` schemas fall back at the
serializer (v1 inlines them via ``unpack_legacy_defs`` — unported, the
fireworks legacy-defs precedent).

THE THINKING MAX-BUMP. ``update_optional_params_with_thinking_tokens`` fires
for ALL models: when ``thinking`` is enabled WITH a ``budget_tokens`` and the
caller gave NO max_tokens/max_completion_tokens, v1 sets ``max_tokens =
budget_tokens + DEFAULT_MAX_TOKENS`` (probed: 2048 -> 6144, 1024 -> 5120 with
DEFAULT_MAX_TOKENS=4096). A caller-supplied max_tokens is NEVER adjusted
(probed: budget 1024 + max_tokens 100 -> over-budget pair sent as-is). When
the ``thinking`` dict omits ``budget_tokens`` NO bump fires (v1's
``is not None`` gate), and the budget key never rides the wire either.
"""

from __future__ import annotations

from litellm.constants import DEFAULT_MAX_TOKENS

from ...errors import TranslationError
from ...ir import (
    Block,
    JsonBlob,
    Option,
    PlainJson,
    ThinkingEnabled,
    ThinkingParam,
    ToolDef,
)

_ANTHROPIC_INPUT_SCHEMA_KEYS = frozenset(
    {"$defs", "additionalProperties", "properties", "required", "strict", "type"}
)
"""The ONE in-package mirror of ``AnthropicInputSchema.__annotations__``; the
request gate's mirror test pins it against v1's TypedDict at HEAD."""


def claude_tools(tools: Block[ToolDef]) -> list[PlainJson] | TranslationError:
    """The claude-substring tool round-trip. Returns a typed fallback when a
    schema carries ``$ref``/``$defs`` (v1 inlines via unpack_legacy_defs —
    unported)."""
    out: list[PlainJson] = []
    for tool in tools:
        converted = _claude_tool(tool)
        if isinstance(converted, TranslationError):
            return converted
        out = [*out, converted]
    return out


def _claude_tool(tool: ToolDef) -> PlainJson | TranslationError:
    parameters = _parameters(tool)
    if _carries_legacy_defs(parameters):
        return TranslationError.of_unsupported(
            "databricks claude tool with $ref/$defs in its parameters: v1 "
            "inlines them via unpack_legacy_defs before the anthropic schema "
            "filter — unported (the fireworks legacy-defs precedent)"
        )
    function: dict[str, PlainJson] = {
        "name": tool.name,
        "parameters": _filtered_schema(parameters),
    }
    match tool.description:
        case _ if tool.description.is_some():
            function = {**function, "description": tool.description.value}
        case _:
            pass
    return {"type": "function", "function": function}


def _parameters(tool: ToolDef) -> dict[str, PlainJson]:
    match tool.parameters:
        case Option(tag="some", some=JsonBlob(value=dict() as schema)):
            pass
        case _:
            return {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        coerced: dict[str, PlainJson] = {**schema, "type": "object"}
        if "properties" not in coerced:
            coerced = {**coerced, "properties": {}}
        return coerced
    return dict(schema)


def _filtered_schema(schema: dict[str, PlainJson]) -> dict[str, PlainJson]:
    return {
        key: value
        for key, value in schema.items()
        if key in _ANTHROPIC_INPUT_SCHEMA_KEYS
    }


def _carries_legacy_defs(schema: dict[str, PlainJson]) -> bool:
    return _scan_legacy_defs(schema, 0)


def _scan_legacy_defs(value: PlainJson, depth: int) -> bool:
    if depth > 100:
        return True
    if isinstance(value, dict):
        if "$ref" in value or "$defs" in value or "definitions" in value:
            return True
        return any(_scan_legacy_defs(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return any(_scan_legacy_defs(item, depth + 1) for item in value)
    return False


def thinking_max_tokens(
    thinking: ThinkingParam, caller_max_tokens: int | None
) -> int | None:
    """v1's max-token bump: ``budget_tokens + DEFAULT_MAX_TOKENS`` when thinking
    is enabled WITH a budget and the caller gave NO max_tokens; the caller's
    value is never adjusted, and a budget-less thinking dict triggers NO bump
    (v1's ``is not None`` gate). Returns the bumped value, or ``None`` when no
    bump applies."""
    if caller_max_tokens is not None:
        return None
    match thinking.tag:
        case "enabled":
            return _enabled_bump(thinking.enabled)
        case "disabled" | "adaptive":
            return None
    return None


def _enabled_bump(enabled: ThinkingEnabled) -> int | None:
    match enabled.budget_tokens:
        case Option(tag="some", some=budget):
            return budget + DEFAULT_MAX_TOKENS
        case _:
            return None

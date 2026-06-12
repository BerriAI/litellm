"""IR tool definitions / tool_choice -> generateContent tools + toolConfig.

Mirrors v1's ``_map_function`` for the surface the IR can express: function
tools only (hosted googleSearch/codeExecution/computerUse tools never parse
into ``ToolDef``), schemas through the same strip + ``_build_vertex_schema``
pipeline, names NEVER sanitized (gemini takes them verbatim), tool-level
``cache_control`` dropped exactly like v1's FunctionDeclaration rebuild.
The emitted key is v1's snake ``function_declarations``.
"""

from __future__ import annotations

from expression import Option
from expression.collections import Block
from typing_extensions import assert_never

from ...errors import TranslationError
from ...ir import PlainJson, ToolChoice, ToolDef
from .schema import build_vertex_schema, strip_additional_properties, strip_strict

_ToolsResult = list[PlainJson] | TranslationError


def serialize_tools(tools: Block[ToolDef]) -> _ToolsResult:
    declarations: list[PlainJson] = []
    for tool in tools:
        declaration = _declaration(tool)
        if isinstance(declaration, TranslationError):
            return declaration
        declarations = [*declarations, declaration]
    if not declarations:
        return []
    return [{"function_declarations": declarations}]


def _declaration(tool: ToolDef) -> PlainJson | TranslationError:
    declaration: dict[str, PlainJson] = {"name": tool.name}
    match tool.description:
        case Option(tag="some", some=description):
            declaration = {**declaration, "description": description}
        case _:
            pass
    match tool.parameters:
        case Option(tag="some", some=blob):
            # the strip passes reconstruct every node; no deepcopy needed
            raw = blob.value
            if not isinstance(raw, dict):
                return TranslationError.of_unsupported(
                    "non-object tool parameters; v1 forwards them unmodified"
                )
            stripped = strip_strict(strip_additional_properties(raw))
            if not isinstance(stripped, dict):
                return TranslationError.of_unsupported(
                    "tool parameters did not survive schema stripping"
                )
            built = build_vertex_schema(stripped, add_property_ordering=False)
            if isinstance(built, TranslationError):
                return built
            return {**declaration, "parameters": built}
        case _:
            pass
    return declaration


def tool_config(choice: ToolChoice) -> PlainJson:
    """v1 ``map_tool_choice_values`` -> ``toolConfig`` body value."""
    match choice.tag:
        case "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        case "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        case "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        case "specific":
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowed_function_names": [choice.specific],
                }
            }
    assert_never(choice.tag)

from __future__ import annotations

import copy
import functools
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, TypeAlias, TypeVar

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs  # noqa: TID251  # narrows JSON in place without copying it

from litellm.proxy.mcp_server.utils import JSONLeafPath, json_string_leaves

DescriptionPath: TypeAlias = JSONLeafPath

_TOOLS: Final = TypeAdapter(list[JsonValue])
_Tool = TypeVar("_Tool")  # rebind-ok: a TypeVar is declared by plain assignment

_SCHEMA_VALUE_KEYWORDS: Final = frozenset({"default", "examples", "enum", "const"})
_NAMED_SUBSCHEMA_MAPS: Final = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"}
)


def _inside_schema_value(path: DescriptionPath) -> bool:
    return any(
        key in _SCHEMA_VALUE_KEYWORDS and (index == 0 or path[index - 1] not in _NAMED_SUBSCHEMA_MAPS)
        for index, key in enumerate(path)
    )


def tool_description_entries(tools: Sequence[JsonValue]) -> tuple[tuple[DescriptionPath, str], ...] | None:
    """Every ``description`` string in ``tools``, or None when they are nested too deeply to read."""
    leaves: Final = json_string_leaves(list(tools))
    if leaves is None:
        return None
    return tuple(
        (path, text)
        for path, text in leaves
        if path and path[-1] == "description" and text.strip() and not _inside_schema_value(path[:-1])
    )


def _is_object(
    value: object,
) -> TypeIs[dict[str, object]]:  # guard-ok: JSON keys are str  # mutable-ok: rewritten in place
    return isinstance(value, dict)


def _is_array(
    value: object,
) -> TypeIs[list[object]]:  # guard-ok: isinstance narrowing  # mutable-ok: rewritten in place
    return isinstance(value, list)


def _child(node: object, key: str | int) -> object:
    if _is_object(node) and isinstance(key, str):
        return node.get(key)
    if _is_array(node) and isinstance(key, int) and 0 <= key < len(node):
        return node[key]
    return None


def with_replaced_descriptions(
    tools: Sequence[_Tool], replacements: Mapping[DescriptionPath, str]
) -> list[_Tool]:  # mutable-ok: GenericGuardrailAPIInputs.tools is declared as a list
    rewritten: Final = copy.deepcopy(list(tools))
    for path, text in replacements.items():
        holder = functools.reduce(_child, path[:-1], rewritten)
        if _is_object(holder) and isinstance(key := path[-1], str):
            holder[key] = text
    return rewritten


def _unscannable(guardrail_name: str | None) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "tool_descriptions_unscannable",
            "message": (
                f"Guardrail '{guardrail_name}' has scan_tool_descriptions_as_text on, but tools[] is not "
                "plain JSON or is nested too deeply to read, so the request was rejected rather than sent "
                "with unscanned tool descriptions"
            ),
        },
    )


async def scanned_tool_descriptions(
    tools: Sequence[_Tool],
    scan_text: Callable[[str], Awaitable[str]],
    guardrail_name: str | None,
) -> list[_Tool] | None:  # mutable-ok: GenericGuardrailAPIInputs.tools is declared as a list
    """Run ``scan_text`` over every ``description`` string in ``tools``, skipping schema values like ``default``.

    Returns the tools with changed descriptions written back at their original paths,
    or None when nothing changed. ``scan_text`` blocks by raising.
    """
    try:
        entries: Final = tool_description_entries(_TOOLS.validate_python(tools))
    except ValidationError as error:
        raise _unscannable(guardrail_name) from error
    if entries is None:
        raise _unscannable(guardrail_name)
    replacements: Final[dict[DescriptionPath, str]] = {}  # mutable-ok: filled one scan at a time
    for path, text in entries:
        scanned = await scan_text(text)
        if scanned != text:
            replacements[path] = scanned
    return with_replaced_descriptions(tools, replacements) if replacements else None

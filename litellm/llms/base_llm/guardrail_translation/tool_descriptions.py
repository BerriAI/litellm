from __future__ import annotations

import copy
import functools
import itertools
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Final, TypeAlias, TypeVar

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs  # noqa: TID251  # narrows JSON in place without copying it

DescriptionPath: TypeAlias = tuple[str | int, ...]
_Located: TypeAlias = tuple[DescriptionPath, JsonValue]

_TOOLS: Final = TypeAdapter(list[JsonValue])
_Tool = TypeVar("_Tool")  # rebind-ok: a TypeVar is declared by plain assignment

_TOOL_SCHEMA_KEYS: Final = ("parameters", "input_schema")
_SUBSCHEMA_KEYWORDS: Final = (
    "items",
    "additionalItems",
    "unevaluatedItems",
    "contains",
    "additionalProperties",
    "unevaluatedProperties",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
)
_SUBSCHEMA_LIST_KEYWORDS: Final = ("items", "prefixItems", "anyOf", "oneOf", "allOf")
_SUBSCHEMA_MAP_KEYWORDS: Final = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
)


def tool_description_entries(tools: Sequence[JsonValue]) -> tuple[tuple[DescriptionPath, str], ...]:
    holders: Final = tuple(
        itertools.chain.from_iterable(_tool_holders(tool, (index,)) for index, tool in enumerate(tools))
    )
    return tuple(
        itertools.chain(
            itertools.chain.from_iterable(_own_description(holder, path) for path, holder in holders),
            itertools.chain.from_iterable(_own_description(schema, path) for path, schema in _schemas(holders)),
        )
    )


def _tool_holders(tool: JsonValue, path: DescriptionPath) -> Iterator[tuple[DescriptionPath, Mapping[str, JsonValue]]]:
    if not isinstance(tool, dict):
        return
    yield path, tool
    if isinstance(function := tool.get("function"), dict):
        yield (*path, "function"), function


def _schemas(
    holders: Sequence[tuple[DescriptionPath, Mapping[str, JsonValue]]],
) -> Iterator[tuple[DescriptionPath, Mapping[str, JsonValue]]]:
    roots: Final = itertools.chain.from_iterable(
        (((*path, key), holder.get(key)) for key in _TOOL_SCHEMA_KEYS) for path, holder in holders
    )
    pending: Final[list[_Located]] = list(roots)  # mutable-ok: work stack keeps the schema walk non-recursive
    while pending:
        path, schema = pending.pop()
        if isinstance(schema, dict):
            yield path, schema
            pending.extend(_subschemas(schema, path))


def _subschemas(schema: Mapping[str, JsonValue], path: DescriptionPath) -> Iterator[tuple[DescriptionPath, JsonValue]]:
    for key in _SUBSCHEMA_KEYWORDS:
        yield (*path, key), schema.get(key)
    for key in _SUBSCHEMA_LIST_KEYWORDS:
        if isinstance(children := schema.get(key), list):
            for index, child in enumerate(children):
                yield (*path, key, index), child
    for key in _SUBSCHEMA_MAP_KEYWORDS:
        if isinstance(named := schema.get(key), dict):
            for name, child in named.items():
                yield (*path, key, name), child


def _own_description(node: Mapping[str, JsonValue], path: DescriptionPath) -> Iterator[tuple[DescriptionPath, str]]:
    description: Final = node.get("description")
    if isinstance(description, str) and description.strip():
        yield (*path, "description"), description


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


async def scanned_tool_descriptions(
    tools: Sequence[_Tool],
    scan_text: Callable[[str], Awaitable[str]],
    guardrail_name: str | None,
) -> list[_Tool] | None:  # mutable-ok: GenericGuardrailAPIInputs.tools is declared as a list
    """Run ``scan_text`` over every tool and schema ``description`` in OpenAI-format ``tools``.

    Returns the tools with changed descriptions written back at their original paths,
    or None when nothing changed. ``scan_text`` blocks by raising.
    """
    try:
        validated: Final = _TOOLS.validate_python(tools)
    except ValidationError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tool_descriptions_unscannable",
                "message": (
                    f"Guardrail '{guardrail_name}' has scan_tool_descriptions_as_text on, but tools[] is not "
                    "plain JSON or is nested too deeply to read, so the request was rejected rather than sent "
                    "with unscanned tool descriptions"
                ),
            },
        ) from error
    replacements: Final[dict[DescriptionPath, str]] = {}  # mutable-ok: filled one scan at a time
    for path, text in tool_description_entries(validated):
        scanned = await scan_text(text)
        if scanned != text:
            replacements[path] = scanned
    return with_replaced_descriptions(tools, replacements) if replacements else None

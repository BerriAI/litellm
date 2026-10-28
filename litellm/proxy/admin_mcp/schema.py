import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Final

from pydantic import JsonValue

from ._types import JSON_ARRAY, JsonObject, object_value, string_value

_SINGLE_SCHEMAS: Final = frozenset(
    {
        "additionalProperties",
        "unevaluatedProperties",
        "items",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "additionalItems",
        "contentSchema",
    }
)
_SCHEMA_MAPS: Final = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_ARRAYS: Final = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})


class SchemaError(ValueError):
    pass


def _pointer_step(value: JsonObject, part: str) -> JsonObject:
    return object_value(value[part.replace("~1", "/").replace("~0", "~")])


def _reference_target(spec: JsonObject, reference: str) -> JsonObject:
    try:
        return reduce(_pointer_step, reference[2:].split("/"), spec)
    except (KeyError, ValueError, TypeError):
        raise SchemaError("Missing or invalid OpenAPI reference") from None


def resolve(spec: JsonObject, value: JsonObject) -> JsonObject:
    current: JsonObject = value  # rebind-ok: iterative reference resolution avoids Python stack growth
    seen: Final[set[str]] = set()  # mutable-ok: local visited set bounds reference-chain traversal
    while "$ref" in current:
        reference = current["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/components/") or reference in seen:
            raise SchemaError("Unsupported OpenAPI reference")
        seen.add(reference)
        target = _reference_target(spec, reference)
        current = {**target, **{key: child for key, child in current.items() if key != "$ref"}}
    return current


def _parameters(spec: JsonObject, path_item: JsonObject, operation: JsonObject) -> Mapping[tuple[str, str], JsonObject]:
    raw: Final = (
        *JSON_ARRAY.validate_python(path_item.get("parameters", [])),
        *JSON_ARRAY.validate_python(operation.get("parameters", [])),
    )
    resolved: Final = tuple(resolve(spec, object_value(value)) for value in raw)
    return {
        (string_value(param["in"]), string_value(param["name"])): param
        for param in resolved
        if param.get("in") in ("path", "query")
    }


def _parameter_group(location: str, parameters: Mapping[tuple[str, str], JsonObject]) -> JsonValue:
    selected: Final = {name: param for (group, name), param in parameters.items() if group == location}
    required: Final[JsonValue] = [
        name for name, param in selected.items() if location == "path" or param.get("required")
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: param.get("schema", {"type": "string"}) for name, param in selected.items()},
        **({"required": required} if required else {}),
    }


def _body_schema(spec: JsonObject, operation: JsonObject) -> tuple[JsonValue, bool] | None:
    if "requestBody" not in operation:
        return None
    body: Final = resolve(spec, object_value(operation["requestBody"]))
    media: Final = object_value(body.get("content", {})).get("application/json")
    if not media:
        raise SchemaError("Administrative tools require JSON request bodies")
    return object_value(media).get("schema", {"type": "object"}), bool(body.get("required"))


def _schema_children(value: JsonValue) -> Iterable[JsonValue]:
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        if key in _SINGLE_SCHEMAS:
            yield child
        elif key in _SCHEMA_MAPS and isinstance(child, dict):
            yield from child.values()
        elif key in _SCHEMA_ARRAYS and isinstance(child, list):
            yield from child


def _rewritten_keyword(key: str, value: JsonValue, rewritten: Mapping[int, JsonValue]) -> JsonValue:
    if key in _SINGLE_SCHEMAS:
        return rewritten[id(value)]
    if key in _SCHEMA_MAPS and isinstance(value, dict):
        return {name: rewritten[id(child)] for name, child in value.items()}
    if key in _SCHEMA_ARRAYS and isinstance(value, list):
        return [rewritten[id(child)] for child in value]
    return value


def _schema_reference(value: JsonObject) -> str | None:
    if any(key in value for key in ("$dynamicRef", "$recursiveRef", "$id")):
        raise SchemaError("Schema reference scopes and dynamic references are unsupported")
    if "$ref" not in value:
        return None
    reference: Final = value["$ref"]
    if not isinstance(reference, str) or not reference.startswith("#/components/schemas/"):
        raise SchemaError("Only local schema references are supported")
    pointer: Final = reference.removeprefix("#/components/schemas/")
    if "/" in pointer or re.search(r"~(?![01])", pointer):
        raise SchemaError("Only component schema references are supported")
    return pointer


@dataclass(frozen=True, slots=True)
class Rewritten:
    value: JsonValue
    references: frozenset[str]


def _rewrite(value: JsonValue) -> Rewritten:
    pending: Final[list[tuple[JsonValue, bool]]] = [(value, False)]  # mutable-ok: iterative postorder
    rewritten: Final[dict[int, JsonValue]] = {}  # mutable-ok: local results let each schema node be assembled once
    references: Final[set[str]] = set()  # mutable-ok: local reference collection avoids copying a set per node
    while pending:
        node, ready = pending.pop()
        if id(node) in rewritten:
            continue
        if not isinstance(node, dict):
            rewritten[id(node)] = node
            continue
        if not ready:
            pending.append((node, True))
            pending.extend((child, False) for child in _schema_children(node))
            continue
        reference = _schema_reference(node)
        if reference is not None:
            references.add(reference)
        result = {
            **{
                key: _rewritten_keyword(key, child, rewritten)
                for key, child in node.items()
                if key not in {"$ref", "nullable"}
            },
            **({"$ref": "#/$defs/" + reference} if reference is not None else {}),
        }
        rewritten[id(node)] = {"anyOf": [result, {"type": "null"}]} if node.get("nullable") else result
    return Rewritten(rewritten[id(value)], frozenset(references))


def _definitions(spec: JsonObject, references: frozenset[str]) -> JsonObject:
    pending: Final[list[str]] = list(references)  # mutable-ok: worklist resolves each referenced definition once
    definitions: Final[dict[str, JsonValue]] = {}  # mutable-ok: visited definitions terminate cyclic schema graphs
    while pending:
        pointer = pending.pop()
        name = pointer.replace("~1", "/").replace("~0", "~")
        if name in definitions:
            continue
        item = _rewrite(dict(_reference_target(spec, "#/components/schemas/" + pointer)))
        definitions[name] = item.value
        pending.extend(item.references)
    return definitions


def _references(value: JsonValue) -> Iterable[str]:
    pending: Final[list[JsonValue]] = [value]  # mutable-ok: iterative schema walk
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            yield reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
        pending.extend(_schema_children(node))


def _reachable(value: JsonValue, definitions: JsonObject) -> frozenset[str]:
    pending: Final[list[str]] = list(_references(value))  # mutable-ok: worklist visits the reachable definition graph
    seen: Final[set[str]] = set()  # mutable-ok: local visited set terminates cycles without recursive calls
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(_references(definitions[name]))
    return frozenset(seen)


def tool_schema(spec: JsonObject, path_item: JsonObject, operation: JsonObject) -> JsonObject:
    parameters: Final = _parameters(spec, path_item, operation)
    locations: Final = frozenset(location for location, _ in parameters)
    body: Final = _body_schema(spec, operation)
    groups: Final = {
        **{location: _parameter_group(location, parameters) for location in locations},
        **({"body": body[0]} if body is not None else {}),
    }
    required: Final = [name for name, group in groups.items() if name != "body" and object_value(group).get("required")]
    required_groups: Final[JsonValue] = [*required, *(["body"] if body is not None and body[1] else [])]
    root: Final = _rewrite(
        {
            "type": "object",
            "properties": groups,
            "additionalProperties": False,
            **({"required": required_groups} if required_groups else {}),
        }
    )
    definitions: Final = _definitions(spec, root.references)
    result: Final = object_value(root.value)
    properties: Final = object_value(result["properties"])
    body_value: Final = properties.get("body", {})
    expanded: Final[JsonObject] = (
        {
            **result,
            "properties": {
                **properties,
                "body": definitions[
                    string_value(body_value["$ref"]).removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
                ],
            },
        }
        if isinstance(body_value, dict) and set(body_value) == {"$ref"}
        else dict(result)
    )
    reachable: Final = _reachable(dict(expanded), definitions)
    retained: Final = {name: value for name, value in definitions.items() if name in reachable}
    return {**expanded, **({"$defs": retained} if retained else {})}


def discovery_schema(schema: JsonObject, name: str) -> JsonObject:
    properties: Final = object_value(schema["properties"])
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            group: value
            if group == "response"
            else {
                "description": f"Before calling {name}, use describe_admin_tool with name={name} for the complete {group} "
                "schema. All original fields remain supported and are validated on execution."
            }
            for group, value in properties.items()
        },
        **({"required": schema["required"]} if schema.get("required") else {}),
    }

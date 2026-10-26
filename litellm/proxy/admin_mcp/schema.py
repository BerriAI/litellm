from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue

from ._types import JSON_ARRAY, JsonObject, object_value, string_value


class SchemaError(ValueError):
    pass


def _reference_target(spec: JsonObject, reference: str) -> JsonObject:
    def descend(value: JsonObject, parts: tuple[str, ...]) -> JsonObject:
        if not parts:
            return value
        return descend(object_value(value[parts[0].replace("~1", "/").replace("~0", "~")]), parts[1:])

    try:
        return descend(spec, tuple(reference[2:].split("/")))
    except (KeyError, ValueError, TypeError):
        raise SchemaError("Missing or invalid OpenAPI reference") from None


def resolve(spec: JsonObject, value: JsonObject, seen: frozenset[str] = frozenset()) -> JsonObject:
    if "$ref" not in value:
        return value
    reference: Final = value["$ref"]
    if not isinstance(reference, str) or not reference.startswith("#/components/") or reference in seen:
        raise SchemaError("Unsupported OpenAPI reference")
    target: Final = _reference_target(spec, reference)
    return resolve(
        spec, {**target, **{key: child for key, child in value.items() if key != "$ref"}}, seen | {reference}
    )


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


@dataclass(frozen=True, slots=True)
class Rewritten:
    value: JsonValue
    references: frozenset[str] = frozenset()


def _rewrite(value: JsonValue) -> Rewritten:
    if isinstance(value, list):
        children: Final = tuple(_rewrite(child) for child in value)
        return Rewritten([child.value for child in children], frozenset[str]().union(*(c.references for c in children)))
    if not isinstance(value, dict):
        return Rewritten(value)
    if "$ref" in value and (
        not isinstance(value["$ref"], str) or not value["$ref"].startswith("#/components/schemas/")
    ):
        raise SchemaError("Only local schema references are supported")
    reference: Final = string_value(value["$ref"]).removeprefix("#/components/schemas/") if "$ref" in value else None
    children_by_key: Final = {key: _rewrite(child) for key, child in value.items() if key not in {"$ref", "nullable"}}
    result: Final = {
        **{key: child.value for key, child in children_by_key.items()},
        **({"$ref": "#/$defs/" + reference} if reference is not None else {}),
    }
    references: Final = frozenset[str]().union(*(child.references for child in children_by_key.values()))
    return Rewritten(
        {"anyOf": [result, {"type": "null"}]} if value.get("nullable") else result,
        references | ({reference} if reference is not None else frozenset[str]()),
    )


def _definitions(spec: JsonObject, references: frozenset[str], known: JsonObject) -> JsonObject:
    missing: Final = references - known.keys()
    if not missing:
        return known
    rewritten: Final = {
        pointer: _rewrite(dict(_reference_target(spec, "#/components/schemas/" + pointer))) for pointer in missing
    }
    definitions: Final = {
        **known,
        **{pointer.replace("~1", "/").replace("~0", "~"): item.value for pointer, item in rewritten.items()},
    }
    next_refs: Final = frozenset[str]().union(*(item.references for item in rewritten.values()))
    encoded_known: Final = {name.replace("~", "~0").replace("/", "~1"): value for name, value in definitions.items()}
    unresolved: Final = next_refs - encoded_known.keys()
    return _definitions(spec, unresolved, definitions) if unresolved else definitions


def _references(value: JsonValue) -> Iterable[str]:
    if isinstance(value, list):
        for child in value:
            yield from _references(child)
    elif isinstance(value, dict):
        reference: Final = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            yield reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
        for child in value.values():
            yield from _references(child)


def _reachable(value: JsonValue, definitions: JsonObject, seen: frozenset[str] = frozenset()) -> frozenset[str]:
    names: Final = frozenset(_references(value)) - seen
    if not names:
        return seen
    children: Final = [definitions[name] for name in names]
    return _reachable(children, definitions, seen | names)


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
    definitions: Final = _definitions(spec, root.references, {})
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

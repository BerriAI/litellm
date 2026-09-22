import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias
from urllib.parse import quote

from jsonschema import Draft202012Validator
from pydantic import JsonValue
from referencing import Registry
from referencing.exceptions import Unresolvable

from litellm.proxy.management_endpoints.liteask.redaction import sanitize

_MAX_SCHEMA_BYTES: Final = 120_000
_MAX_ARGUMENT_BYTES: Final = 32_000
_MAX_REFERENCES: Final = 128
_MAX_DEPTH: Final = 32
_MAX_QUERY_ITEMS: Final = 100
SchemaValue: TypeAlias = str | int | float | bool | None | Mapping[str, "SchemaValue"] | Sequence["SchemaValue"]


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    title: str
    method: Literal["GET", "POST"]
    path: str
    mutation: bool


OPERATIONS: Final = (
    Operation("keys_list", "List virtual keys", "GET", "/key/list", False),
    Operation("key_info", "View a virtual key", "GET", "/key/info", False),
    Operation("key_create", "Create a virtual key", "POST", "/key/generate", True),
    Operation("key_update", "Update a virtual key", "POST", "/key/update", True),
    Operation("key_delete", "Delete virtual keys", "POST", "/key/delete", True),
    Operation("key_block", "Block a virtual key", "POST", "/key/block", True),
    Operation("key_unblock", "Unblock a virtual key", "POST", "/key/unblock", True),
    Operation("teams_list", "List teams", "GET", "/team/list", False),
    Operation("team_info", "View a team", "GET", "/team/info", False),
    Operation("team_create", "Create a team", "POST", "/team/new", True),
    Operation("team_update", "Update a team", "POST", "/team/update", True),
    Operation("team_delete", "Delete teams", "POST", "/team/delete", True),
    Operation("team_member_add", "Add a team member", "POST", "/team/member_add", True),
    Operation("team_member_update", "Update a team member", "POST", "/team/member_update", True),
    Operation("team_member_delete", "Remove a team member", "POST", "/team/member_delete", True),
    Operation("users_list", "List users", "GET", "/user/list", False),
    Operation("user_info", "View a user", "GET", "/user/info", False),
    Operation("user_create", "Create a user", "POST", "/user/new", True),
    Operation("user_update", "Update a user", "POST", "/user/update", True),
    Operation("user_delete", "Delete users", "POST", "/user/delete", True),
    Operation("budgets_list", "List budgets", "GET", "/budget/list", False),
    Operation("budget_info", "View budgets", "POST", "/budget/info", False),
    Operation("budget_create", "Create a budget", "POST", "/budget/new", True),
    Operation("budget_update", "Update a budget", "POST", "/budget/update", True),
    Operation("budget_delete", "Delete a budget", "POST", "/budget/delete", True),
    Operation("spend_report", "View spend by date and group", "GET", "/global/spend/report", False),
    Operation("team_spend_report", "View team spend", "GET", "/team/spend/report", False),
    Operation("key_spend_report", "View virtual key spend", "GET", "/key/spend/report", False),
    Operation("request_logs", "List request and spend logs", "GET", "/spend/logs/v2", False),
    Operation("request_log_detail", "View a request log", "GET", "/spend/logs/ui/{request_id}", False),
)


@dataclass(frozen=True, slots=True)
class Tool:
    operation: Operation
    parameters: dict[str, JsonValue]  # mutable-ok: JSON Schema consumed by jsonschema and model tool serialization


@dataclass(frozen=True, slots=True)
class CatalogError:
    message: str


@dataclass(frozen=True, slots=True)
class ArgumentsError:
    message: str


@dataclass(frozen=True, slots=True)
class ToolRequest:
    path: str
    query: tuple[tuple[str, str], ...]
    body: JsonValue


class _ArgumentValidator(Protocol):
    def is_valid(self, instance: JsonValue) -> bool: ...


def _matches_schema(validator: _ArgumentValidator, args: JsonValue) -> bool:
    try:
        return validator.is_valid(args)
    except (RecursionError, Unresolvable, re.error):
        return False


def _object(value: SchemaValue) -> Mapping[str, SchemaValue]:
    return value if isinstance(value, Mapping) else MappingProxyType({})


def _pointer(value: SchemaValue, segments: tuple[str, ...]) -> SchemaValue | CatalogError:
    if not segments:
        return value
    token: Final = segments[0].replace("~1", "/").replace("~0", "~")
    if not isinstance(value, Mapping) or token not in value:
        return CatalogError("An OpenAPI component reference is missing")
    return _pointer(value[token], segments[1:])


def _local_target(spec: SchemaValue, ref: str) -> SchemaValue | CatalogError:
    if not ref.startswith("#/components/"):
        return CatalogError("Only local OpenAPI component references are supported")
    segments: Final = tuple(ref[2:].split("/"))
    return (
        CatalogError("OpenAPI reference exceeds the depth limit")
        if len(segments) > _MAX_DEPTH
        else _pointer(spec, segments)
    )


def _resolve_object(spec: SchemaValue, value: SchemaValue, depth: int = 0) -> Mapping[str, SchemaValue] | CatalogError:
    if depth > _MAX_DEPTH:
        return CatalogError("OpenAPI references exceed the depth limit")
    source: Final = _object(value)
    ref: Final = source.get("$ref")
    if not isinstance(ref, str):
        return source
    target: Final = _local_target(spec, ref)
    if isinstance(target, CatalogError):
        return target
    resolved: Final = _resolve_object(spec, target, depth + 1)
    if isinstance(resolved, CatalogError):
        return resolved
    return MappingProxyType(
        {**resolved, **MappingProxyType({key: item for key, item in source.items() if key != "$ref"})}
    )


def _sequence(value: SchemaValue) -> Sequence[SchemaValue]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _references(value: SchemaValue, depth: int = 0) -> frozenset[str] | CatalogError:
    if depth > _MAX_DEPTH:
        return CatalogError("OpenAPI schema exceeds the depth limit")
    ref: Final = _object(value).get("$ref")
    if ref is not None and (not isinstance(ref, str) or not ref.startswith("#/components/")):
        return CatalogError("Only local OpenAPI component references are supported")
    children: Final = tuple(
        _references(item, depth + 1) for item in (value.values() if isinstance(value, Mapping) else _sequence(value))
    )
    own: Final[frozenset[str]] = frozenset((ref,)) if isinstance(ref, str) else frozenset()
    error: Final = next((item for item in children if isinstance(item, CatalogError)), None)
    if error is not None:
        return error
    return own.union(*(item for item in children if isinstance(item, frozenset)))


def _definition_name(ref: str) -> str:
    return "schema_" + hashlib.sha256(ref.encode()).hexdigest()[:24]


def _rewrite_refs(value: SchemaValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {  # mutable-ok: final JSON Schema wire representation
            key: "#/$defs/" + _definition_name(item) if key == "$ref" and isinstance(item, str) else _rewrite_refs(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_rewrite_refs(item) for item in value]  # mutable-ok: JSON Schema arrays must be JSON arrays
    return value


def _definitions(
    spec: SchemaValue, refs: frozenset[str], known: Mapping[str, SchemaValue]
) -> Mapping[str, SchemaValue] | CatalogError:
    if not refs:
        return known
    if len(known) + len(refs) > _MAX_REFERENCES:
        return CatalogError("OpenAPI schema exceeds the reference limit")
    targets: Final = tuple((ref, _local_target(spec, ref)) for ref in sorted(refs))
    error: Final = next((value for _, value in targets if isinstance(value, CatalogError)), None)
    if error is not None:
        return error
    additions: Final = MappingProxyType({ref: value for ref, value in targets if not isinstance(value, CatalogError)})
    nested: Final = _references(additions)
    if isinstance(nested, CatalogError):
        return nested
    collected: Final = MappingProxyType({**known, **additions})
    if len(json.dumps(_rewrite_refs(collected)).encode()) > _MAX_SCHEMA_BYTES:
        return CatalogError("OpenAPI schema exceeds the size limit")
    return _definitions(spec, nested.difference(collected), collected)


def _with_definitions(
    spec: SchemaValue, schema: Mapping[str, SchemaValue]
) -> dict[str, JsonValue] | CatalogError:  # mutable-ok: final JSON Schema wire representation
    refs: Final = _references(schema)
    if isinstance(refs, CatalogError):
        return refs
    definitions: Final = _definitions(spec, refs, MappingProxyType({}))
    if isinstance(definitions, CatalogError):
        return definitions
    complete: Final = (
        MappingProxyType(
            {
                **schema,
                "$defs": MappingProxyType({_definition_name(ref): value for ref, value in definitions.items()}),
            }
        )
        if definitions
        else schema
    )
    rewritten: Final = _rewrite_refs(complete)
    if not isinstance(rewritten, dict):
        return CatalogError("The tool schema must be an object")
    return rewritten


def _parameter_group(parameters: tuple[Mapping[str, SchemaValue], ...], location: str) -> Mapping[str, SchemaValue]:
    selected: Final = tuple(item for item in parameters if item.get("in") == location)
    return MappingProxyType(
        {
            "type": "object",
            "properties": MappingProxyType(
                {
                    str(item["name"]): MappingProxyType(
                        {
                            **_object(item.get("schema")),
                            **(
                                MappingProxyType({"description": item["description"]})
                                if "description" in item
                                else MappingProxyType({})
                            ),
                        }
                    )
                    for item in selected
                }
            ),
            "required": tuple(str(item["name"]) for item in selected if item.get("required") is True),
            "additionalProperties": False,
        }
    )


def _tool(spec: SchemaValue, operation: Operation, path_item: Mapping[str, SchemaValue]) -> Tool | CatalogError:
    source: Final = _object(path_item.get(operation.method.lower()))
    raw_parameters: Final = tuple(
        item for container in (path_item, source) for item in _sequence(container.get("parameters"))
    )
    resolved: Final = tuple(_resolve_object(spec, item) for item in raw_parameters)
    error: Final = next((item for item in resolved if isinstance(item, CatalogError)), None)
    if error is not None:
        return error
    parameters: Final = tuple(
        MappingProxyType(
            {
                (str(item.get("in")), str(item.get("name"))): item
                for item in resolved
                if not isinstance(item, CatalogError) and isinstance(item.get("name"), str)
            }
        ).values()
    )
    groups: Final = MappingProxyType(
        {
            location: _parameter_group(parameters, location)
            for location in ("path", "query")
            if any(item.get("in") == location for item in parameters)
        }
    )
    body: Final = _resolve_object(spec, source.get("requestBody"))
    if isinstance(body, CatalogError):
        return body
    body_schema: Final = _object(_object(body.get("content")).get("application/json")).get("schema")
    if body and not isinstance(body_schema, (Mapping, bool)):
        return CatalogError("Only JSON request bodies are supported")
    schema: Final[Mapping[str, SchemaValue]] = MappingProxyType(
        {
            "type": "object",
            "properties": MappingProxyType(
                {**groups, **(MappingProxyType({"body": body_schema}) if body else MappingProxyType({}))}
            ),
            "required": (
                *(location for location, group in groups.items() if group.get("required")),
                *(("body",) if body.get("required") is True else ()),
            ),
            "additionalProperties": False,
        }
    )
    complete: Final = _with_definitions(spec, schema)
    return complete if isinstance(complete, CatalogError) else Tool(operation, complete)


def build_catalog(spec: JsonValue) -> tuple[Tool, ...] | CatalogError:
    paths: Final = _object(_object(spec).get("paths"))
    if not paths:
        return CatalogError("The gateway OpenAPI schema has no paths")
    built: Final = tuple(
        _tool(spec, operation, _object(paths[operation.path]))
        for operation in OPERATIONS
        if operation.path in paths and operation.method.lower() in _object(paths[operation.path])
    )
    error: Final = next((item for item in built if isinstance(item, CatalogError)), None)
    if error is not None:
        return error
    catalog: Final = tuple(item for item in built if isinstance(item, Tool))
    if not catalog or sum(len(json.dumps(item.parameters).encode()) for item in catalog) > _MAX_SCHEMA_BYTES:
        return CatalogError("The management tool catalog is unavailable or exceeds the size limit")
    return catalog


def validate_arguments(
    tool: Tool, args: JsonValue
) -> dict[str, JsonValue] | ArgumentsError:  # mutable-ok: returns caller JSON arguments unchanged after validation
    if tool.operation not in OPERATIONS or not isinstance(args, dict):
        return ArgumentsError("Unknown operation or invalid argument object")
    try:
        encoded: Final = json.dumps(args, allow_nan=False)
    except (ValueError, RecursionError):
        return ArgumentsError("Arguments must contain finite JSON values")
    if len(encoded.encode()) > _MAX_ARGUMENT_BYTES:
        return ArgumentsError("Arguments exceed the size limit")
    if sanitize(args, argument_operation=tool.operation.name) != args:
        return ArgumentsError("Use key hashes or identifiers instead of credentials")
    if not _matches_schema(Draft202012Validator(tool.parameters, registry=Registry()), args):
        return ArgumentsError("Arguments do not match the selected operation schema")
    return args


def _query_values(value: SchemaValue) -> tuple[str, ...] | ArgumentsError:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) > _MAX_QUERY_ITEMS or any(isinstance(item, (list, tuple, Mapping)) for item in value):
            return ArgumentsError("Query arrays must contain at most 100 scalar values")
        return tuple(
            "true" if item is True else "false" if item is False else str(item) for item in value if item is not None
        )
    if isinstance(value, Mapping):
        return ArgumentsError("Query objects are not supported")
    return ("true" if value is True else "false" if value is False else str(value),)


def tool_request(tool: Tool, args: JsonValue) -> ToolRequest | ArgumentsError:
    validated: Final = validate_arguments(tool, args)
    if isinstance(validated, ArgumentsError):
        return validated
    path_args: Final = _object(validated.get("path"))
    if frozenset(re.findall(r"\{([^}]+)\}", tool.operation.path)) != frozenset(path_args):
        return ArgumentsError("Missing resource identifier")
    if any(
        not isinstance(value, str) or value in ("", ".", "..") or re.search(r"[/\\\x00-\x1f\x7f]", value)
        for value in path_args.values()
    ):
        return ArgumentsError("Invalid resource identifier")
    path: Final = re.sub(
        r"\{([^}]+)\}", lambda match: quote(str(path_args[match.group(1)]), safe=""), tool.operation.path
    )
    query: Final = tuple((name, _query_values(value)) for name, value in _object(validated.get("query")).items())
    error: Final = next((values for _, values in query if isinstance(values, ArgumentsError)), None)
    if error is not None:
        return error
    return ToolRequest(
        path,
        tuple((name, value) for name, values in query if isinstance(values, tuple) for value in values),
        validated.get("body"),
    )

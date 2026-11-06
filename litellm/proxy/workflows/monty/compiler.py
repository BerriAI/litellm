import ast
import hashlib
import keyword
import re
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import JsonValue
from referencing.exceptions import Unresolvable

from ._validator_worker import closed_registry
from .models import CompiledProgram, JsonSchema, MontyErrorKind, MontyFailure, MontyLimits, PythonProgram

_FORBIDDEN: Final = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "help",
    }
)
_PREFIX: Final = "_liteflow_"
_MAX_SCHEMA_DEPTH: Final = 64
_SCHEMA_MAPS: Final = frozenset({"properties", "$defs"})
_SCHEMA_VALUES: Final = frozenset({"additionalProperties", "items", "not"})
_SCHEMA_LISTS: Final = frozenset({"prefixItems", "anyOf", "oneOf", "allOf"})
_SCHEMA_ANNOTATIONS: Final = frozenset(
    {
        "const",
        "enum",
        "default",
        "examples",
        "title",
        "description",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
        "format",
        "x-liteflow-record",
        "type",
        "required",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "$ref",
    }
)
_GENERATED_GLOBALS: Final = frozenset({"dataclass", "str", "int", "float", "bool", "list", "dict", "all", "isinstance"})
_SCHEMA_EXPANSION_BUDGET: Final = 65536


class _SchemaWork:
    def __init__(self) -> None:
        self.units = 0
        self.checked: frozenset[tuple[int, str]] = frozenset()
        self.keywords: frozenset[tuple[int, str]] = frozenset()
        self.contains: Mapping[tuple[int, str], bool] = MappingProxyType({})

    def finish_keywords(self, key: tuple[int, str]) -> None:
        self.keywords = self.keywords | {key}

    def finish_record(self, key: tuple[int, str]) -> None:
        self.checked = self.checked | {key}

    def finish_contains(self, key: tuple[int, str], value: bool) -> None:
        self.contains = MappingProxyType(dict(self.contains) | {key: value})

    def spend(self, amount: int = 1) -> None:
        self.units += amount
        if self.units > _SCHEMA_EXPANSION_BUDGET:
            raise ValueError("schema expansion budget exceeded")


def escape_diagnostics(value: str) -> str:
    return "".join(
        f"\\x{ord(char):02x}" if (ord(char) < 32 and char not in "\n\t") or 127 <= ord(char) <= 159 else char
        for char in value
    )


def _invalid(rule: str, node: ast.AST) -> MontyFailure:
    return MontyFailure(
        MontyErrorKind.INVALID_PROGRAM, escape_diagnostics(f"{rule} at line {getattr(node, 'lineno', 1)}")
    )


def _safe_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) and "__" not in name and not name.startswith(_PREFIX)


def _module_statement_allowed(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.FunctionDef):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return isinstance(statement.value.value, str)
    return (
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and statement.level == 0
        and all(alias.name == "annotations" and alias.asname is None for alias in statement.names)
    )


def _function_error(node: ast.FunctionDef, functions: tuple[ast.FunctionDef, ...]) -> MontyFailure | None:
    if node not in functions:
        return _invalid("nested functions and closures are forbidden", node)
    if node.decorator_list:
        return _invalid("decorators are forbidden", node)
    if not _safe_name(node.name) or node.name in _GENERATED_GLOBALS:
        return _invalid("reserved function name", node)
    return None


def _node_error(node: ast.AST, tree: ast.Module, functions: tuple[ast.FunctionDef, ...]) -> MontyFailure | None:
    if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)):
        return _invalid(f"forbidden {type(node).__name__}", node)
    if isinstance(node, ast.FunctionDef):
        return _function_error(node, functions)
    if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
        return _invalid("imports inside functions are forbidden", node)
    if isinstance(node, ast.Name) and (not _safe_name(node.id) or node.id in _FORBIDDEN):
        return _invalid("forbidden builtin or reserved name", node)
    if isinstance(node, ast.arg) and not _safe_name(node.arg):
        return _invalid("reserved parameter name", node)
    if isinstance(node, ast.Attribute) and not _safe_name(node.attr):
        return _invalid("dunder or reserved attribute access", node)
    return None


def _source_error(tree: ast.Module, program: PythonProgram) -> MontyFailure | None:
    for statement in tree.body:
        if not _module_statement_allowed(statement):
            return _invalid("module body permits only function definitions and future annotations", statement)
    functions: Final = tuple(node for node in tree.body if isinstance(node, ast.FunctionDef))
    if len({node.name for node in functions}) != len(functions):
        return _invalid("duplicate function name", tree)
    entries: Final = tuple(node for node in functions if node.name == program.function_name)
    if len(entries) != 1:
        return _invalid("entry function must exist exactly once", tree)
    args: Final = entries[0].args
    if args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults:
        return _invalid("entry parameters must be positional without defaults or variadics", entries[0])
    if tuple(arg.arg for arg in args.args) != program.parameter_names:
        return _invalid("entry parameter order must match parameter_names", entries[0])
    return next(
        (failure for node in ast.walk(tree) if (failure := _node_error(node, tree, functions)) is not None), None
    )


def _mapping(value: JsonValue) -> JsonSchema:
    return value if isinstance(value, dict) else {}


def _resolve(schema: JsonSchema, root: JsonSchema, work: _SchemaWork) -> JsonSchema:
    work.spend()
    ref: Final = schema.get("$ref")
    if not isinstance(ref, str):
        return schema

    def _monty_schema_pointer(value: JsonValue, parts: tuple[str, ...]) -> JsonSchema:
        work.spend()
        if len(parts) > _MAX_SCHEMA_DEPTH:
            return {}
        if not parts:
            return _mapping(value)
        if isinstance(value, dict):
            return _monty_schema_pointer(value.get(parts[0].replace("~1", "/").replace("~0", "~")), parts[1:])
        if isinstance(value, list) and parts[0].isdigit() and int(parts[0]) < len(value):
            return _monty_schema_pointer(value[int(parts[0])], parts[1:])
        return {}

    return _monty_schema_pointer(root, tuple(ref[2:].split("/")))


def _monty_reference_exists(root: JsonValue, parts: tuple[str, ...], work: _SchemaWork) -> bool:
    work.spend()
    if len(parts) > _MAX_SCHEMA_DEPTH:
        return False
    if not parts:
        return isinstance(root, (dict, bool))
    part: Final = parts[0].replace("~1", "/").replace("~0", "~")
    if isinstance(root, dict) and part in root:
        return _monty_reference_exists(root[part], parts[1:], work)
    if isinstance(root, list) and part.isdigit() and int(part) < len(root):
        return _monty_reference_exists(root[int(part)], parts[1:], work)
    return False


def _monty_schema_error(
    value: JsonValue, root: JsonSchema, work: _SchemaWork, depth: int = 0, seen: frozenset[str] = frozenset()
) -> str | None:
    work.spend()
    if depth > _MAX_SCHEMA_DEPTH:
        return "schema depth limit exceeded"
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        work.spend()
        if key not in _SCHEMA_MAPS | _SCHEMA_VALUES | _SCHEMA_LISTS | _SCHEMA_ANNOTATIONS:
            return f"unsupported schema keyword {key}"
        if key == "$ref" and (not isinstance(item, str) or not item.startswith("#/")):
            return "only local #/ JSON references are supported"
        if (
            key == "$ref"
            and isinstance(item, str)
            and not _monty_reference_exists(root, tuple(item[2:].split("/")), work)
        ):
            return "local JSON reference target does not exist"
        if (
            key == "$ref"
            and isinstance(item, str)
            and (reference_error := _reference_schema_error(value, root, work, depth, seen, item)) is not None
        ):
            return reference_error
    return next(
        (
            error
            for child in _children(value, work)
            if (error := _monty_schema_error(child, root, work, depth + 1, seen)) is not None
        ),
        None,
    )


def _reference_schema_error(
    schema: JsonSchema, root: JsonSchema, work: _SchemaWork, depth: int, seen: frozenset[str], ref: str
) -> str | None:
    if ref in seen:
        return "recursive schema references are unsupported"
    cache_key: Final = (id(root), ref)
    if cache_key in work.keywords:
        return None
    result: Final = _monty_schema_error(_resolve(schema, root, work), root, work, depth + 1, seen | {ref})
    if result is None:
        work.finish_keywords(cache_key)
    return result


def _record_shape_error(schema: JsonSchema, root: JsonSchema, work: _SchemaWork) -> str | None:
    if schema.get("x-liteflow-record") is not True:
        return None
    if schema not in tuple(record for _, record in _records(root, work)):
        return "records must be defined in root $defs"
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        return "records require fixed object properties"
    if schema.get("additionalProperties") is not False:
        return "records require additionalProperties false"
    if any(not _safe_name(name) for name in _mapping(schema.get("properties"))):
        return "record field names must be safe identifiers"
    return None


def _record_union_error(schema: JsonSchema, root: JsonSchema, seen: frozenset[str], work: _SchemaWork) -> str | None:
    for keyword_name in ("anyOf", "oneOf", "allOf"):
        if not isinstance(alternatives := schema.get(keyword_name), list):
            continue
        if not any(_monty_contains_record(_mapping(item), root, seen, work) for item in alternatives):
            continue
        if keyword_name != "anyOf" or len(alternatives) != 2:
            return "records in unions are supported only as Optional X"
        if sum(_mapping(item).get("type") == "null" for item in alternatives) != 1:
            return "records in unions are supported only as Optional X"
    return None


def _monty_record_error(
    schema: JsonSchema, root: JsonSchema, work: _SchemaWork, seen: frozenset[str] = frozenset(), depth: int = 0
) -> str | None:
    work.spend()
    if depth > _MAX_SCHEMA_DEPTH:
        return "schema reference depth limit exceeded"
    ref: Final = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return "recursive schema references are unsupported"
        cache_key: Final = (id(root), ref)
        if cache_key in work.checked:
            return None
        result: Final = _monty_record_error(_resolve(schema, root, work), root, work, seen | {ref}, depth + 1)
        if result is None:
            work.finish_record(cache_key)
        return result
    if (shape_error := _record_shape_error(schema, root, work)) is not None:
        return shape_error
    if (union_error := _record_union_error(schema, root, seen, work)) is not None:
        return union_error
    return next(
        (
            error
            for child in _children(schema, work)
            if (error := _monty_record_error(child, root, work, seen, depth + 1)) is not None
        ),
        None,
    )


def _monty_contains_record(
    schema: JsonSchema, root: JsonSchema, seen: frozenset[str], work: _SchemaWork, depth: int = 0
) -> bool:
    work.spend()
    if depth > _MAX_SCHEMA_DEPTH:
        return False
    ref: Final = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return False
        cache_key: Final = (id(root), ref)
        if cache_key in work.contains:
            return work.contains[cache_key]
        result: Final = _monty_contains_record(_resolve(schema, root, work), root, seen | {ref}, work, depth + 1)
        work.finish_contains(cache_key, result)
        return result
    if schema.get("x-liteflow-record") is True:
        return True
    return any(_monty_contains_record(child, root, seen, work, depth + 1) for child in _children(schema, work))


def _children(schema: JsonSchema, work: _SchemaWork) -> tuple[JsonSchema, ...]:
    def children_of(key: str, value: JsonValue) -> tuple[JsonSchema, ...]:
        work.spend()
        if key in {"properties", "$defs"} and isinstance(value, dict):
            work.spend(len(value))
            return tuple(_mapping(child) for child in value.values())
        if key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(value, list):
            work.spend(len(value))
            return tuple(_mapping(child) for child in value)
        if key in {"items", "additionalProperties", "not"} and isinstance(value, dict):
            return (value,)
        return ()

    return tuple(chain.from_iterable(children_of(key, value) for key, value in schema.items()))


def _records(root: JsonSchema, work: _SchemaWork) -> tuple[tuple[str, JsonSchema], ...]:
    work.spend(len(_mapping(root.get("$defs"))))
    return tuple(
        (name, value)
        for name, value in _mapping(root.get("$defs")).items()
        if isinstance(value, dict) and value.get("x-liteflow-record") is True
    )


def _monty_build(schema: JsonSchema, root: JsonSchema, expression: str, work: _SchemaWork, depth: int = 0) -> str:
    work.spend()
    result: Final = _build_expression(schema, root, expression, work, depth)
    work.spend(len(result.encode("utf-8")))
    return result


def _build_expression(schema: JsonSchema, root: JsonSchema, expression: str, work: _SchemaWork, depth: int) -> str:
    if depth > _MAX_SCHEMA_DEPTH:
        raise ValueError("schema builder depth limit exceeded")
    resolved: Final = _resolve(schema, root, work)
    ref: Final = schema.get("$ref")
    if isinstance(ref, str) and "$ref" in resolved:
        return _monty_build(resolved, root, expression, work, depth + 1)
    union: Final = resolved.get("anyOf")
    if isinstance(union, list) and _monty_contains_record(resolved, root, frozenset(), work):
        other: Final = next(_mapping(item) for item in union if _mapping(item).get("type") != "null")
        return f"(None if {expression} is None else {_monty_build(other, root, expression, work, depth + 1)})"
    if resolved.get("x-liteflow-record") is True:
        name: Final = next(name for name, candidate in _records(root, work) if candidate == resolved)
        required: Final = resolved.get("required", [])

        def field(name: str, child: JsonValue) -> str:
            access: Final = (
                f"{expression}[{name!r}]"
                if isinstance(required, list) and name in required
                else f"{expression}.get({name!r})"
            )
            built: Final = _monty_build(_mapping(child), root, access, work, depth + 1)
            return (
                f"{name}={built}"
                if isinstance(required, list) and name in required
                else f"{name}=(None if {access} is None else {built})"
            )

        return (
            f"{name}("
            + ", ".join(field(key, child) for key, child in _mapping(resolved.get("properties")).items())
            + ")"
        )
    if resolved.get("type") == "array":
        variable: Final = f"{_PREFIX}item{depth}"
        return f"[{_monty_build(_mapping(resolved.get('items')), root, variable, work, depth + 1)} for {variable} in {expression}]"
    if resolved.get("type") == "object" and isinstance(resolved.get("properties"), dict):
        fields: Final = _mapping(resolved.get("properties"))
        variable_key: Final = f"{_PREFIX}key{depth}"
        variable_value: Final = f"{_PREFIX}value{depth}"
        branches: Final = "".join(
            f"{_monty_build(_mapping(child), root, variable_value, work, depth + 1)} if {variable_key} == {key!r} else "
            for key, child in fields.items()
        )
        fallback: Final = _monty_build(
            _mapping(resolved.get("additionalProperties")), root, variable_value, work, depth + 1
        )
        return (
            f"{{{variable_key}: ({branches}{fallback}) for {variable_key}, {variable_value} in {expression}.items()}}"
        )
    return expression


def _record_class(name: str, schema: JsonSchema, work: _SchemaWork) -> str:
    required: Final = schema.get("required", [])
    fields: Final = _mapping(schema.get("properties"))
    work.spend(len(fields))
    mandatory: Final = tuple(key for key in fields if isinstance(required, list) and key in required)
    optional: Final = tuple(key for key in fields if key not in mandatory)
    return (
        f"@dataclass(frozen=True)\nclass {name}:\n"
        + (
            "\n".join(f"    {key}: object" for key in mandatory)
            + "\n"
            + "\n".join(f"    {key}: object = None" for key in optional)
            if fields
            else "    pass"
        )
        + "\n"
    )


def _source(program: PythonProgram, work: _SchemaWork) -> str:
    records: Final = dict(_records(program.input_schema, work) + _records(program.output_schema, work))
    classes: Final = "\n".join(_record_class(name, schema, work) for name, schema in records.items())
    record_dump: Final = "\n".join(
        f"    if isinstance(value, {name}):\n        return {_PREFIX}dump({{"
        + ", ".join(f"{field!r}: value.{field}" for field in _mapping(schema.get("properties")))
        + "})"
        for name, schema in records.items()
    )
    dumper: Final = f"""def {_PREFIX}dump(value):
{record_dump}
    if value is None or isinstance(value, (str, int, float, bool)):
        return [True, value]
    if isinstance(value, list):
        pairs = [{_PREFIX}dump(item) for item in value]
        return [all(pair[0] for pair in pairs), [pair[1] for pair in pairs]]
    if isinstance(value, dict):
        pairs = {{key: {_PREFIX}dump(item) for key, item in value.items()}}
        return [all(isinstance(key, str) and pair[0] for key, pair in pairs.items()), {{key: pair[1] for key, pair in pairs.items()}}]
    return [False, None]
"""
    arguments: Final = ", ".join(
        _monty_build(
            _mapping(_mapping(program.input_schema.get("properties")).get(name)),
            program.input_schema,
            f"{_PREFIX}input[{name!r}]",
            work,
        )
        for name in program.parameter_names
    )
    return (
        "from dataclasses import dataclass\n"
        + classes
        + "\n"
        + program.source
        + "\n"
        + dumper
        + f"\n{_PREFIX}dump({program.function_name}({arguments}))"
    )


def _schemas_error(program: PythonProgram, tree: ast.Module, work: _SchemaWork) -> MontyFailure | None:
    function_names: Final = frozenset(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
    validator: Final = Draft202012Validator(Draft202012Validator.META_SCHEMA, registry=closed_registry())
    for schema in (program.input_schema, program.output_schema):
        if (error := _monty_schema_error(schema, schema, work)) is not None:
            return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(error))
        validator.validate(schema)  # pyright: ignore[reportUnknownMemberType]  # upstream kwargs are untyped
        if (record_error := _monty_record_error(schema, schema, work)) is not None:
            return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(record_error))
        for name, _ in _records(schema, work):
            if not _safe_name(name) or name in _GENERATED_GLOBALS or name in function_names:
                return MontyFailure(
                    MontyErrorKind.UNSUPPORTED_SCHEMA, "record class name must be a distinct safe identifier"
                )
    input_records: Final = dict(_records(program.input_schema, work))
    if any(
        name in input_records and input_records[name] != schema
        for name, schema in _records(program.output_schema, work)
    ):
        return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, "conflicting record definitions")
    return _input_schema_error(program)


def _input_schema_error(program: PythonProgram) -> MontyFailure | None:
    inputs: Final = program.input_schema
    required: Final = inputs.get("required")
    if (
        inputs.get("type") != "object"
        or set(_mapping(inputs.get("properties"))) != set(program.parameter_names)
        or inputs.get("additionalProperties") is not False
        or not isinstance(required, list)
        or set(required) != set(program.parameter_names)
    ):
        return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, "input schema must require exactly the named parameters")
    return None


def compile_program(program: PythonProgram, limits: MontyLimits) -> CompiledProgram | MontyFailure:
    try:
        encoded: Final = program.source.encode("utf-8")
        if len(encoded) > limits.max_source_bytes:
            return MontyFailure(MontyErrorKind.INVALID_PROGRAM, "source byte limit exceeded at line 1")
        if hashlib.sha256(encoded).hexdigest() != program.source_sha256:
            return MontyFailure(MontyErrorKind.INVALID_PROGRAM, "source_sha256 mismatch at line 1")
        if re.search(r"(?m)^\s*#\s*///\s*script\b", program.source):
            return MontyFailure(MontyErrorKind.INVALID_PROGRAM, "PEP 723 script blocks forbidden at line 1")
        tree: Final = ast.parse(program.source)
        if (failure := _source_error(tree, program)) is not None:
            return failure
        work: Final = _SchemaWork()
        if (schema_failure := _schemas_error(program, tree, work)) is not None:
            return schema_failure
        generated: Final = _source(program, work)
        work.spend(len(generated.encode("utf-8")))
        return CompiledProgram(program, generated, program.input_schema, program.output_schema)
    except SyntaxError as exc:
        return MontyFailure(
            MontyErrorKind.INVALID_PROGRAM, escape_diagnostics(f"syntax error at line {exc.lineno}: {exc.msg}")
        )
    except (SchemaError, ValidationError, Unresolvable, ValueError, RecursionError, StopIteration) as exc:
        return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(str(exc)))

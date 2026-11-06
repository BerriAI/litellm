import ast
import hashlib
import keyword
import re
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import JsonValue

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
_GENERATED_GLOBALS: Final = frozenset({"dataclass", "str", "int", "float", "bool", "list", "dict", "all", "isinstance"})


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


def _source_error(tree: ast.Module, program: PythonProgram) -> MontyFailure | None:
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__" and statement.level == 0:
            if all(alias.name == "annotations" and alias.asname is None for alias in statement.names):
                continue
        if not isinstance(statement, ast.FunctionDef):
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
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)):
            return _invalid(f"forbidden {type(node).__name__}", node)
        if isinstance(node, ast.FunctionDef):
            if node not in functions:
                return _invalid("nested functions and closures are forbidden", node)
            if node.decorator_list:
                return _invalid("decorators are forbidden", node)
            if not _safe_name(node.name) or node.name in _GENERATED_GLOBALS:
                return _invalid("reserved function name", node)
            if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
                return _invalid("function defaults are forbidden", node)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            return _invalid("imports inside functions are forbidden", node)
        if isinstance(node, ast.Name) and (not _safe_name(node.id) or node.id in _FORBIDDEN):
            return _invalid("forbidden builtin or reserved name", node)
        if isinstance(node, ast.arg) and not _safe_name(node.arg):
            return _invalid("reserved parameter name", node)
        if isinstance(node, ast.Attribute) and not _safe_name(node.attr):
            return _invalid("dunder or reserved attribute access", node)
    return None


def _mapping(value: JsonValue) -> JsonSchema:
    return value if isinstance(value, dict) else {}


def _resolve(schema: JsonSchema, root: JsonSchema) -> JsonSchema:
    ref: Final = schema.get("$ref")
    if not isinstance(ref, str):
        return schema

    def descend(value: JsonValue, parts: tuple[str, ...]) -> JsonSchema:
        if not parts:
            return _mapping(value)
        if isinstance(value, dict):
            return descend(value.get(parts[0].replace("~1", "/").replace("~0", "~")), parts[1:])
        if isinstance(value, list) and parts[0].isdigit() and int(parts[0]) < len(value):
            return descend(value[int(parts[0])], parts[1:])
        return {}

    return descend(root, tuple(ref[2:].split("/")))


def _schema_error(value: JsonValue) -> str | None:
    if isinstance(value, list):
        return next((error for item in value if (error := _schema_error(item)) is not None), None)
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if key in {"$id", "$dynamicRef", "$dynamicAnchor", "$anchor", "$schema"}:
            return f"unsupported schema keyword {key}"
        if key == "$ref" and (not isinstance(item, str) or not item.startswith("#/")):
            return "only local #/ JSON references are supported"
        if key not in {"const", "enum", "default", "examples"}:
            if (error := _schema_error(item)) is not None:
                return error
    return None


def _record_error(schema: JsonSchema, root: JsonSchema, seen: frozenset[str] = frozenset()) -> str | None:
    ref: Final = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return "recursive schema references are unsupported"
        return _record_error(_resolve(schema, root), root, seen | {ref})
    if schema.get("x-liteflow-record") is True:
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            return "records require fixed object properties"
        if schema.get("additionalProperties") is not False:
            return "records require additionalProperties false"
        if any(not _safe_name(name) for name in _mapping(schema.get("properties"))):
            return "record field names must be safe identifiers"
    for keyword_name in ("anyOf", "oneOf", "allOf"):
        alternatives: Final = schema.get(keyword_name)
        if isinstance(alternatives, list):
            if any(_contains_record(_mapping(item), root, seen) for item in alternatives):
                if keyword_name != "anyOf" or len(alternatives) != 2:
                    return "records in unions are supported only as Optional X"
                if sum(_mapping(item).get("type") == "null" for item in alternatives) != 1:
                    return "records in unions are supported only as Optional X"
    for key, value in schema.items():
        if key in {"properties", "$defs"} and isinstance(value, dict):
            for child in value.values():
                if (error := _record_error(_mapping(child), root, seen)) is not None:
                    return error
        if key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(value, list):
            for child_schema in value:
                if (error := _record_error(_mapping(child_schema), root, seen)) is not None:
                    return error
        if key in {"items", "additionalProperties"} and isinstance(value, dict):
            if (error := _record_error(value, root, seen)) is not None:
                return error
    return None


def _contains_record(schema: JsonSchema, root: JsonSchema, seen: frozenset[str]) -> bool:
    ref: Final = schema.get("$ref")
    if isinstance(ref, str):
        return ref not in seen and _contains_record(_resolve(schema, root), root, seen | {ref})
    if schema.get("x-liteflow-record") is True:
        return True
    return any(_contains_record(child, root, seen) for child in _children(schema))


def _children(schema: JsonSchema) -> tuple[JsonSchema, ...]:
    def children_of(key: str, value: JsonValue) -> tuple[JsonSchema, ...]:
        if key in {"properties", "$defs"} and isinstance(value, dict):
            return tuple(_mapping(child) for child in value.values())
        if key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(value, list):
            return tuple(_mapping(child) for child in value)
        if key in {"items", "additionalProperties"} and isinstance(value, dict):
            return (value,)
        return ()

    return tuple(child for group in (children_of(key, value) for key, value in schema.items()) for child in group)


def _records(root: JsonSchema) -> tuple[tuple[str, JsonSchema], ...]:
    return tuple(
        (name, value)
        for name, value in _mapping(root.get("$defs")).items()
        if isinstance(value, dict) and value.get("x-liteflow-record") is True
    )


def _build(schema: JsonSchema, root: JsonSchema, expression: str, depth: int = 0) -> str:
    resolved: Final = _resolve(schema, root)
    ref: Final = schema.get("$ref")
    if isinstance(ref, str) and "$ref" in resolved:
        return _build(resolved, root, expression, depth + 1)
    union: Final = resolved.get("anyOf")
    if isinstance(union, list) and _contains_record(resolved, root, frozenset()):
        other: Final = next(_mapping(item) for item in union if _mapping(item).get("type") != "null")
        return f"(None if {expression} is None else {_build(other, root, expression, depth + 1)})"
    if resolved.get("x-liteflow-record") is True:
        name: Final = next(name for name, candidate in _records(root) if candidate == resolved)
        required: Final = resolved.get("required", [])

        def field(name: str, child: JsonValue) -> str:
            access: Final = (
                f"{expression}[{name!r}]"
                if isinstance(required, list) and name in required
                else f"{expression}.get({name!r})"
            )
            built: Final = _build(_mapping(child), root, access, depth + 1)
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
        return f"[{_build(_mapping(resolved.get('items')), root, variable, depth + 1)} for {variable} in {expression}]"
    if resolved.get("type") == "object" and isinstance(resolved.get("properties"), dict):
        fields: Final = _mapping(resolved.get("properties"))
        variable_key: Final = f"{_PREFIX}key{depth}"
        variable_value: Final = f"{_PREFIX}value{depth}"
        branches: Final = "".join(
            f"{_build(_mapping(child), root, variable_value, depth + 1)} if {variable_key} == {key!r} else "
            for key, child in fields.items()
        )
        fallback: Final = _build(_mapping(resolved.get("additionalProperties")), root, variable_value, depth + 1)
        return (
            f"{{{variable_key}: ({branches}{fallback}) for {variable_key}, {variable_value} in {expression}.items()}}"
        )
    return expression


def _record_class(name: str, schema: JsonSchema) -> str:
    required: Final = schema.get("required", [])
    fields: Final = _mapping(schema.get("properties"))
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


def _source(program: PythonProgram) -> str:
    records: Final = dict(_records(program.input_schema) + _records(program.output_schema))
    classes: Final = "\n".join(_record_class(name, schema) for name, schema in records.items())
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
        _build(
            _mapping(_mapping(program.input_schema.get("properties")).get(name)),
            program.input_schema,
            f"{_PREFIX}input[{name!r}]",
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
        for schema in (program.input_schema, program.output_schema):
            if (error := _schema_error(schema)) is not None:
                return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(error))
            Draft202012Validator.check_schema(schema)
            if (record_error := _record_error(schema, schema)) is not None:
                return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(record_error))
            for name, record in _records(schema):
                if (
                    not _safe_name(name)
                    or name in _GENERATED_GLOBALS
                    or name in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                ):
                    return MontyFailure(
                        MontyErrorKind.UNSUPPORTED_SCHEMA, "record class name must be a distinct safe identifier"
                    )
        input_records: Final = dict(_records(program.input_schema))
        if any(
            name in input_records and input_records[name] != schema for name, schema in _records(program.output_schema)
        ):
            return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, "conflicting record definitions")
        inputs: Final = program.input_schema
        if (
            inputs.get("type") != "object"
            or set(_mapping(inputs.get("properties"))) != set(program.parameter_names)
            or inputs.get("additionalProperties") is not False
            or not isinstance(inputs.get("required"), list)
            or set(inputs["required"]) != set(program.parameter_names)
        ):
            return MontyFailure(
                MontyErrorKind.UNSUPPORTED_SCHEMA, "input schema must require exactly the named parameters"
            )
        return CompiledProgram(program, _source(program), program.input_schema, program.output_schema)
    except SyntaxError as exc:
        return MontyFailure(
            MontyErrorKind.INVALID_PROGRAM, escape_diagnostics(f"syntax error at line {exc.lineno}: {exc.msg}")
        )
    except (SchemaError, ValueError, RecursionError, StopIteration) as exc:
        return MontyFailure(MontyErrorKind.UNSUPPORTED_SCHEMA, escape_diagnostics(str(exc)))

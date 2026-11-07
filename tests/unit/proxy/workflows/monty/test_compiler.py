import ast
import hashlib
from functools import reduce
from typing import Final

import pytest

from litellm.proxy.workflows.monty.compiler import _SchemaWork, _schemas_error, compile_program, escape_diagnostics
from litellm.proxy.workflows.monty.models import (
    CompiledProgram,
    JsonSchema,
    MontyErrorKind,
    MontyFailure,
    MontyLimits,
    PythonProgram,
)


def _program(source: str, *, output: JsonSchema | None = None) -> PythonProgram:
    return PythonProgram(
        contract_version="1",
        source=source,
        function_name="transform",
        parameter_names=("value",),
        input_schema={
            "type": "object",
            "properties": {"value": {}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema=output if output is not None else {},
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef transform(value): return value",
        "from typing import TypedDict\ndef transform(value): return value",
        "x=1\ndef transform(value): return value",
        "print('side effect')\ndef transform(value): return value",
        "async def transform(value): return value",
        "class X: pass\ndef transform(value): return value",
        "@print\ndef transform(value): return value",
        "def transform(value):\n def closure(): return value\n return closure()",
        "def transform(value): return (lambda: value)()",
        "def transform(value):\n global x\n return value",
        "def transform(value):\n nonlocal x\n return value",
        "def transform(value): return value.__class__",
        "def transform(value): return __name__",
        "# /// script\ndef transform(value): return value",
        "def transform(other): return other",
        "def transform(value=1): return value",
        "def transform(*value): return value",
        "def transform(value, **kwargs): return value",
        "def transform(*, value): return value",
        "def transform(value, /): return value",
        "def transform(value):\n import json\n return value",
        "def transform(value): return value\ndef isinstance(x,y): return True",
    ],
)
def test_rejects_unsupported_source_without_running_it(source: str) -> None:
    program: Final = _program(source)
    result: Final = compile_program(program, MontyLimits())
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.INVALID_PROGRAM
    assert "line" in result.message


@pytest.mark.parametrize(
    "builtin",
    [
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
    ],
)
def test_rejects_dangerous_builtin_references(builtin: str) -> None:
    result: Final = compile_program(_program(f"def transform(value): return {builtin}"), MontyLimits())
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.INVALID_PROGRAM


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "file:///etc/passwd"},
        {"$id": "https://example.invalid/schema"},
        {"$dynamicRef": "#/x"},
        {"$schema": "https://example.invalid/schema"},
        {"$ref": "#/$defs/Missing"},
        {"type": "not-a-json-type"},
        {"$defs": {"Bad": {"type": "object", "x-liteflow-record": True, "properties": {}}}},
        {
            "$defs": {
                "Bad-Name": {
                    "type": "object",
                    "x-liteflow-record": True,
                    "properties": {},
                    "additionalProperties": False,
                }
            }
        },
    ],
)
def test_rejects_unsupported_schema_without_remote_retrieval(schema: JsonSchema) -> None:
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA


def test_hash_covers_exact_source_bytes() -> None:
    program: Final = _program("def transform(value): return value")
    result: Final = compile_program(program.model_copy(update={"source": program.source + "\n"}), MontyLimits())
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.INVALID_PROGRAM
    assert "source_sha256" in result.message


def test_annotations_are_not_evaluated() -> None:
    result: Final = compile_program(
        _program("def transform(value: Unknown[Foo]) -> Missing: return value"), MontyLimits()
    )
    assert isinstance(result, CompiledProgram)


def test_source_size_is_bounded() -> None:
    result: Final = compile_program(_program("def transform(value): return value"), MontyLimits(max_source_bytes=8))
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.INVALID_PROGRAM


def test_diagnostics_escape_terminal_controls_but_preserve_layout() -> None:
    assert escape_diagnostics("a\x1b[31m\x00\x7f\x9b\n\t") == "a\\x1b[31m\\x00\\x7f\\x9b\n\t"


def _array_schema(child: JsonSchema, index: int) -> JsonSchema:
    return {"type": "array", "items": child}


@pytest.mark.parametrize("depth,accepted", [(4, True), (70, False)])
def test_schema_nesting_is_bounded(depth: int, accepted: bool) -> None:
    schema: Final = reduce(_array_schema, range(depth), {})
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    if accepted:
        assert isinstance(result, CompiledProgram), result
        return
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert "depth limit" in result.message


@pytest.mark.parametrize("depth,accepted", [(4, True), (70, False)])
def test_schema_reference_expansion_is_bounded(depth: int, accepted: bool) -> None:
    schema: Final[JsonSchema] = {
        "$defs": {
            f"N{index}": {"$ref": f"#/$defs/N{index + 1}"} if index < depth else {"type": "integer"}
            for index in range(depth + 1)
        },
        "$ref": "#/$defs/N0",
    }
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    if accepted:
        assert isinstance(result, CompiledProgram), result
        return
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert "depth limit" in result.message


def test_schema_reference_cycle_is_rejected() -> None:
    schema: Final[JsonSchema] = {"$defs": {"X": {"$ref": "#/$defs/X"}}, "$ref": "#/$defs/X"}
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert "recursive schema" in result.message


_RECORD: Final[JsonSchema] = {
    "type": "object",
    "x-liteflow-record": True,
    "properties": {"amount": {"type": "integer"}, "label": {"type": ["string", "null"]}},
    "required": ["amount"],
    "additionalProperties": False,
}
_RECORD_REF: Final[JsonSchema] = {"$ref": "#/$defs/Record"}


@pytest.mark.parametrize(
    "argument",
    [
        _RECORD_REF,
        {"type": "array", "items": _RECORD_REF},
        {"anyOf": [_RECORD_REF, {"type": "null"}]},
        {"type": "object", "properties": {"payload": _RECORD_REF}, "additionalProperties": True},
        {"type": "object", "properties": {}, "additionalProperties": _RECORD_REF},
        {"$ref": "#/$defs/Envelope"},
    ],
)
def test_registration_accepts_records_in_supported_input_shapes(argument: JsonSchema) -> None:
    definitions: Final[JsonSchema] = {
        "Record": _RECORD,
        "Empty": {**_RECORD, "properties": {}, "required": []},
        "Envelope": {**_RECORD, "properties": {"child": _RECORD_REF}, "required": ["child"]},
    }
    program: Final = _program("def transform(value): return value", output={"$defs": definitions})
    typed: Final = program.model_copy(
        update={
            "input_schema": {
                **program.input_schema,
                "properties": {"value": argument},
                "$defs": definitions,
            }
        }
    )
    result: Final = compile_program(typed, MontyLimits())
    assert isinstance(result, CompiledProgram), result


@pytest.mark.parametrize(
    "schema",
    [
        _RECORD,
        {"$defs": {"Record": {**_RECORD, "type": "string"}}},
        {"$defs": {"Record": {**_RECORD, "properties": {"bad-field": {}}}}},
        {"$defs": {"str": _RECORD}},
        {"$defs": {"transform": _RECORD}},
        {"$defs": {"Record": _RECORD}, "oneOf": [_RECORD_REF, {"type": "null"}]},
        {"$defs": {"Record": _RECORD}, "anyOf": [_RECORD_REF, {"type": "string"}]},
        {"$defs": {"Record": _RECORD}, "anyOf": [_RECORD_REF, {"type": "null"}, {"type": "string"}]},
        {"$defs": {"Record": _RECORD}, "allOf": [_RECORD_REF]},
    ],
)
def test_registration_rejects_ambiguous_or_unconstructible_record_shapes(schema: JsonSchema) -> None:
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "array"},
        {"properties": {"extra": {}}},
        {"required": []},
        {"required": ["value", "extra"]},
        {"additionalProperties": True},
    ],
)
def test_input_schema_must_require_exactly_the_entry_parameters(changes: JsonSchema) -> None:
    program: Final = _program("def transform(value): return value")
    invalid: Final = program.model_copy(update={"input_schema": {**program.input_schema, **changes}})
    result: Final = compile_program(invalid, MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA


def test_registration_rejects_conflicting_record_definitions() -> None:
    program: Final = _program("def transform(value): return value", output={"$defs": {"Record": _RECORD}})
    conflicting: Final = program.model_copy(
        update={"input_schema": {**program.input_schema, "$defs": {"Record": {**_RECORD, "required": []}}}}
    )
    result: Final = compile_program(conflicting, MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert "conflicting" in result.message


@pytest.mark.parametrize(
    "source",
    ["def transform(value):\n", "def transform(value): return value\ndef transform(value): return 1"],
)
def test_registration_rejects_syntax_errors_and_duplicate_entry_names(source: str) -> None:
    result: Final = compile_program(_program(source), MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.INVALID_PROGRAM


def test_registration_accepts_local_json_pointer_escaping_and_annotations_import() -> None:
    schema: Final[JsonSchema] = {"$defs": {"a/b~c": {"type": "integer"}}, "$ref": "#/$defs/a~1b~0c"}
    source: Final = "from __future__ import annotations\ndef transform(value: Missing) -> Unknown: return value"
    result: Final = compile_program(_program(source, output=schema), MontyLimits())
    assert isinstance(result, CompiledProgram), result


@pytest.mark.parametrize("name", ["default", "pattern", "enum", "$ref", "$ref-like"])
def test_keyword_looking_property_names_are_data(name: str) -> None:
    schema: Final[JsonSchema] = {"type": "object", "properties": {name: {"type": "string"}}}
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, CompiledProgram), result


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"default": {"$ref": "https://example.invalid/schema"}}},
        {"$defs": {"default": {"items": {"$ref": "https://example.invalid/schema"}}}},
        {"anyOf": [{"properties": {"enum": {"$ref": "https://example.invalid/schema"}}}]},
        {"default": {"$ref": "https://example.invalid/schema"}, "$ref": "#/default"},
    ],
)
def test_remote_references_under_keyword_looking_names_are_rejected(schema: JsonSchema) -> None:
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert "local" in result.message


@pytest.mark.parametrize(
    "keyword",
    [
        "pattern",
        "patternProperties",
        "uniqueItems",
        "$id",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$schema",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "contains",
        "minContains",
        "maxContains",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
        "discriminator",
        "unknown",
    ],
)
def test_unsafe_keywords_are_explicitly_refused(keyword: str) -> None:
    result: Final = compile_program(_program("def transform(value): return value", output={keyword: {}}), MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert keyword in result.message


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_annotation_and_data_values_are_not_interpreted_as_schemas(keyword: str) -> None:
    payload: Final[JsonSchema] = {"$ref": "https://example.invalid/data", "pattern": "inert"}
    schema: Final[JsonSchema] = {keyword: [payload] if keyword in {"enum", "examples"} else payload}
    result: Final = compile_program(_program("def transform(value): return value", output=schema), MontyLimits())
    assert isinstance(result, CompiledProgram), result


@pytest.mark.parametrize("depth", [8, 12, 14])
def test_repeated_reference_analysis_has_linear_work(depth: int) -> None:
    definitions: Final[JsonSchema] = {
        f"N{index}": {"anyOf": [{"$ref": f"#/$defs/N{index + 1}"}, {"$ref": f"#/$defs/N{index + 1}"}]}
        for index in range(depth)
    }
    schema: Final[JsonSchema] = {"$defs": {**definitions, f"N{depth}": {}}, "$ref": "#/$defs/N0"}
    program: Final = _program("def transform(value): return value", output=schema)
    work: Final = _SchemaWork()
    assert _schemas_error(program, ast.parse(program.source), work) is None
    assert work.units < 100 * (depth + 1)
    result: Final = compile_program(program, MontyLimits())
    assert isinstance(result, CompiledProgram), result


def test_expanded_input_builder_fails_closed_at_shared_work_budget() -> None:
    definitions: Final[JsonSchema] = {
        f"N{index}": {
            "type": "object",
            "properties": {"left": {"$ref": f"#/$defs/N{index + 1}"}, "right": {"$ref": f"#/$defs/N{index + 1}"}},
        }
        for index in range(14)
    }
    program: Final = _program("def transform(value): return value")
    expanded: Final = program.model_copy(
        update={
            "input_schema": {
                **program.input_schema,
                "$defs": {**definitions, "N14": {"type": "integer"}},
                "properties": {"value": {"$ref": "#/$defs/N0"}},
            }
        }
    )
    result: Final = compile_program(expanded, MontyLimits())
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.UNSUPPORTED_SCHEMA
    assert result.message == "schema expansion budget exceeded"

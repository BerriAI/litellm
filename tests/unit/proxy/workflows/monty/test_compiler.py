import hashlib
from typing import Final

import pytest

from litellm.proxy.workflows.monty.compiler import compile_program, escape_diagnostics
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

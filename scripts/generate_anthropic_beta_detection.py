# /// script
# requires-python = ">=3.10"
# dependencies = ["datamodel-code-generator==0.66.0", "pydantic==2.13.4", "ruff==0.15.3"]
# ///
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import assert_never

ROOT: Final = Path(__file__).resolve().parents[1]
RULES: Final = ROOT / "litellm/llms/anthropic/beta_detection_rules.json"
PYTHON: Final = ROOT / "litellm/llms/anthropic/beta_detection_generated.py"
MODELS: Final = ROOT / "litellm/llms/anthropic/beta_detection_models.py"
SCHEMA: Final = ROOT / "litellm/llms/anthropic/beta_detection.schema.json"
RUST: Final = ROOT / "litellm-rust/crates/llms/src/anthropic/beta_detection_generated.rs"


class Membership(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    op: Literal["in"]
    field: str
    values: tuple[str, ...] = Field(min_length=1)


class All(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    op: Literal["all"]
    conditions: tuple[Condition, ...] = Field(min_length=1)


class AnyItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    op: Literal["any"]
    field: str
    condition: Condition


Condition: TypeAlias = Annotated[Membership | All | AnyItem, Field(discriminator="op")]
Scope: TypeAlias = Literal["request", "message", "block"]


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    beta: str = Field(pattern=r"^[a-z]+(?:-[a-z0-9]+)+-\d{4}-\d{2}-\d{2}$")
    when: Condition


class Definition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    rules: tuple[Rule, ...] = Field(min_length=1)


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True)
    check: bool


PYTHON_FIELDS: Final[Mapping[tuple[Scope, str], str]] = MappingProxyType(
    {
        ("request", "thinking.type"): "thinking_type(request)",
        ("request", "thinking.display"): "thinking_display(request)",
        ("message", "role"): "message.role",
        ("block", "type"): "block.type",
        ("block", "tool.type"): "tool_type(block)",
    }
)
RUST_FIELDS: Final[Mapping[tuple[Scope, str], str]] = MappingProxyType(
    {
        ("block", "tool.type"): "block.tool.as_ref().and_then(Recognized::known).map(|tool| &tool.type_)",
    }
)
COLLECTIONS: Final[Mapping[tuple[Scope, str], tuple[Scope, str, str]]] = MappingProxyType(
    {
        ("request", "messages"): ("message", "request.messages", "request.messages.iter()"),
        ("message", "content"): ("block", "content_blocks(message)", "content_blocks(message)"),
    }
)


def rust_membership(condition: Membership, scope: Scope) -> str:
    literals: Final = ", ".join(json.dumps(value) for value in condition.values)
    match scope, condition.field:
        case "request", "thinking.type":
            patterns: Final = " | ".join(f"Some(BetaThinkingType::{value.title()})" for value in condition.values)
            return f"matches!(thinking_type(request), {patterns})"
        case "request", "thinking.display":
            displays: Final = " | ".join(f"Some(BetaThinkingDisplay::{value.title()})" for value in condition.values)
            return f"matches!(thinking_display(request), {displays})"
        case "block", "type":
            actions: Final = " | ".join(
                "BetaContentType::" + "".join(part.title() for part in value.split("_")) for value in condition.values
            )
            return f"matches!(block.type_, {actions})"
        case "block", "tool.type":
            targets: Final = " | ".join(
                "Some(BetaToolTargetType::" + "".join(part.title() for part in value.split("_")) + ")"
                for value in condition.values
            )
            return f"matches!({RUST_FIELDS[(scope, condition.field)]}, {targets})"
        case "message", "role":
            return f"[{literals}].contains(&message.role.as_str())"
        case _:
            expression: Final = RUST_FIELDS[(scope, condition.field)]
            return f"({expression}).is_some_and(|value| [{literals}].contains(&value))"


def compile_condition(condition: Condition, scope: Scope, language: Literal["python", "rust"]) -> str:
    match condition:
        case Membership():
            if language == "rust":
                return rust_membership(condition, scope)
            field_expression: Final = PYTHON_FIELDS[(scope, condition.field)]
            enum_name: Final = {
                ("request", "thinking.type"): "BetaThinkingType",
                ("request", "thinking.display"): "BetaThinkingDisplay",
                ("block", "type"): "BetaContentType",
                ("block", "tool.type"): "BetaToolTargetType",
            }.get((scope, condition.field))
            values: Final = ", ".join(
                f"{enum_name}.{value}" if enum_name else json.dumps(value) for value in condition.values
            )
            return f"{field_expression} in ({values},)"
        case All():
            operator: Final = " and " if language == "python" else " &&\n"
            all_expression: Final = operator.join(
                compile_condition(child, scope, language) for child in condition.conditions
            )
            return f"({all_expression})" if language == "python" else all_expression
        case AnyItem():
            child_scope, python_collection, rust_collection = COLLECTIONS[(scope, condition.field)]
            child: Final = compile_condition(condition.condition, child_scope, language)
            if language == "rust":
                return f"{rust_collection}.any(|{child_scope}| {{\n{child}\n}})"
            return f"any({child} for {child_scope} in {python_collection})"
        case _:
            assert_never(condition)


def python_source(definition: Definition) -> str:
    rows: Final = "\n".join(
        f"            ({compile_condition(rule.when, 'request', 'python')}, {json.dumps(rule.beta)}),"
        for rule in definition.rules
    )
    return f"""from .beta_detection import content_blocks, thinking_display, thinking_type, tool_type
from .beta_detection_models import (
    BetaContentType,
    BetaDetectionRequest,
    BetaThinkingDisplay,
    BetaThinkingType,
    BetaToolTargetType,
)


def detect_anthropic_betas(request: BetaDetectionRequest) -> frozenset[str]:
    return frozenset(
        beta
        for used, beta in (
{rows}
        )
        if used
    )
"""


def rust_source(definition: Definition) -> str:
    predicates: Final = "\n".join(
        f"fn {rule.feature}_used(request: &BetaDetectionRequest) -> bool {{\n"
        f"{compile_condition(rule.when, 'request', 'rust')}\n}}\n"
        for rule in definition.rules
    )
    rows: Final = "\n".join(
        f"        {rule.feature}_used(request)"
        f".then_some(AnthropicBeta::{''.join(part.title() for part in rule.beta.split('-'))}),"
        for rule in definition.rules
    )
    return f"""{predicates}
pub fn detect_anthropic_betas(request: &BetaDetectionRequest) -> BetaSet {{
    [
{rows}
    ]
    .into_iter()
    .flatten()
    .collect()
}}
"""


def formatted_source(source: str, language: Literal["python", "rust"]) -> str:
    command: Final = (
        (sys.executable, "-m", "ruff", "format", "--line-length", "120", "--stdin-filename", str(PYTHON), "-")
        if language == "python"
        else ("rustfmt", "--edition", "2024", "--emit", "stdout")
    )
    return subprocess.run(command, input=source, text=True, capture_output=True, check=True).stdout


def python_models(directory: Path) -> str:
    output: Final = directory / "beta_detection_models.py"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(SCHEMA),
            "--input-file-type",
            "jsonschema",
            "--output",
            str(output),
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--target-python-version",
            "3.10",
            "--disable-timestamp",
            "--custom-file-header",
            "# @generated by scripts/generate_anthropic_beta_detection.py, do not edit",
            "--use-standard-collections",
            "--use-generic-container-types",
            "--use-union-operator",
            "--use-title-as-name",
            "--union-mode",
            "left_to_right",
            "--enable-faux-immutability",
            "--strict-types",
            "str",
            "int",
            "float",
            "bool",
            "--type-overrides",
            '{"JsonValue":"pydantic.JsonValue"}',
            "--additional-imports",
            "collections.abc.Mapping,typing.TypeAlias,pydantic.JsonValue",
            "--formatters",
            "ruff-check",
            "ruff-format",
        ),
        check=True,
    )
    subprocess.run(
        (sys.executable, "-m", "ruff", "check", "--select", "I,F401", "--fix", str(output)),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return formatted_source(output.read_text(), "python")


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Compile shared Anthropic beta rules to Python and Rust")
    parser.add_argument("--check", action="store_true")
    arguments: Final = Arguments.model_validate(vars(parser.parse_args()))
    definition: Final = Definition.model_validate_json(RULES.read_text())
    with TemporaryDirectory(prefix="anthropic-beta-codegen-") as temporary:
        outputs: Final = (
            (MODELS, python_models(Path(temporary))),
            (PYTHON, formatted_source(python_source(definition), "python")),
            (RUST, formatted_source(rust_source(definition), "rust")),
        )
    stale: Final = tuple(path for path, source in outputs if not path.exists() or path.read_text() != source)
    if arguments.check:
        sys.stderr.write("".join(f"stale {path.relative_to(ROOT)}\n" for path in stale))
        return int(bool(stale))
    for path, source in outputs:
        path.write_text(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import assert_never

ROOT: Final = Path(__file__).resolve().parents[1]
RULES: Final = ROOT / "litellm/llms/anthropic/beta_detection_rules.json"
PYTHON: Final = ROOT / "litellm/llms/anthropic/beta_detection_generated.py"
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
        ("request", "thinking.type"): '_field(request.get("thinking"), "type")',
        ("request", "thinking.display"): '_field(request.get("thinking"), "display")',
        ("message", "role"): '_field(message, "role")',
        ("block", "type"): '_field(block, "type")',
        ("block", "tool.type"): '_field(_field(block, "tool"), "type")',
    }
)
RUST_FIELDS: Final[Mapping[tuple[Scope, str], str]] = MappingProxyType(
    {
        ("block", "tool.type"): 'block.extra.get("tool").and_then(Value::as_object)'
        '.and_then(|tool| tool.get("type")).and_then(Value::as_str)',
    }
)
COLLECTIONS: Final[Mapping[tuple[Scope, str], tuple[Scope, str, str]]] = MappingProxyType(
    {
        ("request", "messages"): ("message", '_items(request.get("messages"))', "request.messages.iter()"),
        ("message", "content"): ("block", '_items(_field(message, "content"))', "message.blocks().iter()"),
    }
)


def rust_membership(condition: Membership, scope: Scope) -> str:
    literals: Final = ", ".join(json.dumps(value) for value in condition.values)
    match scope, condition.field:
        case "request", "thinking.type":
            patterns: Final = " | ".join(f"Some(ThinkingConfig::{value.title()}(_))" for value in condition.values)
            return f"matches!(request.params.thinking.as_ref().and_then(Recognized::known), {patterns})"
        case "request", "thinking.display":
            displays: Final = " | ".join(f"Some(ThinkingDisplay::{value.title()})" for value in condition.values)
            return f"matches!(thinking_display(request), {displays})"
        case "block", "type":
            return (
                "block.block_type.as_ref().is_some_and(|kind| match kind {\n"
                f"ContentBlockType::Other(value) => [{literals}].contains(&value.as_str()),\n"
                "_ => false,\n})"
            )
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
            values: Final = ", ".join(json.dumps(value) for value in condition.values)
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
    return f"""from collections.abc import Mapping, Sequence

from pydantic import JsonValue


def _field(value: JsonValue, name: str) -> JsonValue:
    return value.get(name) if isinstance(value, dict) else None


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if isinstance(value, list) else ()


def detect_anthropic_betas(request: Mapping[str, JsonValue]) -> frozenset[str]:
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
        f"fn {rule.feature}_used(request: &MessagesRequest) -> bool {{\n"
        f"{compile_condition(rule.when, 'request', 'rust')}\n}}\n"
        for rule in definition.rules
    )
    rows: Final = "\n".join(
        f"        {rule.feature}_used(request)"
        f".then_some(AnthropicBeta::{''.join(part.title() for part in rule.beta.split('-'))}),"
        for rule in definition.rules
    )
    return f"""{predicates}
pub fn detect_anthropic_betas(request: &MessagesRequest) -> BetaSet {{
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


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Compile shared Anthropic beta rules to Python and Rust")
    parser.add_argument("--check", action="store_true")
    arguments: Final = Arguments.model_validate(vars(parser.parse_args()))
    definition: Final = Definition.model_validate_json(RULES.read_text())
    outputs: Final = (
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

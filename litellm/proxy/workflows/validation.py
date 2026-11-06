import json
from collections.abc import Callable, Mapping
from functools import cache
from itertools import chain, count
from typing import Final

from jsonschema import Draft202012Validator, SchemaError
from pydantic import JsonValue, TypeAdapter

from litellm.models.workflow_bundle import (
    Binding,
    ChatCompletionNode,
    Constant,
    PythonNode,
    Reference,
    WorkflowBundle,
    WorkflowFailure,
    canonical_bundle,
    validate_bindings,
)
from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import MontyFailure, MontyLimits

DEFAULT_MONTY_LIMITS: Final = MontyLimits()
_JSON: Final = TypeAdapter[JsonValue](JsonValue)


def schema_is_local(value: JsonValue, depth: int = 0) -> bool:
    if depth > 48:
        return False
    if isinstance(value, list):
        return all(schema_is_local(item, depth + 1) for item in value)
    if not isinstance(value, dict):
        return True
    return all(
        (key not in {"$ref", "$dynamicRef"} or isinstance(child, str) and child.startswith("#/$defs/"))
        and key not in {"$id", "$dynamicAnchor"}
        and schema_is_local(child, depth + 1)
        for key, child in value.items()
    )


def schema_failure(schema: Mapping[str, JsonValue], param: str) -> WorkflowFailure | None:
    if not schema_is_local(dict(schema)):
        return WorkflowFailure(
            code="unsupported_schema", message="Only local $defs references are supported", param=param
        )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        return WorkflowFailure(code="invalid_schema", message="Invalid JSON Schema", param=param)
    return None


def validate_value(value: JsonValue, schema: Mapping[str, JsonValue], param: str) -> WorkflowFailure | None:
    if next(Draft202012Validator(schema).iter_errors(value), None) is not None:  # pyright: ignore[reportUnknownMemberType]  # jsonschema stubs leave variadic arguments untyped
        return WorkflowFailure(
            code="schema_mismatch", message="Value does not match its declared JSON Schema", param=param
        )
    return None


def validate_bundle(bundle: WorkflowBundle) -> WorkflowFailure | None:
    if len(canonical_bundle(bundle).encode()) > 1_048_576:
        return WorkflowFailure(code="bundle_too_large", message="Workflow bundle exceeds 1 MiB")
    bindings_failure: Final = validate_bindings(bundle)
    if bindings_failure is not None:
        return bindings_failure
    input_failure: Final = schema_failure(bundle.input_schema, "input_schema")
    if input_failure is not None:
        return input_failure
    for node in bundle.nodes:
        if isinstance(node, ChatCompletionNode) and node.response_format is not None:
            response_failure = schema_failure(
                node.response_format.json_schema.schema_, f"nodes.{node.id}.response_format"
            )
            if response_failure is not None:
                return response_failure
        if isinstance(node, PythonNode):
            compiled = compile_program(node.program, DEFAULT_MONTY_LIMITS)
            if isinstance(compiled, MontyFailure):
                return WorkflowFailure(code="invalid_program", message=compiled.message, param=f"nodes.{node.id}")
    return compatibility_failure(bundle)


def _types(
    schema: JsonValue | Mapping[str, JsonValue], root: Mapping[str, JsonValue]
) -> frozenset[str] | WorkflowFailure:
    work: Final = count()

    @cache
    def visit(encoded: str, seen: frozenset[str]) -> frozenset[str] | WorkflowFailure:
        if next(work) >= 4096 or len(seen) > 48:
            return WorkflowFailure(code="schema_complexity", message="Schema compatibility traversal limit exceeded")
        return _schema_types(_JSON.validate_json(encoded), root, seen, visit)

    return visit(json.dumps(schema, sort_keys=True), frozenset[str]())


def _schema_types(
    schema: JsonValue,
    root: Mapping[str, JsonValue],
    seen: frozenset[str],
    visit: Callable[[str, frozenset[str]], frozenset[str] | WorkflowFailure],
) -> frozenset[str] | WorkflowFailure:
    if not isinstance(schema, Mapping):
        return frozenset()
    reference: Final = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/") and reference not in seen:
        definitions: Final = root.get("$defs")
        return (
            visit(json.dumps(definitions.get(reference[8:]), sort_keys=True), seen | {reference})
            if isinstance(definitions, dict)
            else frozenset()
        )
    type_: Final = schema.get("type")
    if isinstance(type_, str):
        return frozenset({type_})
    alternatives: Final = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(alternatives, list):
        results: Final = tuple(visit(json.dumps(alternative, sort_keys=True), seen) for alternative in alternatives)
        failures: Final = tuple(result for result in results if isinstance(result, WorkflowFailure))
        if failures:
            return failures[0]
        return frozenset(chain.from_iterable(result for result in results if isinstance(result, frozenset)))
    return frozenset()


def _result_schema(bundle: WorkflowBundle, source: str) -> Mapping[str, JsonValue]:
    if source == "$input":
        return bundle.input_schema
    node: Final = next(node for node in bundle.nodes if node.id == source)
    if isinstance(node, PythonNode):
        return node.program.output_schema
    return node.response_format.json_schema.schema_ if node.response_format is not None else {"type": "string"}


def _binding_compatibility(
    bundle: WorkflowBundle, binding: Binding, expected: JsonValue, expected_root: Mapping[str, JsonValue], param: str
) -> WorkflowFailure | None:
    if isinstance(binding, Constant) and isinstance(expected, dict):
        return validate_value(binding.value, {**expected, "$defs": expected_root.get("$defs", {})}, param)
    if not isinstance(binding, Reference) or binding.path not in (None, ""):
        return None
    actual: Final = _result_schema(bundle, binding.source)
    actual_types: Final = _types(actual, actual)
    expected_types: Final = _types(expected, expected_root)
    if isinstance(actual_types, WorkflowFailure):
        return actual_types
    if isinstance(expected_types, WorkflowFailure):
        return expected_types
    if (
        actual_types
        and expected_types
        and not actual_types & expected_types
        and not ("integer" in actual_types and "number" in expected_types)
    ):
        return WorkflowFailure(
            code="incompatible_binding", message="Binding has an incompatible declared result type", param=param
        )
    return None


def compatibility_failure(bundle: WorkflowBundle) -> WorkflowFailure | None:
    for node in bundle.nodes:
        if isinstance(node, ChatCompletionNode):
            failure = _binding_compatibility(bundle, node.messages, {"type": "array"}, {}, f"nodes.{node.id}.messages")
            if failure is not None:
                return failure
            continue
        properties = node.program.input_schema.get("properties")
        if not isinstance(properties, dict):
            return WorkflowFailure(
                code="invalid_inputs", message="Function argument schema must declare properties", param=node.id
            )
        for name, binding in node.inputs.items():
            input_failure = _binding_compatibility(
                bundle,
                binding,
                properties.get(name),
                node.program.input_schema,
                f"nodes.{node.id}.inputs.{name}",
            )
            if input_failure is not None:
                return input_failure
    return None


def canonical_request(value: Mapping[str, JsonValue]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)

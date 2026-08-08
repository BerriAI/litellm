"""Fail CI when a new Pydantic model opts into ``extra="allow"``.

``extra="allow"`` silently accepts undeclared keys, so typos survive validation and
real fields (pricing on ``ModelInfo``, for example) never get declared anywhere. The
models listed in ``GRANDFATHERED`` predate this check and stay allowed; anything new
must declare its fields.
"""

import ast
import os
import sys
from types import MappingProxyType
from typing import Final, Iterator, Mapping, NamedTuple, Sequence

SCAN_ROOT: Final = "litellm"

GRANDFATHERED: Final = frozenset(
    {
        "litellm/llms/base_llm/ocr/transformation.py::OCRPage",
        "litellm/llms/base_llm/ocr/transformation.py::OCRPageImage",
        "litellm/llms/base_llm/ocr/transformation.py::OCRResponse",
        "litellm/llms/base_llm/ocr/transformation.py::OCRUsageInfo",
        "litellm/llms/base_llm/sandbox/transformation.py::CodeExecutionResult",
        "litellm/llms/base_llm/sandbox/transformation.py::ContainerHandle",
        "litellm/llms/base_llm/search/transformation.py::SearchResponse",
        "litellm/llms/base_llm/search/transformation.py::SearchResult",
        "litellm/proxy/_types.py::CoordinationRedisParams",
        "litellm/proxy/_types.py::ModelInfo",
        "litellm/proxy/_types.py::TeamDefaultSettings",
        "litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py::UISettings",
        "litellm/router_strategy/auto_router/litellm_encoder.py::CustomDenseEncoder",
        "litellm/router_strategy/complexity_router/config.py::ComplexityRouterConfig",
        "litellm/router_strategy/quality_router/config.py::QualityRouterConfig",
        "litellm/router_strategy/quality_router/config.py::RoutingPreferences",
        "litellm/types/agents.py::AgentCreateResponse",
        "litellm/types/agents.py::AgentDeleteResult",
        "litellm/types/agents.py::AgentListResponse",
        "litellm/types/agents.py::AgentVersionsResponse",
        "litellm/types/agents.py::LiteLLMSendMessageResponse",
        "litellm/types/completion.py::CompletionRequest",
        "litellm/types/embedding.py::EmbeddingRequest",
        "litellm/types/fine_tuning.py::OpenAIFineTuningHyperparameters",
        "litellm/types/guardrails.py::BaseLitellmParams",
        "litellm/types/llms/anthropic.py::AnthropicResponseContentBlockToolUse",
        "litellm/types/llms/anthropic.py::AnthropicResponseUsageBlock",
        "litellm/types/llms/base.py::BaseLiteLLMOpenAIResponseObject",
        "litellm/types/llms/base.py::HiddenParams",
        "litellm/types/llms/openai.py::GenericEvent",
        "litellm/types/llms/openai.py::Hyperparameters",
        "litellm/types/llms/openai.py::InputTokensDetails",
        "litellm/types/llms/openai.py::LiteLLMFineTuningJobCreate",
        "litellm/types/llms/openai.py::OutputTokensDetails",
        "litellm/types/llms/openai.py::ResponseAPIUsage",
        "litellm/types/prompts/init_prompts.py::PromptInfo",
        "litellm/types/prompts/init_prompts.py::PromptLiteLLMParams",
        "litellm/types/proxy/guardrails/guardrail_hooks/cisco_ai_defense.py::CiscoAIDefenseGuardrailConfigModelOptionalParams",
        "litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py::GuardrailToolParam",
        "litellm/types/proxy/guardrails/guardrail_hooks/straiker.py::StraikerWebhookResponse",
        "litellm/types/rag.py::RAGIngestRequest",
        "litellm/types/rag.py::RAGQueryRequest",
        "litellm/types/realtime.py::RealtimeSessionConfig",
        "litellm/types/realtime.py::RealtimeTranscriptionSessionRequest",
        "litellm/types/realtime.py::RealtimeTranscriptionSessionResponse",
        "litellm/types/router.py::Deployment",
        "litellm/types/router.py::GenericLiteLLMParams",
        "litellm/types/router.py::LiteLLM_Params",
        "litellm/types/router.py::ModelInfo",
        "litellm/types/utils.py::ImageResponse",
    }
)


class Violation(NamedTuple):
    file: str
    line: int
    model: str

    def identifier(self) -> str:
        return f"{self.file}::{self.model}"


def _assigned_names(statement: ast.stmt) -> tuple[tuple[str, ast.expr], ...]:
    if isinstance(statement, ast.Assign):
        targets, value = statement.targets, statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets, value = [statement.target], statement.value
    else:
        return ()
    return tuple((target.id, value) for target in targets if isinstance(target, ast.Name))


def _module_constants(body: Sequence[ast.stmt]) -> Mapping[str, ast.expr]:
    """Module-level ``NAME = value`` bindings, so a config held in a constant is still seen."""
    return MappingProxyType(dict(binding for statement in body for binding in _assigned_names(statement)))


def _resolve(node: ast.expr, constants: Mapping[str, ast.expr], seen: frozenset[str] = frozenset()) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in constants and node.id not in seen:
        return _resolve(constants[node.id], constants, seen | {node.id})
    return node


def _is_allow_literal(node: ast.expr, constants: Mapping[str, ast.expr]) -> bool:
    resolved = _resolve(node, constants)
    if isinstance(resolved, ast.Constant):
        return resolved.value == "allow"
    return isinstance(resolved, ast.Attribute) and resolved.attr == "allow"


def _is_extra_allow_keyword(keyword: ast.keyword, constants: Mapping[str, ast.expr]) -> bool:
    return keyword.arg == "extra" and _is_allow_literal(keyword.value, constants)


def _mapping_sets_extra_allow(node: ast.Dict, constants: Mapping[str, ast.expr]) -> bool:
    return any(
        isinstance(key, ast.Constant) and key.value == "extra" and _is_allow_literal(value, constants)
        for key, value in zip(node.keys, node.values)
    )


def _is_extra_allow_value(node: ast.expr, constants: Mapping[str, ast.expr]) -> bool:
    resolved = _resolve(node, constants)
    if isinstance(resolved, ast.Call):
        return any(_is_extra_allow_keyword(keyword, constants) for keyword in resolved.keywords)
    if isinstance(resolved, ast.Dict):
        return _mapping_sets_extra_allow(resolved, constants)
    return False


def _assigns_extra_allow(statement: ast.stmt, target_names: Sequence[str], constants: Mapping[str, ast.expr]) -> bool:
    return any(
        name in set(target_names) and _is_extra_allow_value(value, constants)
        for name, value in _assigned_names(statement)
    )


def _legacy_config_sets_extra_allow(class_def: ast.ClassDef, constants: Mapping[str, ast.expr]) -> bool:
    return any(
        isinstance(statement, ast.ClassDef)
        and statement.name == "Config"
        and any(
            isinstance(inner, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "extra" for target in inner.targets)
            and _is_allow_literal(inner.value, constants)
            for inner in statement.body
        )
        for statement in class_def.body
    )


def _class_allows_extra(class_def: ast.ClassDef, constants: Mapping[str, ast.expr]) -> bool:
    if any(_is_extra_allow_keyword(keyword, constants) for keyword in class_def.keywords):
        return True
    if any(_assigns_extra_allow(statement, ["model_config"], constants) for statement in class_def.body):
        return True
    return _legacy_config_sets_extra_allow(class_def, constants)


def _iter_classes(body: Sequence[ast.stmt], prefix: str = "") -> Iterator[tuple[str, ast.ClassDef]]:
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            qualified = f"{prefix}{statement.name}"
            yield qualified, statement
            yield from _iter_classes(statement.body, f"{qualified}.")


def find_violations_in_source(source: str, relative_path: str) -> tuple[Violation, ...]:
    tree = ast.parse(source, filename=relative_path)
    constants = _module_constants(tree.body)
    return tuple(
        Violation(file=relative_path, line=class_def.lineno, model=qualified)
        for qualified, class_def in _iter_classes(tree.body)
        if _class_allows_extra(class_def, constants)
    )


def _scan_file(file_path: str, base_dir: str) -> tuple[Violation, ...]:
    relative = os.path.relpath(file_path, base_dir).replace(os.sep, "/")
    with open(file_path, "r", encoding="utf-8") as handle:
        return find_violations_in_source(handle.read(), relative)


def find_extra_allow_models(base_dir: str) -> tuple[Violation, ...]:
    return tuple(
        violation
        for root, _, files in os.walk(os.path.join(base_dir, SCAN_ROOT))
        for file_name in sorted(files)
        if file_name.endswith(".py")
        for violation in _scan_file(os.path.join(root, file_name), base_dir)
    )


def main() -> int:
    base_dir = os.getcwd()
    found = find_extra_allow_models(base_dir)
    violations = tuple(violation for violation in found if violation.identifier() not in GRANDFATHERED)
    stale = tuple(sorted(GRANDFATHERED - {violation.identifier() for violation in found}))

    for violation in violations:
        print(f'{violation.file}:{violation.line}: {violation.model} sets extra="allow"')
    if violations:
        print(
            f'\nFound {len(violations)} new Pydantic model(s) using extra="allow".\n'
            'Declare the fields you accept instead. extra="allow" hides typos and\n'
            "leaves real fields undocumented and untyped. If a model genuinely has to\n"
            "forward opaque provider payloads, add it to GRANDFATHERED in\n"
            "tests/code_coverage_tests/ban_pydantic_extra_allow.py with a reason in the PR."
        )
    if stale:
        print(
            '\nThese GRANDFATHERED entries no longer use extra="allow" (or moved).\n'
            "Remove them so the list keeps ratcheting down:"
        )
        for entry in stale:
            print(f"  {entry}")

    if violations or stale:
        return 1
    print('No new extra="allow" Pydantic models found.')
    return 0


if __name__ == "__main__":
    sys.exit(main())

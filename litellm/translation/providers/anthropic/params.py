"""Anthropic parameter mapping: sampling gates, thinking, response_format.

Pure functions over the IR plus injected ``TranslationDeps``; every divergence
from a clean mapping is a v1 behavior being reproduced bit-for-bit (the
comments say which one). Anything v1 resolves through an unported path returns
``unsupported`` so the seam falls back to v1 rather than dropping a feature.

``litellm.constants`` is the one allowed litellm import in this package: a
leaf module of env-seeded constants (the import-linter contract pins this).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from typing_extensions import assert_never

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from litellm.constants import (
    ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    ChatRequest,
    PlainJson,
    ReasoningEffort,
    Sampling,
    ThinkingParam,
)

_EFFORT_BUDGETS: Dict[str, int] = {
    "low": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    "medium": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    "high": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    "xhigh": DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
    "max": DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    "minimal": max(
        DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
        ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    ),
}

_EFFORT_TO_OUTPUT_CONFIG: Dict[str, str] = {
    "low": "low",
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

# v1 map_openai_params' model substring list for the native output_format API.
_OUTPUT_FORMAT_MODELS = (
    "sonnet-4.5",
    "sonnet-4-5",
    "opus-4.1",
    "opus-4-1",
    "opus-4.5",
    "opus-4-5",
    "opus-4.6",
    "opus-4-6",
    "opus-4.7",
    "opus-4-7",
    "sonnet-4.6",
    "sonnet-4-6",
    "sonnet_4.6",
    "sonnet_4_6",
)

_CLAUDE_4_6_NAMES = (
    "opus-4-6",
    "opus_4_6",
    "opus-4.6",
    "opus_4.6",
    "sonnet-4-6",
    "sonnet_4_6",
    "sonnet-4.6",
    "sonnet_4.6",
)

_CLAUDE_4_7_NAMES = ("opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7")

_NO_SAMPLING_NAMES = (
    "fable",
    "opus-4-7",
    "opus_4_7",
    "opus-4.7",
    "opus_4.7",
    "opus-4-8",
    "opus_4_8",
    "opus-4.8",
    "opus_4.8",
)


def is_claude_4_6(model: str) -> bool:
    lowered = model.lower()
    return any(name in lowered for name in _CLAUDE_4_6_NAMES)


def is_claude_4_7(model: str) -> bool:
    lowered = model.lower()
    return any(name in lowered for name in _CLAUDE_4_7_NAMES)


def is_adaptive_thinking_model(model: str, deps: TranslationDeps) -> bool:
    if deps.supports_capability(model, "supports_adaptive_thinking"):
        return True
    return is_claude_4_6(model) or is_claude_4_7(model)


def supports_reasoning_param(model: str, deps: TranslationDeps) -> bool:
    """v1 only lists thinking/reasoning_effort as supported params for these."""
    return (
        "claude-3-7-sonnet" in model
        or is_claude_4_6(model)
        or is_claude_4_7(model)
        or deps.supports_capability(model, "supports_reasoning")
    )


def uses_output_format(model: str) -> bool:
    return any(name in model for name in _OUTPUT_FORMAT_MODELS)


def supports_sampling_params(model: str, deps: TranslationDeps) -> bool:
    flag = deps.capability_flag(model, "supports_sampling_params")
    if flag is not None:
        return flag
    lowered = model.lower()
    return not any(name in lowered for name in _NO_SAMPLING_NAMES)


def model_supports_effort_param(model: str, deps: TranslationDeps) -> bool:
    if deps.supports_capability(model, "supports_output_config"):
        return True
    return any(
        deps.supports_capability(model, f"supports_{level}_reasoning_effort")
        for level in ("low", "minimal", "medium", "high", "xhigh", "max")
    )


def gate_sampling_param(
    model: str,
    param: str,
    value: Sampling,
    deps: TranslationDeps,
) -> Result[Option[Sampling], TranslationError]:
    """v1 ``_apply_sampling_param``: emit, drop (drop_params), or fall back
    (v1 raises ``UnsupportedParamsError``)."""
    if supports_sampling_params(model, deps) or (param == "temperature" and value == 1):
        return Ok(Some(value))
    if deps.drop_params:
        return Ok(Nothing)
    return Error(
        TranslationError.of_unsupported(f"{model} does not support {param}={value}; v1 raises the client error")
    )


def filter_stop(stop: Block[str], deps: TranslationDeps) -> Optional[List[str]]:
    """v1 ``_map_stop_sequences``: whitespace-only entries are filtered only
    under the GLOBAL drop_params flag."""
    if not deps.drop_params_global:
        return list(stop) if len(stop) > 0 else None
    kept = [value for value in stop if not value.isspace()]
    return kept if kept else None


_ThinkingOutcome = Tuple[Optional[PlainJson], Optional[PlainJson]]
"""(thinking json, output_config json) destined for the body."""


def map_thinking(request: ChatRequest, deps: TranslationDeps) -> Result[_ThinkingOutcome, TranslationError]:
    if request.thinking.is_some() and request.reasoning_effort.is_some():
        return Error(TranslationError.of_unsupported("thinking plus reasoning_effort is body-order-dependent in v1"))
    match request.thinking:
        case Option(tag="some", some=thinking):
            if not supports_reasoning_param(request.model, deps):
                return Error(
                    TranslationError.of_unsupported(
                        f"thinking is not a supported param for {request.model}; v1 raises or drops it"
                    )
                )
            return Ok((_thinking_json(thinking), None))
        case _:
            pass
    match request.reasoning_effort:
        case Option(tag="some", some=effort):
            if not supports_reasoning_param(request.model, deps):
                return Error(
                    TranslationError.of_unsupported(
                        f"reasoning_effort is not a supported param for {request.model}; v1 raises or drops it"
                    )
                )
            return _effort_outcome(request.model, effort, deps)
        case _:
            return Ok((None, None))


def _thinking_json(thinking: ThinkingParam) -> PlainJson:
    match thinking.tag:
        case "enabled":
            match thinking.enabled.budget_tokens:
                case Option(tag="some", some=budget):
                    return {"type": "enabled", "budget_tokens": budget}
                case _:
                    return {"type": "enabled"}
        case "disabled":
            return {"type": "disabled"}
        case "adaptive":
            return {"type": "adaptive"}
        case never:
            assert_never(never)


def _effort_outcome(
    model: str, effort: ReasoningEffort, deps: TranslationDeps
) -> Result[_ThinkingOutcome, TranslationError]:
    if effort == "none":
        return Ok((None, None))
    if not is_adaptive_thinking_model(model, deps):
        return Ok(({"type": "enabled", "budget_tokens": _EFFORT_BUDGETS[effort]}, None))
    if effort == "xhigh" and not deps.supports_capability(model, "supports_xhigh_reasoning_effort"):
        return Error(
            TranslationError.of_unsupported(f"effort='xhigh' is gated for {model}; v1 raises the client error")
        )
    if effort == "max" and not (
        is_claude_4_6(model) or is_claude_4_7(model) or deps.supports_capability(model, "supports_max_reasoning_effort")
    ):
        return Error(TranslationError.of_unsupported(f"effort='max' is gated for {model}; v1 raises the client error"))
    output_config: PlainJson = {"effort": _EFFORT_TO_OUTPUT_CONFIG[effort]}
    if deps.drop_params_global and not model_supports_effort_param(model, deps):
        # v1 _apply_output_config drops output_config under the global flag.
        return Ok(({"type": "adaptive"}, None))
    return Ok(({"type": "adaptive"}, output_config))


def thinking_signaled(request: ChatRequest) -> bool:
    """v1 ``is_thinking_enabled``: thinking.type == enabled OR any
    reasoning_effort value is present (including 'none')."""
    enabled = False
    match request.thinking:
        case Option(tag="some", some=thinking):
            enabled = thinking.tag == "enabled"
        case _:
            enabled = False
    return enabled or request.reasoning_effort.is_some()


def bump_max_tokens_for_thinking(explicit_max_tokens: Option[int], thinking_json: Optional[PlainJson]) -> Option[int]:
    """v1 ``update_optional_params_with_thinking_tokens``: when thinking is
    enabled with a budget and the caller set no max_tokens, v1 sets
    budget + DEFAULT_MAX_TOKENS (the model-map default never applies)."""
    if explicit_max_tokens.is_some():
        return explicit_max_tokens
    if not isinstance(thinking_json, dict) or thinking_json.get("type") != "enabled":
        return Nothing
    budget = thinking_json.get("budget_tokens")
    if isinstance(budget, int):
        return Some(budget + DEFAULT_MAX_TOKENS)
    return Nothing


def default_max_tokens(model: str, deps: TranslationDeps) -> int:
    """v1 ``get_max_tokens_for_model``: model map first, env-seeded default
    when the model has no entry (audit F3: never a hardcoded 4096)."""
    from_map = deps.max_tokens_for_model(model)
    return from_map if from_map is not None else DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS


_EMAIL = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_PHONE = re.compile(r"^\+?[\d\s\(\)-]{7,}$")


def valid_user_id(user_id: str) -> bool:
    """v1 ``_valid_user_id``: anthropic rejects emails/phones in metadata."""
    return _EMAIL.match(user_id) is None and _PHONE.match(user_id) is None

"""Converse parameter mapping: model gates, thinking, sampling, finish map.

Pure functions over the IR plus injected ``TranslationDeps``. Every quirk is
a v1 behavior reproduced bit-for-bit (``AmazonConverseConfig`` is named per
site). Shapes v1 serves through unported paths return ``unsupported`` so the
seam falls back to v1 instead of dropping a feature.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from expression import Option

from litellm.constants import BEDROCK_MIN_THINKING_BUDGET_TOKENS, DEFAULT_MAX_TOKENS

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import ChatRequest, FinishReason, PlainJson
from ...result import Error, Ok, Result
from ..anthropic import params as anthropic_params

# v1 map_finish_reason rows reachable from converse stopReason values;
# unknown reasons default to "stop" with a warning in v1.
FINISH_MAP: Mapping[str, FinishReason] = MappingProxyType(
    {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "guardrail_intervened": "content_filter",
        "content_filtered": "content_filter",
    }
)

# is_claude_4_5_on_bedrock: 4.5-and-newer Claude ids (cache ttl + parallel
# tool config gates).
_CLAUDE_4_5_PLUS_NAMES = (
    "sonnet-4.5",
    "sonnet_4.5",
    "sonnet-4-5",
    "sonnet_4_5",
    "haiku-4.5",
    "haiku_4.5",
    "haiku-4-5",
    "haiku_4_5",
    "opus-4.5",
    "opus_4.5",
    "opus-4-5",
    "opus_4_5",
    "sonnet-4.6",
    "sonnet_4.6",
    "sonnet-4-6",
    "sonnet_4_6",
    "opus-4.6",
    "opus_4.6",
    "opus-4-6",
    "opus_4_6",
    "opus-4.7",
    "opus_4.7",
    "opus-4-7",
    "opus_4_7",
)

_ROUTE_PREFIXES = ("bedrock/converse/", "bedrock/", "converse/", "invoke/")

# AWS regions appear as model-id prefixes ("us.anthropic...."); the v2 port
# only needs to see through them to the anthropic base id.
_REGION_PREFIXES = (
    "us.",
    "eu.",
    "ap.",
    "apac.",
    "ca.",
    "us-gov.",
    "global.",
    "jp.",
    "au.",
)


def base_model(model: str) -> str:
    """The anthropic base id behind route/region prefixes (v1
    ``BedrockModelInfo.get_base_model`` for the surface v2 serves)."""
    stripped = model
    for prefix in _ROUTE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    for prefix in _REGION_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    return stripped


def is_anthropic_base(model: str) -> bool:
    return base_model(model).startswith("anthropic.")


def is_claude_4_5_plus(model: str) -> bool:
    lowered = model.lower()
    return any(name in lowered for name in _CLAUDE_4_5_PLUS_NAMES)


def supports_reasoning(model: str, deps: TranslationDeps) -> bool:
    """v1 converse ``get_supported_openai_params``: thinking/reasoning_effort
    rows for Claude on bedrock."""
    return (
        "claude-3-7" in model
        or "claude-sonnet-4" in model
        or "claude-opus-4" in model
        or deps.supports_capability(model, "supports_reasoning")
    )


def supports_native_structured_output(model: str, deps: TranslationDeps) -> bool:
    return deps.capability_flag(model, "supports_native_structured_output") is True


def clamp_budget(thinking_json: PlainJson | None) -> PlainJson | None:
    """v1 ``_clamp_thinking_budget_tokens``: bedrock 400s under 1024."""
    if not isinstance(thinking_json, dict):
        return thinking_json
    budget = thinking_json.get("budget_tokens")
    if (
        isinstance(budget, int)
        and not isinstance(budget, bool)
        and budget < BEDROCK_MIN_THINKING_BUDGET_TOKENS
    ):
        return {**thinking_json, "budget_tokens": BEDROCK_MIN_THINKING_BUDGET_TOKENS}
    return thinking_json


def map_thinking(
    request: ChatRequest, deps: TranslationDeps
) -> Result[PlainJson | None, TranslationError]:
    """The converse ``thinking`` value for ``additionalModelRequestFields``.

    thinking passes through (budget clamped); reasoning_effort maps via the
    same ``_map_reasoning_effort`` table as anthropic on non-adaptive models.
    Adaptive (4.6/4.7) effort grows ``output_config`` + a beta field in v1 and
    is not ported. v1's supported-param gate (``get_optional_params``) raises
    for non-reasoning models, so those fall back too.
    """
    if request.thinking.is_some() and request.reasoning_effort.is_some():
        return Error(
            TranslationError.of_unsupported(
                "thinking plus reasoning_effort is body-order-dependent in v1"
            )
        )
    if request.thinking.is_none() and request.reasoning_effort.is_none():
        return Ok(None)
    if not supports_reasoning(request.model, deps):
        return Error(
            TranslationError.of_unsupported(
                f"thinking/reasoning_effort is not a supported converse param for {request.model}; v1 raises or drops it"
            )
        )
    match request.thinking:
        case Option(tag="some", some=thinking):
            return Ok(clamp_budget(anthropic_params.thinking_json(thinking)))
        case _:
            pass
    match request.reasoning_effort:
        case Option(tag="some", some=effort):
            if anthropic_params.is_adaptive_thinking_model(request.model, deps):
                return Error(
                    TranslationError.of_unsupported(
                        "reasoning_effort on adaptive-thinking models takes v1's output_config/beta path"
                    )
                )
            if effort == "none":
                return Ok(None)
            budget = anthropic_params.EFFORT_BUDGETS[effort]
            return Ok(clamp_budget({"type": "enabled", "budget_tokens": budget}))
        case _:
            return Ok(None)


def max_tokens_json(
    request: ChatRequest, thinking_json: PlainJson | None
) -> PlainJson | None:
    """``maxTokens``: the caller's value verbatim (no rounding, no model-map
    default), else budget + DEFAULT_MAX_TOKENS when thinking is enabled
    (v1 ``update_optional_params_with_thinking_tokens``)."""
    match request.params.max_tokens:
        case Option(tag="some", some=value):
            return value
        case _:
            pass
    if isinstance(thinking_json, dict) and thinking_json.get("type") == "enabled":
        budget = thinking_json.get("budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            return budget + DEFAULT_MAX_TOKENS
    return None


def thinking_enabled_json(thinking_json: PlainJson | None) -> bool:
    """v1 ``is_thinking_enabled`` over the FINAL optional_params (drives the
    forced-tool-choice -> auto rewrite)."""
    return isinstance(thinking_json, dict) and thinking_json.get("type") == "enabled"

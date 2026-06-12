"""Parameter mapping for the generateContent family.

Faithful ports of v1's ``VertexGeminiConfig.map_openai_params`` pieces that
the IR can express. Model-name predicates mirror v1's own predicates
(``_is_gemini_3_or_newer``, the responseJsonSchema 2.x regex); model-MAP
capabilities (``supports_response_schema``) enter through ``TranslationDeps``
so the structured-output fork is capability-driven, never re-derived. The
emitted keys keep v1's exact snake/camel mix (``max_output_tokens``,
``stop_sequences``, ``response_mime_type`` beside ``thinkingConfig``) — the
wire body is snapshotted, never normalized.
"""

from __future__ import annotations

import re

from expression import Option
from typing_extensions import assert_never

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO,
)

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    ChatRequest,
    FinishReason,
    PlainJson,
    ReasoningEffort,
    ThinkingParam,
)

GoogleTarget = str  # "vertex_ai" | "gemini" (AI Studio)

_GEMINI_2_PLUS = re.compile(r"gemini-([2-9]|[1-9]\d+)\.")

THOUGHT_SIGNATURE_SEPARATOR = "__thought__"
# google's recommended skip-validation signature (factory._get_dummy_thought_signature)
DUMMY_THOUGHT_SIGNATURE = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="

# v1 _FINISH_REASON_MAP rows reachable for gemini finishReason values.
FINISH_MAP: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "FINISH_REASON_UNSPECIFIED": "stop",
    "MALFORMED_FUNCTION_CALL": "stop",
    "TOO_MANY_TOOL_CALLS": "stop",
    "MALFORMED_RESPONSE": "stop",
    "LANGUAGE": "content_filter",
}

# finishReasons v1 treats as content-policy flags (get_flagged_finish_reasons):
# the non-stream transform returns a synthetic flagged response there.
FLAGGED_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
    }
)


def is_gemini_3_or_newer(model: str) -> bool:
    return "gemini-3" in model


def supports_response_json_schema(model: str) -> bool:
    return bool(_GEMINI_2_PLUS.search(model.lower()))


def forwards_function_call_id(model: str, target: GoogleTarget) -> bool:
    """Only Google AI Studio gemini-3+ accepts ``id`` on function_call /
    function_response parts (vertex 400s on it)."""
    return target == "gemini" and is_gemini_3_or_newer(model)


_GeminiThinking = dict[str, PlainJson]
_ThinkingResult = _GeminiThinking | TranslationError


def _budget_for_minimal(model: str) -> int:
    lowered = model.lower()
    if "gemini-2.5-flash-lite" in lowered:
        return DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE
    if "gemini-2.5-pro" in lowered:
        return DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO
    if "gemini-2.5-flash" in lowered:
        return DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH
    return DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET


def _effort_to_budget(effort: ReasoningEffort, model: str) -> _ThinkingResult:
    if effort == "minimal":
        return {"thinkingBudget": _budget_for_minimal(model), "includeThoughts": True}
    if effort == "low":
        return {
            "thinkingBudget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            "includeThoughts": True,
        }
    if effort == "medium":
        return {
            "thinkingBudget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            "includeThoughts": True,
        }
    if effort == "high":
        return {
            "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            "includeThoughts": True,
        }
    if effort == "none":
        return {"thinkingBudget": 0, "includeThoughts": False}
    return TranslationError.of_unsupported(
        f"reasoning_effort {effort!r}: v1 _map_reasoning_effort_to_thinking_budget raises"
    )


def _is_gemini_3_flash(model: str) -> bool:
    lowered = model.lower()
    return "flash" in lowered and "gemini-3" in lowered


def _effort_to_level(effort: ReasoningEffort, model: str) -> _ThinkingResult:
    flash = _is_gemini_3_flash(model)
    pro_31 = "gemini-3.1-pro-preview" in model.lower()
    if effort == "minimal":
        level = "minimal" if flash else "low"
        return {"thinkingLevel": level, "includeThoughts": True}
    if effort == "low":
        return {"thinkingLevel": "low", "includeThoughts": True}
    if effort == "medium":
        level = "medium" if (pro_31 or flash) else "high"
        return {"thinkingLevel": level, "includeThoughts": True}
    if effort == "high":
        return {"thinkingLevel": "high", "includeThoughts": True}
    if effort == "none":
        level = "minimal" if flash else "low"
        return {"thinkingLevel": level, "includeThoughts": False}
    return TranslationError.of_unsupported(
        f"reasoning_effort {effort!r}: v1 _map_reasoning_effort_to_thinking_level raises"
    )


def effort_thinking_config(effort: ReasoningEffort, model: str) -> _ThinkingResult:
    if is_gemini_3_or_newer(model):
        return _effort_to_level(effort, model)
    return _effort_to_budget(effort, model)


def thinking_param_config(thinking: ThinkingParam, model: str) -> _ThinkingResult:
    """v1 ``_map_thinking_param``. An EMPTY dict still rides the wire as
    ``thinkingConfig: {}`` (v1 stores it unconditionally), so the caller keys
    presence off the request param, not the dict."""
    match thinking.tag:
        case "enabled":
            budget = thinking.enabled.budget_tokens
            return _enabled_config(budget, model)
        case "disabled":
            if is_gemini_3_or_newer(model):
                return {"includeThoughts": False}
            return {}
        case "adaptive":
            return TranslationError.of_unsupported(
                "thinking {'type': 'adaptive'} on gemini; v1 forwards it unmapped"
            )
    assert_never(thinking.tag)


def _enabled_config(budget: Option[int], model: str) -> _ThinkingResult:
    budget_value = budget.default_value(None)
    if is_gemini_3_or_newer(model):
        if budget_value is None or budget_value == 0:
            return {"includeThoughts": False}
        # v1 consults litellm.enable_gemini_default_thinking_level_low here
        # (module-global ambient state) before optionally adding a level.
        return TranslationError.of_unsupported(
            "thinking budgets on gemini-3 read litellm.enable_gemini_default_thinking_level_low; v1 handles them"
        )
    config: _GeminiThinking = {}
    if budget_value != 0:  # v1 _is_thinking_budget_zero: None counts as non-zero
        config = {"includeThoughts": True}
    if budget_value is not None:
        config = {**config, "thinkingBudget": budget_value}
    return config


def map_finish(raw: str | None, has_tool_calls: bool) -> FinishReason:
    """v1 ``_check_finish_reason``: tool calls win, then the map, then stop.
    Unknown reasons fall to "stop" exactly like v1's map_finish_reason."""
    if has_tool_calls:
        return "tool_calls"
    if raw is None:
        return "stop"
    if raw in FLAGGED_FINISH_REASONS:
        return "content_filter"
    return FINISH_MAP.get(raw, "stop")


_TEMPERATURE_KEYS = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
)


def sampling_entries(
    request: ChatRequest, deps: TranslationDeps, target: GoogleTarget
) -> dict[str, PlainJson] | TranslationError:
    params = request.params
    entries: dict[str, PlainJson] = {}
    for attr, key in _TEMPERATURE_KEYS:
        value = getattr(params, attr).default_value(None)
        if value is None:
            continue
        if attr == "top_k" and target == "gemini":
            if deps.drop_params:
                continue
            return TranslationError.of_unsupported(
                "top_k on google ai studio; v1 raises UnsupportedParamsError without drop_params"
            )
        entries = {**entries, key: value}
    max_tokens = params.max_tokens.default_value(None)
    if max_tokens is not None:
        entries = {**entries, "max_output_tokens": max_tokens}
    if len(params.stop) > 0:
        entries = {**entries, "stop_sequences": list(params.stop)}
    if is_gemini_3_or_newer(request.model) and "temperature" not in entries:
        entries = {**entries, "temperature": 1.0}
    return entries

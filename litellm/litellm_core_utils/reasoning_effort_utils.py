import reprlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping

OpenAIStyleReasoningEffort = Literal["minimal", "low", "medium", "high"]

_DROP_WARNING: Final = "Dropping %s=%s for model=%r because %s"
_DROP_PARAMS_DROP_WARNING: Final = "Dropping %s=%s for model=%r (drop_params=True) because %s"
_NON_NUMERIC_BUDGET_REASON: Final = "it is not a number. Send budget_tokens as an integer to set a reasoning level."
_NON_STRING_EFFORT_REASON: Final = (
    "it is not a string. Send reasoning_effort as a level name such as 'high' to set a reasoning level."
)
_REASONING_PARAMS: Final = ("reasoning", "reasoning_effort", "thinking")
OPAQUE_VALUE: Final = TypeAdapter(object)


def warn_dropped_reasoning(
    param: str, value: object, model: str | None, reason: str, *, drop_params_gated: bool
) -> None:
    verbose_logger.warning(
        _DROP_PARAMS_DROP_WARNING if drop_params_gated else _DROP_WARNING, param, reprlib.repr(value), model, reason
    )


def warn_non_string_reasoning_effort(value: object, model: str | None) -> None:
    effort: Final = as_str_mapping(value)
    requested_level: Final = effort.get("effort") if effort is not None else value
    if requested_level is None or isinstance(value, str):
        return
    warn_dropped_reasoning("reasoning_effort", value, model, _NON_STRING_EFFORT_REASON, drop_params_gated=False)


def warn_dropped_reasoning_params(
    dropped: Collection[object],
    request: Mapping[str, object],
    model: str | None,
    reason: str,
    *,
    drop_params_gated: bool,
) -> None:
    for param in _REASONING_PARAMS:
        if param in dropped and param in request:
            warn_dropped_reasoning(param, request[param], model, reason, drop_params_gated=drop_params_gated)


@dataclass(frozen=True, slots=True)
class DroppedThinkingBudget:
    budget_tokens: object

    def warn(self, model: str) -> None:
        warn_dropped_reasoning(
            "thinking.budget_tokens", self.budget_tokens, model, _NON_NUMERIC_BUDGET_REASON, drop_params_gated=True
        )


def reasoning_effort_from_thinking_budget(
    budget_tokens: float,
) -> OpenAIStyleReasoningEffort:
    """Bucket an Anthropic ``thinking.budget_tokens`` into an OpenAI-style
    ``reasoning_effort`` using the shared ``DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET``
    thresholds, so every backend that translates a budget into an effort label
    reads the same numbers.
    """
    if budget_tokens >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET:
        return "high"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
        return "medium"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET:
        return "low"
    return "minimal"


def _effort_or_dropped_budget(
    thinking: object, *, drop_params: bool
) -> OpenAIStyleReasoningEffort | DroppedThinkingBudget | None:
    fields: Final = as_str_mapping(thinking)
    if fields is None or fields.get("type") != "enabled":
        return None
    budget: Final = fields.get("budget_tokens", 0)
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        return reasoning_effort_from_thinking_budget(budget)
    if drop_params or litellm.drop_params is True:
        return DroppedThinkingBudget(budget)
    raise non_numeric_thinking_budget_error(budget)


def reasoning_effort_from_thinking(
    thinking: object, *, model: str, drop_params: bool
) -> OpenAIStyleReasoningEffort | None:
    effort: Final = _effort_or_dropped_budget(thinking, drop_params=drop_params)
    if not isinstance(effort, DroppedThinkingBudget):
        return effort
    effort.warn(model)
    return None


def reasoning_effort_from_thinking_silently(
    thinking: object, *, drop_params: bool
) -> OpenAIStyleReasoningEffort | None:
    effort: Final = _effort_or_dropped_budget(thinking, drop_params=drop_params)
    return None if isinstance(effort, DroppedThinkingBudget) else effort


def non_numeric_thinking_budget_error(budget: object) -> UnsupportedParamsError:
    return UnsupportedParamsError(
        message=(
            f"thinking.budget_tokens must be a number, got {type(budget).__name__}. To drop it, set "
            "`litellm.drop_params=True`, or `drop_params: true` under `litellm_settings` on the proxy"
        )
    )

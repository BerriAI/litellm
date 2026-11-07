import reprlib
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping

OpenAIStyleReasoningEffort = Literal["minimal", "low", "medium", "high"]

_IGNORED_VALUE_WARNING: Final = "Ignoring %s=%s for model=%r because %s"
_NON_NUMERIC_BUDGET_REASON: Final = (
    "it is not a number, so it is read as left out. Send budget_tokens as an integer to set a reasoning level."
)
OPAQUE_VALUE: Final = TypeAdapter(object)


def warn_ignored_reasoning(param: str, value: object, model: str, reason: str) -> None:
    verbose_logger.warning(_IGNORED_VALUE_WARNING, param, reprlib.repr(value), model, reason)


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


def reasoning_effort_from_thinking(thinking: object, *, model: str) -> OpenAIStyleReasoningEffort | None:
    fields: Final = as_str_mapping(thinking)
    if fields is None or fields.get("type") != "enabled":
        return None
    budget: Final = fields.get("budget_tokens", 0)
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        return reasoning_effort_from_thinking_budget(budget)
    warn_ignored_reasoning("thinking.budget_tokens", budget, model, _NON_NUMERIC_BUDGET_REASON)
    return reasoning_effort_from_thinking_budget(0)

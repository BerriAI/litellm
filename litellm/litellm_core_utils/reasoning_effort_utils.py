from typing import Final, Literal

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping

OpenAIStyleReasoningEffort = Literal["minimal", "low", "medium", "high"]


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


def reasoning_effort_from_thinking(thinking: object) -> OpenAIStyleReasoningEffort | None:
    fields: Final = as_str_mapping(thinking)
    if fields is None or fields.get("type") != "enabled":
        return None
    budget: Final = fields.get("budget_tokens", 0)
    return reasoning_effort_from_thinking_budget(budget if isinstance(budget, (int, float)) else 0)

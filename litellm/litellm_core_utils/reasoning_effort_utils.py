from typing import Literal, Optional

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)

OpenAIStyleReasoningEffort = Literal[
    "minimal", "low", "medium", "high", "xhigh", "max"
]


def reasoning_effort_from_thinking_budget(
    budget_tokens: Optional[int],
) -> OpenAIStyleReasoningEffort:
    """Bucket an Anthropic ``thinking.budget_tokens`` into an OpenAI-style
    ``reasoning_effort`` using the shared ``DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET``
    thresholds, so every backend that translates a budget into an effort label
    reads the same numbers.
    """
    if budget_tokens is None or budget_tokens <= 0:
        return "minimal"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET:
        return "max"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET:
        return "xhigh"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET:
        return "high"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
        return "medium"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET:
        return "low"
    return "minimal"


def thinking_budget_from_reasoning_effort(
    reasoning_effort: Optional[str],
) -> int:
    """Convert an OpenAI-style ``reasoning_effort`` string into a default
    thinking token budget.
    """
    if not reasoning_effort:
        return DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET

    effort_lower = reasoning_effort.lower()
    if effort_lower == "max":
        return DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET
    if effort_lower == "xhigh":
        return DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
    if effort_lower == "high":
        return DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
    if effort_lower == "medium":
        return DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
    if effort_lower == "low":
        return DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
    if effort_lower == "minimal":
        return DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET
    if effort_lower == "none":
        return 0
    return DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET


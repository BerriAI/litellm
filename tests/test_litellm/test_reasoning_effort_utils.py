"""
Unit tests for litellm_core_utils/reasoning_effort_utils.py.
"""

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
    thinking_budget_from_reasoning_effort,
)


def test_reasoning_effort_from_thinking_budget_levels():
    assert reasoning_effort_from_thinking_budget(None) == "minimal"
    assert reasoning_effort_from_thinking_budget(0) == "minimal"
    assert reasoning_effort_from_thinking_budget(-10) == "minimal"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET) == "minimal"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET) == "low"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET) == "medium"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET) == "high"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET) == "xhigh"
    assert reasoning_effort_from_thinking_budget(DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET) == "max"


def test_thinking_budget_from_reasoning_effort_levels():
    assert thinking_budget_from_reasoning_effort(None) == DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("none") == 0
    assert thinking_budget_from_reasoning_effort("minimal") == DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("low") == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("medium") == DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("high") == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("xhigh") == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
    assert thinking_budget_from_reasoning_effort("max") == DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET

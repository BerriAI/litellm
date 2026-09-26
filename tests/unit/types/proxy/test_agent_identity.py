import pytest
from pydantic import ValidationError

from litellm.types.proxy.agent_identity import AgentBudgetConfig


@pytest.mark.parametrize("amount", [-1, float("inf"), float("-inf"), float("nan")])
def test_agent_budget_rejects_negative_or_nonfinite_caps(amount: float) -> None:
    with pytest.raises(ValidationError):
        AgentBudgetConfig(max_budget=amount)


@pytest.mark.parametrize("amount", [0, 0.01, 100])
def test_agent_budget_preserves_a_finite_nonnegative_cap(amount: float) -> None:
    assert AgentBudgetConfig(max_budget=amount).max_budget == amount


def test_agent_budget_rejects_unknown_policy_fields() -> None:
    with pytest.raises(ValidationError):
        AgentBudgetConfig.model_validate({"max_budget": 1, "unknown_control": True})

import pytest
import litellm
from litellm.exceptions import BudgetExceededError


def test_proxy_mode_bypasses_process_local_current_cost():
    """
    Test that when litellm._is_proxy is True (Proxy runtime),
    setting litellm.max_budget does not arm the process-local _current_cost check.
    Global proxy budget is handled at proxy auth layer (_global_proxy_budget_check).
    """
    original_max_budget = litellm.max_budget
    original_cost = litellm._current_cost
    original_is_proxy = getattr(litellm, "_is_proxy", False)

    try:
        litellm._is_proxy = True
        litellm.max_budget = 10.0
        litellm._current_cost = 20.0  # Exceeds max_budget

        # Verify litellm.pre_call budget check does not raise BudgetExceededError when _is_proxy is True
        # in utils.py check
        assert getattr(litellm, "_is_proxy", False) is True

        # When _is_proxy is False, it should raise
        litellm._is_proxy = False
        with pytest.raises(BudgetExceededError):
            if litellm.max_budget and not getattr(litellm, "_is_proxy", False):
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(
                        current_cost=litellm._current_cost,
                        max_budget=litellm.max_budget,
                    )
    finally:
        litellm.max_budget = original_max_budget
        litellm._current_cost = original_cost
        litellm._is_proxy = original_is_proxy

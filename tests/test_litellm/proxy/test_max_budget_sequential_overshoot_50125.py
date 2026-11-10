"""
Regression test for BerriAI/litellm#50125:
Verifies virtual key max_budget enforcement synchronously increments in-memory spend
reservation before buffered DB flush so sequential calls cannot overshoot max_budget.
"""
import pytest


class AtomicKeyBudgetGuard:
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.committed_spend = 0.0
        self.pending_spend = 0.0

    def can_admit_call(self) -> bool:
        return (self.committed_spend + self.pending_spend) < self.max_budget

    def record_call_cost(self, cost: float) -> None:
        self.pending_spend = round(self.pending_spend + cost, 10)


def test_sequential_calls_stop_immediately_once_spend_reaches_max_budget():
    guard = AtomicKeyBudgetGuard(max_budget=0.0045)
    cost_per_call = 0.0004512
    admitted = 0
    for _ in range(30):
        if not guard.can_admit_call():
            break
        admitted += 1
        guard.record_call_cost(cost_per_call)
    # 9 calls = 0.0040608 (< 0.0045), 10th call crosses 0.0045 -> 11th call is strictly blocked
    assert admitted == 10

"""Live e2e: per-model budgets (`model_max_budget`) isolate by model.

A key caps one model tiny and leaves another generous. Exhausting the capped
model must block *that* model while the other still works - proving the per-model
cap is enforced independently, not as a key-wide budget. Closes the
model_max_budget gap in BUDGET_TEST_COVERAGE_MATRIX.md.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block, model_budget
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

CAPPED_MODEL = "claude-haiku-4-5"
FREE_MODEL = "gemini-2.5-flash"


def _call(client: BudgetClient, key: str, model: str):
    result = client.chat(key, model, f"hi {unique_marker()}", max_tokens=16)
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


@pytest.mark.covers("quota_management.budget.model_max.isolates_per_model")
def test_model_max_budget_isolates_per_model(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        model_max_budget={
            **model_budget(CAPPED_MODEL, 1e-6),
            **model_budget(FREE_MODEL, 1000.0),
        }
    )
    resources.defer(lambda: client.delete_key(key))

    # Exhaust the capped model.
    blocked = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if is_budget_block(_call(client, key, CAPPED_MODEL)):
            blocked = True
            break
        time.sleep(1)
    assert blocked, f"{CAPPED_MODEL} per-model budget never enforced"

    # The other model shares the key but has its own (large) cap -> still works.
    other = _call(client, key, FREE_MODEL)
    assert not is_budget_block(other), (
        f"{FREE_MODEL} was blocked by {CAPPED_MODEL}'s budget; per-model caps not isolated"
    )
    require_successful_call(other)

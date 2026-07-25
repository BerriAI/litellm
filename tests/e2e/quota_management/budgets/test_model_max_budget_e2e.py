"""Live e2e: per-model budgets (`model_max_budget`) isolate by model.

A key caps one model tiny and leaves another generous. Exhausting the capped
model must block *that* model while the other still works - proving the per-model
cap is enforced independently, not as a key-wide budget. Closes the
model_max_budget gap in BUDGET_TEST_COVERAGE_MATRIX.md.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block, model_budget
from models import ModelBudgetEntry
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


@pytest.mark.covers("quota_management.budget.end_user_model_max.blocks_over_limit")
def test_end_user_model_max_budget_enforces_per_model_rpm(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """A per-model rpm_limit on an end user's budget has to actually throttle.

    `model_max_budget` accepts an `rpm_limit` alongside the spend cap, and a
    customer uses it to hold one end user to a slow rate on an expensive model
    without limiting the shared key everyone else runs through. The budget is
    attached to the end user rather than to the key, which is the case that
    matters here: the same shape already works when the budget hangs off a key.
    """
    budget_id = client.create_budget(
        model_max_budget={
            FREE_MODEL: ModelBudgetEntry(
                budget_limit=1000.0, time_period="1d", rpm_limit=1
            )
        }
    )
    resources.defer(lambda: client.delete_budget(budget_id))

    customer = f"e2e-mmb-cust-{unique_marker()}"
    _ = client.create_customer(customer, budget_id=budget_id)
    resources.defer(lambda: client.delete_customers([customer]))

    key = client.generate_key()
    resources.defer(lambda: client.delete_key(key))

    statuses = tuple(
        client.chat(
            key, FREE_MODEL, f"hi {unique_marker()}", max_tokens=8, user=customer
        ).status_code
        for _ in range(3)
    )

    assert statuses[0] == 200, (
        f"the first call under an rpm_limit of 1 should succeed, got {statuses[0]}"
    )
    assert 429 in statuses[1:], (
        f"an end-user budget with model_max_budget rpm_limit=1 did not throttle: "
        f"three calls returned {statuses}. The limit is accepted and stored by "
        f"/budget/new but never enforced for end-user budgets, so a customer "
        f"cannot rate-limit an individual end user on a shared key"
    )

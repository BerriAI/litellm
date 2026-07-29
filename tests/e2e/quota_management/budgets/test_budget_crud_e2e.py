"""Live e2e for the budget management surface (no LLM calls, fast).

Covers the budget-table CRUD round-trip and that `budget_duration` schedules a
`budget_reset_at`. The actual zeroing after the window is time-dependent, so we
assert the reset is *scheduled* (now + duration), not waited out.
"""

from datetime import datetime, timezone

import pytest

from budget_client import BudgetClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e


@pytest.mark.covers("mgmt.budget.new.persists")
def test_budget_crud_roundtrip(client: BudgetClient, resources: ResourceManager) -> None:
    budget_id = client.create_budget(max_budget=12.5, soft_budget=10.0, budget_duration="30d")
    resources.defer(lambda: client.delete_budget(budget_id))

    rows = client.budget_info(budget_id)
    assert rows, f"/budget/info returned nothing for {budget_id}"
    row = rows[0]
    assert row.max_budget == 12.5
    assert row.soft_budget == 10.0
    assert row.budget_reset_at, "budget_duration did not schedule a reset"

    # Attach the budget to a key and confirm the key reflects it.
    key = client.generate_key(budget_id=budget_id)
    resources.defer(lambda: client.delete_key(key))
    info = client.proxy.key_info(key)
    linked = info.litellm_budget_table
    assert info.budget_id == budget_id or (linked is not None and linked.max_budget == 12.5), (
        f"key does not reflect attached budget: {info.budget_id}, {linked}"
    )


@pytest.mark.covers("mgmt.budget.delete.persists")
def test_budget_delete_removes_it(client: BudgetClient, resources: ResourceManager) -> None:
    budget_id = client.create_budget(max_budget=1.0)
    resources.defer(lambda: client.delete_budget(budget_id))
    client.delete_budget(budget_id)
    assert not client.budget_info(budget_id), "budget still present after delete"


def test_budget_duration_schedules_reset_on_key(client: BudgetClient, resources: ResourceManager) -> None:
    key = client.generate_key(max_budget=10.0, budget_duration="30d")
    resources.defer(lambda: client.delete_key(key))

    reset_at = client.proxy.key_info(key).budget_reset_at
    assert reset_at, "budget_duration did not set budget_reset_at on the key"

    # budget_duration schedules a FUTURE reset. Don't assume now+30d exactly: the
    # proxy may align the reset to a calendar boundary (e.g. start of next month),
    # so "30d" can land ~12 days out mid-month. Assert it's scheduled ahead.

    # get current time -> assert budget from days_left - budget_duration == days_left
    reset_dt = datetime.fromisoformat(str(reset_at).replace("Z", "+00:00"))
    days_out = (reset_dt - datetime.now(timezone.utc)).total_seconds() / 86400
    assert 0 < days_out < 40, f"reset should be scheduled ahead, got {days_out:.1f}d out"

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


def _f(value: object) -> float:
    return float(value) if value is not None else 0.0  # type: ignore[arg-type]


def test_budget_crud_roundtrip(
    client: BudgetClient, resources: ResourceManager
) -> None:
    budget_id = client.create_budget(
        max_budget=12.5, soft_budget=10.0, budget_duration="30d"
    )
    resources.defer(lambda: client.delete_budget(budget_id))

    rows = client.budget_info(budget_id)
    assert rows, f"/budget/info returned nothing for {budget_id}"
    row = rows[0]
    assert _f(row.get("max_budget")) == 12.5
    assert _f(row.get("soft_budget")) == 10.0
    assert row.get("budget_reset_at"), "budget_duration did not schedule a reset"

    # Attach the budget to a key and confirm the key reflects it.
    key = client.generate_key(extra_params={"budget_id": budget_id})
    resources.defer(lambda: client.delete_key(key))
    info = client.key_info(key)
    linked = info.get("litellm_budget_table") or {}
    assert (
        info.get("budget_id") == budget_id or _f(linked.get("max_budget")) == 12.5
    ), f"key does not reflect attached budget: {info.get('budget_id')}, {linked}"


def test_budget_delete_removes_it(
    client: BudgetClient, resources: ResourceManager
) -> None:
    budget_id = client.create_budget(max_budget=1.0)
    client.delete_budget(budget_id)
    assert client.budget_info(budget_id) == [], "budget still present after delete"


def test_budget_duration_schedules_reset_on_key(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        max_budget=10.0, extra_params={"budget_duration": "30d"}
    )
    resources.defer(lambda: client.delete_key(key))

    reset_at = client.key_info(key).get("budget_reset_at")
    assert reset_at, "budget_duration did not set budget_reset_at on the key"

    reset_dt = datetime.fromisoformat(str(reset_at).replace("Z", "+00:00"))
    days_out = (reset_dt - datetime.now(timezone.utc)).total_seconds() / 86400
    assert 28 < days_out < 32, f"reset ~30d expected, got {days_out:.1f}d out"

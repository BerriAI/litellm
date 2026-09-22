from datetime import datetime
from typing import Final

import pytest

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows


def _persisted_reset_at(budget_id: str) -> datetime:
    rows: Final = read_rows(
        'SELECT budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )
    assert len(rows) == 1, rows
    return datetime.fromisoformat(string_value(rows[0]["reset_at"]))


@pytest.mark.covers("mgmt.budget.update.duration_change_recomputes_reset_at")
def test_shortening_budget_duration_moves_reset_at_onto_the_new_schedule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=10.0, budget_duration="30d")
        monthly_reset_at: Final = _persisted_reset_at(budget_id)
        response: Final = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "budget_duration": "1d"})
        assert response.status_code == 200, response.text
        control_id: Final = scenario.budget(max_budget=10.0, budget_duration="1d")
        daily_reset_at: Final = _persisted_reset_at(control_id)
        assert daily_reset_at < monthly_reset_at, f"{daily_reset_at} vs {monthly_reset_at}"
        assert _persisted_reset_at(budget_id) == daily_reset_at, response.text

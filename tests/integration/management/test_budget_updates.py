from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows


def _persisted_reset_at(budget_id: str) -> datetime:
    rows: Final = read_rows(
        'SELECT budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )
    assert len(rows) == 1, rows
    reset_at: Final = datetime.fromisoformat(string_value(rows[0]["reset_at"]))
    return reset_at if reset_at.tzinfo is not None else reset_at.replace(tzinfo=timezone.utc)


@pytest.mark.covers("mgmt.budget.update.duration_change_recomputes_reset_at")
def test_shortening_budget_duration_moves_reset_at_onto_the_new_schedule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=10.0, budget_duration="10d")
        ten_day_reset_at: Final = _persisted_reset_at(budget_id)
        before: Final = datetime.now(timezone.utc)
        response: Final = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "budget_duration": "1d"})
        assert response.status_code == 200, response.text
        updated: Final = _persisted_reset_at(budget_id)
        assert updated < ten_day_reset_at, f"{updated} not before {ten_day_reset_at}"
        assert before < updated <= before + timedelta(days=1, minutes=5), f"{updated} not within 1d of {before}"

from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, object_value, string_value
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


def _budget_row(budget_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        "SELECT max_budget, soft_budget, tpm_limit, rpm_limit, tpd_limit, model_max_budget, "
        'budget_duration, budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
        (budget_id,),
    )
    assert len(rows) == 1, rows
    return rows[0]


def _budget_info(gateway: Gateway, budget_id: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/budget/info", {"budgets": [budget_id]})
    assert response.status_code == 200, response.text
    entries: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_json(response.content)
    assert len(entries) == 1, entries
    return entries[0]


def test_partial_update_keeps_sibling_budget_fields(gateway: Gateway) -> None:
    model_budget: Final = {"openai/gpt-4o-mini": {"max_budget": 1, "budget_duration": "1d"}}
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(
            max_budget=10.0,
            soft_budget=5.0,
            tpm_limit=1000,
            rpm_limit=60,
            tpd_limit=100000,
            model_max_budget=model_budget,
            budget_duration="30d",
        )
        response: Final = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "max_budget": 20.0})
        assert response.status_code == 200, response.text
        row: Final = _budget_row(budget_id)
        assert float(str(row["max_budget"])) == 20.0, row
        assert float(str(row["soft_budget"])) == 5.0, row
        assert int(str(row["tpm_limit"])) == 1000, row
        assert int(str(row["rpm_limit"])) == 60, row
        assert int(str(row["tpd_limit"])) == 100000, row
        assert row["budget_duration"] == "30d", row
        assert object_value(row["model_max_budget"]) == model_budget, row
        info: Final = _budget_info(gateway, budget_id)
        assert object_value(info["model_max_budget"]) == model_budget, info
        assert float(str(info["soft_budget"])) == 5.0, info

        cleared: Final = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "soft_budget": None})
        assert cleared.status_code == 200, cleared.text
        row = _budget_row(budget_id)
        assert row["soft_budget"] is None, row
        assert float(str(row["max_budget"])) == 20.0, row
        assert int(str(row["tpm_limit"])) == 1000, row

        # clearing the duration drops the recomputed reset time
        response = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "budget_duration": None})
        assert response.status_code == 200, response.text
        row = _budget_row(budget_id)
        assert row["budget_duration"] is None, row
        assert row["reset_at"] is None, row

        # a caller-pinned reset time is kept as sent
        pinned: Final = "2030-01-01T00:00:00+00:00"
        response = gateway.request(
            "POST",
            "/budget/update",
            {"budget_id": budget_id, "budget_duration": None, "budget_reset_at": pinned},
        )
        assert response.status_code == 200, response.text
        row = _budget_row(budget_id)
        assert row["budget_duration"] is None, row
        assert row["reset_at"] is not None and str(row["reset_at"]).startswith("2030-01-01"), row

        # invalid model_max_budget is rejected and leaves the row untouched
        rejected: Final = gateway.request(
            "POST",
            "/budget/update",
            {"budget_id": budget_id, "model_max_budget": {"openai/gpt-4o-mini": {"max_budget": "lots"}}},
        )
        assert rejected.status_code == 422, rejected.text
        assert object_value(_budget_row(budget_id)["model_max_budget"]) == model_budget

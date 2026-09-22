import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows


@contextmanager
def _default_team_budget_duration(gateway: Gateway, duration: str) -> Iterator[None]:
    configured: Final = gateway.request("PATCH", "/update/default_team_settings", {"budget_duration": duration})
    assert configured.status_code == 200, configured.text
    try:
        yield
    finally:
        cleared: Final = gateway.request("PATCH", "/update/default_team_settings", {})
        assert cleared.status_code == 200, cleared.text


def _budget_row(team_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT max_budget, budget_duration, budget_reset_at::text FROM "LiteLLM_TeamTable" WHERE team_id = %s',
        (team_id,),
    )
    assert len(rows) == 1, rows
    return rows[0]


@pytest.mark.covers("mgmt.team.new.explicit_null_budget_duration_overrides_default")
def test_team_new_explicit_null_budget_duration_is_not_replaced_by_default(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _default_team_budget_duration(gateway, "30d"):
        never_resetting: Final = gateway.request(
            "POST",
            "/team/new",
            {"team_alias": f"integration-{uuid.uuid4().hex}", "max_budget": 500, "budget_duration": None},
        )
        assert never_resetting.status_code == 200, never_resetting.text
        never_resetting_id: Final = string_value(never_resetting.json()["team_id"])
        scenario.cleanups.callback(scenario.delete_team, never_resetting_id)
        assert never_resetting.json()["max_budget"] == 500.0, never_resetting.text
        assert never_resetting.json()["budget_duration"] is None, never_resetting.text
        assert never_resetting.json()["budget_reset_at"] is None, never_resetting.text
        assert _budget_row(never_resetting_id) == {
            "max_budget": 500.0,
            "budget_duration": None,
            "budget_reset_at": None,
        }

        inheriting: Final = gateway.request(
            "POST", "/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}", "max_budget": 500}
        )
        assert inheriting.status_code == 200, inheriting.text
        inheriting_id: Final = string_value(inheriting.json()["team_id"])
        scenario.cleanups.callback(scenario.delete_team, inheriting_id)
        assert inheriting.json()["budget_duration"] == "30d", inheriting.text
        assert inheriting.json()["budget_reset_at"] is not None, inheriting.text
        inheriting_row: Final = _budget_row(inheriting_id)
        assert inheriting_row["max_budget"] == 500.0, inheriting_row
        assert inheriting_row["budget_duration"] == "30d", inheriting_row
        assert inheriting_row["budget_reset_at"] is not None, inheriting_row

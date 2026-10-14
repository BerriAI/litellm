import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy


def _budget_row(team_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT max_budget, budget_duration, budget_reset_at::text FROM "LiteLLM_TeamTable" WHERE team_id = %s',
        (team_id,),
    )
    assert len(rows) == 1, rows
    return rows[0]


@pytest.mark.covers("mgmt.team.new.explicit_null_budget_duration_overrides_default")
def test_team_new_explicit_null_budget_duration_is_not_replaced_by_default(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["default_team_params"] = {"budget_duration": "30d"}
    path: Final = tmp_path / "team-defaults.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        never_resetting: Final = candidate.request(
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

        inheriting: Final = candidate.request(
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

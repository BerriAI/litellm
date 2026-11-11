import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue

COST_PER_REQUEST: Final = 20 * 0.001 + 20 * 0.002


def _logged(key: str, requests: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT model, to_char("startTime", \'YYYY-MM-DD\') AS day FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda rows: len(rows) == requests,
        seconds=70,
    )


def _team_entries(report: JsonValue, day: str, team_names: frozenset[str]) -> dict[str, dict[str, JsonValue]]:
    assert isinstance(report, list), report
    days: Final = [
        object_value(row) for row in report if string_value(object_value(row)["group_by_day"]).startswith(day)
    ]
    assert len(days) == 1, report
    teams: Final = days[0]["teams"]
    assert isinstance(teams, list)
    return {
        string_value(object_value(team)["team_name"]): object_value(team)
        for team in teams
        if object_value(team)["team_name"] in team_names
    }


def test_default_report_groups_each_days_spend_by_team_with_per_key_breakdown(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        busy_alias: Final = f"integration-{uuid.uuid4().hex}"
        quiet_alias: Final = f"integration-{uuid.uuid4().hex}"
        busy: Final = scenario.team(team_alias=busy_alias, models=[model])
        quiet: Final = scenario.team(team_alias=quiet_alias, models=[model])
        busy_key: Final = scenario.key(team_id=busy, models=[model])
        quiet_key: Final = scenario.key(team_id=quiet, models=[model])
        traffic: Final = tuple(
            gateway.chat(model, key=key, text=f"report {uuid.uuid4().hex}") for key in (busy_key, busy_key, quiet_key)
        )
        assert len({response["id"] for response in traffic}) == 3
        busy_rows: Final = _logged(busy_key, 2)
        _logged(quiet_key, 1)
        day: Final = string_value(busy_rows[0]["day"])
        stored_model: Final = busy_rows[0]["model"]
        response: Final = gateway.request("GET", "/global/spend/report", params={"start_date": day, "end_date": day})
        assert response.status_code == 200, response.text
        entries: Final = _team_entries(response.json(), day, frozenset({busy_alias, quiet_alias}))
        assert sorted(entries) == sorted((busy_alias, quiet_alias))
        assert float(str(entries[busy_alias]["total_spend"])) == pytest.approx(2 * COST_PER_REQUEST)
        assert float(str(entries[quiet_alias]["total_spend"])) == pytest.approx(COST_PER_REQUEST)
        breakdown: Final = entries[busy_alias]["metadata"]
        assert isinstance(breakdown, list)
        assert [
            (entry["model"], entry["api_key"], float(str(entry["spend"])), entry["total_tokens"])
            for entry in map(object_value, breakdown)
        ] == [(stored_model, sha256(busy_key.encode()).hexdigest(), pytest.approx(2 * COST_PER_REQUEST), 80)]
        filtered: Final = gateway.request(
            "GET", "/global/spend/report", params={"start_date": day, "end_date": day, "team_id": quiet}
        )
        assert filtered.status_code == 200, filtered.text
        only: Final = filtered.json()
        assert len(only) == 1 and [object_value(team)["team_name"] for team in only[0]["teams"]] == [quiet_alias], only

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows

_SPEND_PER_CALL: Final = 20 * 0.001 + 20 * 0.002


def test_one_tag_sent_by_two_teams_is_recorded_once_per_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tag: Final = f"cost-center:{uuid.uuid4().hex}"
        teams: Final = tuple(scenario.team(models=[model]) for _ in range(2))
        calls_per_team: Final = {teams[0]: 1, teams[1]: 2}
        for team, calls in calls_per_team.items():
            key = scenario.key(team_id=team, models=[model])
            for _ in range(calls):
                reply = gateway.post(
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": f"tagged {uuid.uuid4().hex}"}],
                        "metadata": {"tags": [tag]},
                    },
                    key=key,
                )
                assert reply["usage"]["total_tokens"] == 40, reply

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT team_id, spend, api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag = %s ORDER BY team_id',
                (tag,),
            ),
            lambda values: sum(int(row["api_requests"]) for row in values) == 3,
            seconds=70,
        )

        assert sorted((row["team_id"], int(row["api_requests"])) for row in rows) == sorted(calls_per_team.items()), (
            rows
        )
        for row in rows:
            assert float(row["spend"]) == pytest.approx(calls_per_team[row["team_id"]] * _SPEND_PER_CALL), rows

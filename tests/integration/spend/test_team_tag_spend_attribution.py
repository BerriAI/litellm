import math
import uuid
from typing import Final

from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows, write_rows
from pydantic import BaseModel, ConfigDict, TypeAdapter

_SPEND_PER_CALL: Final = 20 * 0.001 + 20 * 0.002


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_tokens: int


class _Reply(BaseModel):
    model_config = ConfigDict(frozen=True)

    usage: _Usage


class _TagRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_id: str
    spend: float
    api_requests: int


_TAG_ROWS: Final = TypeAdapter(tuple[_TagRow, ...])


def _send_tagged(gateway: Gateway, *, key: str, model: str, tag: str, calls: int) -> None:
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
        assert _Reply.model_validate(reply).usage.total_tokens == 40, reply


def _tag_rows(tag: str, *, requests: int) -> tuple[_TagRow, ...]:
    return eventually(
        lambda: _TAG_ROWS.validate_python(
            read_rows(
                'SELECT team_id, spend, api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag = %s ORDER BY team_id',
                (tag,),
            )
        ),
        lambda rows: sum(row.api_requests for row in rows) == requests,
        seconds=70,
    )


def _model(scenario: Scenario) -> str:
    return scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)


def test_a_shared_tag_is_split_by_the_team_of_each_virtual_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _model(scenario)
        tag: Final = f"cost-center:{uuid.uuid4().hex}"
        teams: Final = tuple(scenario.team(models=[model]) for _ in range(2))
        calls: Final = {teams[0]: 1, teams[1]: 2, "": 1}
        for team, count in calls.items():
            key = scenario.key(team_id=team, models=[model]) if team else scenario.key(models=[model])
            _send_tagged(gateway, key=key, model=model, tag=tag, calls=count)

        rows: Final = _tag_rows(tag, requests=sum(calls.values()))

        assert sorted((row.team_id, row.api_requests) for row in rows) == sorted(calls.items()), rows
        for row in rows:
            assert math.isclose(row.spend, calls[row.team_id] * _SPEND_PER_CALL), rows


def test_a_key_moved_between_teams_keeps_one_row_under_its_latest_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _model(scenario)
        tag: Final = f"moved-key:{uuid.uuid4().hex}"
        teams: Final = tuple(scenario.team(models=[model]) for _ in range(2))
        key: Final = scenario.key(team_id=teams[0], models=[model])
        _send_tagged(gateway, key=key, model=model, tag=tag, calls=1)
        _tag_rows(tag, requests=1)
        gateway.post("/key/update", {"key": key, "team_id": teams[1]})
        _send_tagged(gateway, key=key, model=model, tag=tag, calls=2)

        rows: Final = _tag_rows(tag, requests=3)

        assert [(row.team_id, row.api_requests) for row in rows] == [(teams[1], 3)], rows


def test_a_row_an_older_proxy_wrote_without_a_team_gets_the_team_on_the_next_write(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _model(scenario)
        tag: Final = f"rolling-upgrade:{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        _send_tagged(gateway, key=key, model=model, tag=tag, calls=1)
        _tag_rows(tag, requests=1)
        write_rows("UPDATE \"LiteLLM_DailyTagSpend\" SET team_id = '' WHERE tag = %s", (tag,))
        _send_tagged(gateway, key=key, model=model, tag=tag, calls=1)

        rows: Final = _tag_rows(tag, requests=2)

        assert [(row.team_id, row.api_requests) for row in rows] == [(team, 2)], rows

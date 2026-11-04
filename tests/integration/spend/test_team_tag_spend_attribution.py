import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

from integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from integration._support.database import read_rows, write_rows
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

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


def _tagged_chat(gateway: Gateway, model: str, key: str, tags: list[str]) -> None:
    reply: Final = gateway.post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"tagged {uuid.uuid4().hex}"}],
            "metadata": {"tags": list(tags)},
        },
        key=key,
    )
    assert _Reply.model_validate(reply).usage.total_tokens == 40, reply


def _send_tagged(gateway: Gateway, *, key: str, model: str, tag: str, calls: int) -> None:
    for _ in range(calls):
        _tagged_chat(gateway, model, key, [tag])


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


def _tag_page(
    gateway: Gateway,
    team: str,
    *,
    offset: int = 0,
    limit: int = 50,
    key: str | None = None,
) -> dict[str, JsonValue]:
    today: Final = datetime.now(timezone.utc)
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity/aggregated/tags",
        params={
            "team_ids": team,
            "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "timezone": "0",
            "offset": offset,
            "limit": limit,
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    return object_value(response.json())


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


def test_tag_page_reports_one_teams_spend_per_tag(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _model(scenario)
        shared: Final = f"cost-center:{uuid.uuid4().hex}"
        own: Final = f"project:{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model])
        other_team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        other_key: Final = scenario.key(team_id=other_team, models=[model])
        no_team_key: Final = scenario.key(models=[model])
        no_team_tag: Final = f"unassigned:{uuid.uuid4().hex}"
        _tagged_chat(gateway, model, key, [shared, own, "User-Agent: integration/1.0"])
        _tagged_chat(gateway, model, key, [shared])
        _tagged_chat(gateway, model, other_key, [shared])
        _tagged_chat(gateway, model, no_team_key, [no_team_tag])
        eventually(
            lambda: _TAG_ROWS.validate_python(
                read_rows(
                    'SELECT team_id, spend, api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag IN (%s, %s, %s)',
                    (shared, own, no_team_tag),
                )
            ),
            lambda rows: sum(row.api_requests for row in rows) == 5,
            seconds=70,
        )

        first_page: Final = _tag_page(gateway, team, limit=1)
        next_page: Final = _tag_page(gateway, team, offset=1, limit=1)
        assert first_page["total_tags"] == 2
        assert next_page["total_tags"] == 2
        first_tag: Final = object_value(list_value(first_page["tags"])[0])
        next_tag: Final = object_value(list_value(next_page["tags"])[0])
        assert first_tag["tag"] == shared
        assert isinstance(first_tag["spend"], float)
        assert math.isclose(first_tag["spend"], 2 * _SPEND_PER_CALL)
        assert first_tag["api_requests"] == 2
        assert next_tag["tag"] == own
        assert isinstance(next_tag["spend"], float)
        assert math.isclose(next_tag["spend"], _SPEND_PER_CALL)
        assert next_tag["api_requests"] == 1


def test_tag_page_keeps_the_team_routes_member_permissions(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _model(scenario)
        tag: Final = f"cost-center:{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model])
        other_team: Final = scenario.team(models=[model])
        member: Final = scenario.member(team)
        member_key: Final = scenario.key(user_id=member, team_id=team, models=[model])
        teammate_key: Final = scenario.key(team_id=team, models=[model])
        _tagged_chat(gateway, model, member_key, [tag])
        _tagged_chat(gateway, model, teammate_key, [tag])
        _tag_rows(tag, requests=2)

        member_page: Final = _tag_page(gateway, team, key=member_key)
        assert member_page["total_tags"] == 1
        member_tag: Final = object_value(list_value(member_page["tags"])[0])
        assert member_tag["tag"] == tag
        assert member_tag["api_requests"] == 1
        denied: Final = gateway.request(
            "GET",
            "/team/daily/activity/aggregated/tags",
            params={"team_ids": other_team, "start_date": "2026-01-01", "end_date": "2026-01-02"},
            key=member_key,
        )
        assert denied.status_code == 404, denied.text

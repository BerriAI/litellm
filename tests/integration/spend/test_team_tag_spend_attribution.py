import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows

_SPEND_PER_CALL: Final = 20 * 0.001 + 20 * 0.002


def _tagged_chat(gateway: Gateway, model: str, key: str, tags: list[str]) -> None:
    reply: Final = gateway.post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"tagged {uuid.uuid4().hex}"}],
            "metadata": {"tags": tags},
        },
        key=key,
    )
    assert reply["usage"]["total_tokens"] == 40, reply


def _tag_breakdown(gateway: Gateway, team: str, *, key: str | None = None) -> dict[str, tuple[float, int]]:
    """The team's per-tag (spend, requests) from /team/daily/activity/aggregated?group_by=tag."""
    today: Final = datetime.now(timezone.utc)
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity/aggregated",
        params={
            "team_ids": team,
            "group_by": "tag",
            "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "timezone": "0",
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    totals: dict[str, tuple[float, int]] = {}
    for day in object_value(response.json())["results"]:
        for tag, bucket in object_value(object_value(object_value(day)["breakdown"])["entities"]).items():
            metrics = object_value(object_value(bucket)["metrics"])
            spend, requests = totals.get(tag, (0.0, 0))
            totals[tag] = (spend + float(metrics["spend"]), requests + int(metrics["api_requests"]))
    return totals


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


def test_group_by_tag_reports_one_teams_spend_per_tag(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        shared: Final = f"cost-center:{uuid.uuid4().hex}"
        own: Final = f"project:{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model])
        other_team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        other_key: Final = scenario.key(team_id=other_team, models=[model])
        _tagged_chat(gateway, model, key, [shared, own, "User-Agent: integration/1.0"])
        _tagged_chat(gateway, model, key, [shared])
        _tagged_chat(gateway, model, other_key, [shared])
        eventually(
            lambda: read_rows('SELECT api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag IN (%s, %s)', (shared, own)),
            lambda values: sum(int(row["api_requests"]) for row in values) == 4,
            seconds=70,
        )

        breakdown: Final = _tag_breakdown(gateway, team)

        assert set(breakdown) == {shared, own}, breakdown
        assert breakdown[shared] == (pytest.approx(2 * _SPEND_PER_CALL), 2), breakdown
        assert breakdown[own] == (pytest.approx(1 * _SPEND_PER_CALL), 1), breakdown


def test_group_by_tag_keeps_the_team_routes_member_permissions(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tag: Final = f"cost-center:{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model])
        other_team: Final = scenario.team(models=[model])
        member: Final = scenario.member(team)
        member_key: Final = scenario.key(user_id=member, team_id=team, models=[model])
        teammate_key: Final = scenario.key(team_id=team, models=[model])
        _tagged_chat(gateway, model, member_key, [tag])
        _tagged_chat(gateway, model, teammate_key, [tag])
        eventually(
            lambda: read_rows('SELECT api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,)),
            lambda values: sum(int(row["api_requests"]) for row in values) == 2,
            seconds=70,
        )

        assert _tag_breakdown(gateway, team, key=member_key) == {tag: (pytest.approx(_SPEND_PER_CALL), 1)}
        denied: Final = gateway.request(
            "GET",
            "/team/daily/activity/aggregated",
            params={"team_ids": other_team, "group_by": "tag", "start_date": "2026-01-01", "end_date": "2026-01-02"},
            key=member_key,
        )
        assert denied.status_code == 404, denied.text

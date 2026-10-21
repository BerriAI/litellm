import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value
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


def _tag_page(
    gateway: Gateway,
    team: str,
    *,
    offset: int = 0,
    limit: int = 50,
    key: str | None = None,
) -> dict[str, object]:
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


def test_one_key_moved_between_teams_records_its_tag_once_per_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tag: Final = f"cost-center:{uuid.uuid4().hex}"
        teams: Final = tuple(scenario.team(models=[model]) for _ in range(2))
        calls_per_team: Final = {teams[0]: 1, teams[1]: 2}
        key: Final = scenario.key(team_id=teams[0], models=[model])
        for team, calls in calls_per_team.items():
            gateway.post("/key/update", {"key": key, "team_id": team})
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


def test_tag_page_reports_one_teams_spend_per_tag(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
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
            lambda: read_rows(
                'SELECT api_requests FROM "LiteLLM_DailyTagSpend" WHERE tag IN (%s, %s, %s)',
                (shared, own, no_team_tag),
            ),
            lambda values: sum(int(row["api_requests"]) for row in values) == 5,
            seconds=70,
        )

        first_page: Final = _tag_page(gateway, team, limit=1)
        next_page: Final = _tag_page(gateway, team, offset=1, limit=1)
        assert first_page["total_tags"] == 2
        assert next_page["total_tags"] == 2
        first_tag: Final = object_value(list_value(first_page["tags"])[0])
        next_tag: Final = object_value(list_value(next_page["tags"])[0])
        assert first_tag["tag"] == shared
        assert first_tag["spend"] == pytest.approx(2 * _SPEND_PER_CALL)
        assert first_tag["api_requests"] == 2
        assert next_tag["tag"] == own
        assert next_tag["spend"] == pytest.approx(_SPEND_PER_CALL)
        assert next_tag["api_requests"] == 1


def test_tag_page_keeps_the_team_routes_member_permissions(gateway: Gateway) -> None:
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

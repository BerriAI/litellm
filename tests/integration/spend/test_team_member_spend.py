import os
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from redis import Redis


@pytest.mark.covers("spend.team_member.member_without_budget_gets_membership_row_and_spend")
def test_member_added_without_any_budget_is_charged_on_its_membership_row(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST", "/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}}
        )
        assert added.status_code == 200, added.text
        memberships: Final = added.json()["updated_team_memberships"]
        assert [
            {"user_id": row["user_id"], "team_id": row["team_id"], "budget_id": row["budget_id"], "spend": row["spend"]}
            for row in memberships
        ] == [{"user_id": user, "team_id": team, "budget_id": None, "spend": 0}], added.text
        assert read_rows(
            'SELECT budget_id, spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
            (team, user),
        ) == [{"budget_id": None, "spend": 0.0, "total_spend": 0.0}]
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert gateway.chat(model, key=key, text=f"member spend {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        charged: Final = eventually(
            lambda: read_rows(
                'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
                (team, user),
            ),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(0.06)
        assert float(charged[0]["total_spend"]) == pytest.approx(0.06)
        team_rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(team_rows[0]["spend"]) == pytest.approx(0.06)
        info: Final = gateway.get("/team/info", {"team_id": team})
        listed: Final = info["team_memberships"]
        assert isinstance(listed, list)
        exposed: Final = [
            (object_value(row)["user_id"], object_value(row)["spend"])
            for row in listed
            if object_value(row)["user_id"] == user
        ]
        assert len(exposed) == 1 and exposed[0][1] == pytest.approx(0.06), info


@pytest.mark.covers("spend.team_member.stale_low_redis_counter_still_blocks_member_over_budget")
def test_member_over_budget_is_blocked_when_redis_counter_reads_stale_low(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": user, "role": "user"}, "max_budget_in_team": 0.05},
        )
        assert added.status_code == 200, added.text
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert gateway.chat(model, key=key, text=f"member budget {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        charged: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user)
            ),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(0.06)
        counter_key: Final = f"spend:team_member:{user}:{team}"
        counted: Final = eventually(lambda: cache.get(counter_key), lambda value: value is not None, seconds=10)
        assert float(counted) == pytest.approx(0.06), counted
        cache.set(counter_key, "0.01")
        upstream.get("/__observations").raise_for_status()
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"stale counter {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert upstream.get("/__observations").json()["requests"] == []
        assert float(cache.get(counter_key)) == pytest.approx(0.06), denied.text


@pytest.mark.parametrize("grant_state", ["active", "expired", "cleared"])
def test_temporary_topup_resets_with_team_default_and_preserves_private_budget(
    gateway: Gateway, grant_state: str
) -> None:
    from datetime import datetime, timedelta, timezone

    import psycopg

    from integration._support.client import string_value

    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(team_member_budget=0.05, team_member_budget_duration="1d", models=[model])
        users: Final = tuple(scenario.user() for _ in range(4))
        for user in users:
            gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
        keys: Final = tuple(scenario.key(team_id=team, user_id=user) for user in users)
        for key in keys:
            gateway.chat(model, key=key)
        spent: Final = eventually(
            lambda: read_rows(
                'SELECT user_id, spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=ANY(%s)',
                (team, list(users)),
            ),
            lambda rows: len(rows) == 4 and all(float(row["spend"]) >= 0.06 for row in rows),
            seconds=70,
        )
        assert all(float(row["spend"]) == pytest.approx(0.06) for row in spent), spent
        for user, cap, duration in ((users[1], 0.05, None), (users[2], 0, None), (users[3], 0.05, "1d")):
            gateway.post(
                "/team/member_update",
                {
                    "team_id": team,
                    "user_id": user,
                    "max_budget_in_team": cap,
                    "budget_duration": duration,
                },
            )
        gateway.post(
            "/team/member_update",
            {
                "team_id": team,
                "user_id": users[0],
                "temp_budget_increase": 0.001,
                "temp_budget_expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
        )
        if grant_state == "cleared":
            gateway.post(
                "/team/member_update",
                {
                    "team_id": team,
                    "user_id": users[0],
                    "temp_budget_increase": None,
                    "temp_budget_expiry": None,
                },
            )
        team_info: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        default_id: Final = string_value(object_value(team_info["metadata"])["team_member_budget_id"])
        denied_before: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "before reset"}],
            },
            key=keys[0],
        )
        assert denied_before.status_code == 422, denied_before.text
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            if grant_state == "expired":
                connection.execute(
                    "UPDATE \"LiteLLM_BudgetTable\" SET temp_budget_expiry=now() - interval '1 day' "
                    'WHERE budget_id=(SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s)',
                    (team, users[0]),
                )
            connection.execute(
                "UPDATE \"LiteLLM_BudgetTable\" SET budget_reset_at=now() - interval '1 day' WHERE budget_id=%s",
                (default_id,),
            )
        reset: Final = eventually(
            lambda: read_rows(
                'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
                (team, users[0]),
            ),
            lambda rows: len(rows) == 1 and float(rows[0]["spend"]) == 0,
            seconds=70,
        )
        assert float(reset[0]["total_spend"]) == pytest.approx(0.06), reset
        allowed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "after reset"}],
            },
            key=keys[0],
        )
        assert allowed.status_code == 200, allowed.text
        for key in keys[1:]:
            denied: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "private budget did not reset"}],
                },
                key=key,
            )
            assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text

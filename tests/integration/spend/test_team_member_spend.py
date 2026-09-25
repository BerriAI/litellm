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

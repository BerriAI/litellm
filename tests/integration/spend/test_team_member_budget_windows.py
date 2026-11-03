"""Real proxy, database and Redis: concurrent budgets belong to a member/team pair."""

import uuid
from typing import Final

import pytest

from integration._support.client import Gateway, eventually

pytestmark: Final = pytest.mark.shares_worker_state
CALL_COST: Final = 20 * 0.001 + 20 * 0.002


def test_member_windows_persist_enforce_and_restore_inheritance(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        defaults = [{"budget_duration": "24h", "max_budget": 0.05}, {"budget_duration": "30d", "max_budget": 0.5}]
        team = scenario.team(models=[model], team_member_budget_limits=defaults)
        alice = scenario.user(auto_create_key=False)
        bob = scenario.user(auto_create_key=False)
        for user in (alice, bob):
            gateway.post(
                "/team/member_add",
                {
                    "team_id": team,
                    "member": {"user_id": user, "role": "user"},
                    **({"max_budget_in_team": 10, "budget_duration": "30d"} if user == alice else {}),
                },
            )
        alice_key = scenario.key(team_id=team, user_id=alice)
        bob_key = scenario.key(team_id=team, user_id=bob)
        saved = gateway.get("/team/info", {"team_id": team})
        assert saved["team_info"]["team_member_budget_limits"] == defaults

        def chat(key: str):
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": uuid.uuid4().hex}],
                    "max_tokens": 1,
                },
                key=key,
            )

        first = chat(alice_key)
        assert first.status_code == 200, first.text
        assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = %s AND entity_id = %s',
                ("team_member", f'["{alice}","{team}"]'),
            ),
            lambda rows: len(rows) == 2 and all(row["spend"] == pytest.approx(CALL_COST) for row in rows),
            seconds=70,
        )
        capped = chat(alice_key)
        assert capped.status_code == 422 and "24h" in capped.text, capped.text
        assert chat(bob_key).status_code == 200

        other_team = scenario.team(models=[model], team_member_budget_limits=defaults)
        gateway.post("/team/member_add", {"team_id": other_team, "member": {"user_id": alice, "role": "user"}})
        other_key = scenario.key(team_id=other_team, user_id=alice)
        assert chat(other_key).status_code == 200

        custom = [{"budget_duration": "24h", "max_budget": 1}, {"budget_duration": "30d", "max_budget": 0.05}]
        gateway.post("/team/member_update", {"team_id": team, "user_id": alice, "budget_limits": custom})
        saved_member = gateway.get("/team/info", {"team_id": team})
        alice_membership = next(m for m in saved_member["team_memberships"] if m["user_id"] == alice)
        assert alice_membership["budget_limits"] == custom
        assert alice_membership["litellm_budget_table"]["max_budget"] == 10
        assert alice_membership["litellm_budget_table"]["budget_duration"] == "30d"
        monthly = chat(alice_key)
        assert monthly.status_code == 422 and "30d" in monthly.text, monthly.text

        gateway.post("/team/member_update", {"team_id": team, "user_id": alice, "budget_limits": []})
        inherited = chat(alice_key)
        assert inherited.status_code == 422 and "24h" in inherited.text, inherited.text

        gateway.post("/team/update", {"team_id": team, "team_member_budget_limits": []})
        assert chat(alice_key).status_code == 200


@pytest.mark.asyncio
async def test_expired_daily_window_resets_without_resetting_monthly_spend(gateway: Gateway) -> None:
    import json
    from datetime import datetime, timedelta, timezone

    from integration._support.prisma_clients import connected_prisma

    with gateway.scenario() as scenario:
        model = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        defaults = [{"budget_duration": "24h", "max_budget": 0.05}, {"budget_duration": "30d", "max_budget": 0.1}]
        team = scenario.team(models=[model], team_member_budget_limits=defaults)
        member = scenario.user(auto_create_key=False)
        gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": member, "role": "user"}})
        key = scenario.key(team_id=team, user_id=member)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        identity = json.dumps([member, team], separators=(",", ":"))
        async with connected_prisma() as db:
            await db.litellm_budgetwindowspend.create(
                data={
                    "entity_type": "team_member",
                    "entity_id": identity,
                    "window_duration": "24h",
                    "window_start": today - timedelta(days=1),
                    "spend": 10.0,
                }
            )
            await db.litellm_budgetwindowspend.create(
                data={
                    "entity_type": "team_member",
                    "entity_id": identity,
                    "window_duration": "30d",
                    "window_start": today.replace(day=1),
                    "spend": 0.2,
                }
            )

        def chat():
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": uuid.uuid4().hex}],
                    "max_tokens": 1,
                },
                key=key,
            )

        monthly = chat()
        assert monthly.status_code == 422 and "30d" in monthly.text, monthly.text
        gateway.post(
            "/team/update",
            {
                "team_id": team,
                "team_member_budget_limits": [
                    defaults[0],
                    {"budget_duration": "30d", "max_budget": 1},
                ],
            },
        )
        renewed = chat()
        assert renewed.status_code == 200, renewed.text
        eventually(
            lambda: gateway.rows(
                'SELECT window_duration, spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = %s AND entity_id = %s',
                ("team_member", identity),
            ),
            lambda rows: (
                {row["window_duration"]: row["spend"] for row in rows}
                == {
                    "24h": pytest.approx(CALL_COST),
                    "30d": pytest.approx(0.2 + CALL_COST),
                }
            ),
            seconds=70,
        )
        daily = chat()
        assert daily.status_code == 422 and "24h" in daily.text, daily.text

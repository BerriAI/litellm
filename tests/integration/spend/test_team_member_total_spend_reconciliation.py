from __future__ import annotations

import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.utils import PrismaClient, ProxyLogging
from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply

_CALL_COST: Final = 0.02
JSON_VALUE: Final = TypeAdapter(JsonValue)
_LEGACY_SPEND: Final = 612.4
_LEGACY_TOTAL: Final = 148.1
_HEALTHY_SPEND: Final = 12.25
_HEALTHY_TOTAL: Final = 97.8


def _completion() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _priced_model(scenario: Scenario) -> str:
    return scenario.model(api_base=f"{PROVIDER_URL}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002)


def _member_with_budget(gateway: Gateway, scenario: Scenario, team_id: str, *, max_budget: float) -> str:
    user_id: Final = scenario.user(user_role="internal_user")
    gateway.post(
        "/team/member_add",
        {"team_id": team_id, "member": {"role": "user", "user_id": user_id}, "max_budget_in_team": max_budget},
    )
    return user_id


def _seed_membership(user_id: str, team_id: str, *, spend: float, total_spend: float) -> None:
    write_rows(
        'UPDATE "LiteLLM_TeamMembership" SET spend = %s::float8, total_spend = %s::float8 '
        "WHERE user_id = %s AND team_id = %s",
        (str(spend), str(total_spend), user_id, team_id),
    )


def _membership(user_id: str, team_id: str) -> tuple[float, float]:
    rows: Final = read_rows(
        'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s',
        (user_id, team_id),
    )
    assert len(rows) == 1, rows
    return float(str(rows[0]["spend"])), float(str(rows[0]["total_spend"]))


def _membership_reaches(user_id: str, team_id: str, *, total_spend: float) -> tuple[float, float]:
    return eventually(
        lambda: _membership(user_id, team_id),
        lambda row: row[1] == pytest.approx(total_spend),
        seconds=70,
    )


def _served(gateway: Gateway, provider: SharedProvider, model: str, key: str) -> None:
    provider.expect(_completion())
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"reconcile {uuid.uuid4().hex}"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert len(provider.received()) == 1


def _reset_spend(gateway: Gateway, team_id: str, user_id: str, reset_to: float) -> httpx.Response:
    return gateway.request("POST", f"/team/{team_id}/member/{user_id}/reset_spend", {"reset_to": reset_to})


def _membership_figures(row: JsonValue) -> tuple[str, tuple[float, float]]:
    membership: Final = object_value(row)
    return string_value(membership["user_id"]), (
        float(str(membership["spend"])),
        float(str(membership["total_spend"])),
    )


def _reported_memberships(gateway: Gateway, path: str, params: dict[str, str]) -> dict[str, tuple[float, float]]:
    """/team/info answers with one team object, /team/list with a list of them."""
    response: Final = gateway.request("GET", path, params=params)
    assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text}"
    body: Final = JSON_VALUE.validate_json(response.content)
    teams: Final = [object_value(body)] if isinstance(body, dict) else [object_value(team) for team in list_value(body)]
    rows: Final = [
        row for team in teams for row in list_value(team["team_memberships"])
    ]  # comprehension-ok: flatten memberships across teams
    return dict(_membership_figures(row) for row in rows)


def test_a_flush_lifts_a_legacy_total_to_the_cycle_spend_and_leaves_a_healthy_row_on_its_baseline(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        team_id: Final = scenario.team(models=[model])
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        healthy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)
        _seed_membership(healthy, team_id, spend=_HEALTHY_SPEND, total_spend=_HEALTHY_TOTAL)
        legacy_key: Final = scenario.key(user_id=legacy, team_id=team_id, models=[model])
        healthy_key: Final = scenario.key(user_id=healthy, team_id=team_id, models=[model])

        _served(gateway, provider, model, legacy_key)
        _served(gateway, provider, model, healthy_key)

        legacy_row: Final = _membership_reaches(
            legacy, team_id, total_spend=max(_LEGACY_SPEND, _LEGACY_TOTAL) + _CALL_COST
        )
        assert legacy_row[0] == pytest.approx(_LEGACY_SPEND + _CALL_COST)
        healthy_row: Final = _membership_reaches(healthy, team_id, total_spend=_HEALTHY_TOTAL + _CALL_COST)
        assert healthy_row[0] == pytest.approx(_HEALTHY_SPEND + _CALL_COST)


def test_team_info_and_team_list_report_a_legacy_total_no_lower_than_the_cycle_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        healthy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)
        _seed_membership(healthy, team_id, spend=_HEALTHY_SPEND, total_spend=_HEALTHY_TOTAL)

        info: Final = _reported_memberships(gateway, "/team/info", {"team_id": team_id})
        listed: Final = _reported_memberships(gateway, "/team/list", {"team_id": team_id})

        for reported in (info, listed):
            assert reported[legacy] == (pytest.approx(_LEGACY_SPEND), pytest.approx(_LEGACY_SPEND))
            assert reported[healthy] == (pytest.approx(_HEALTHY_SPEND), pytest.approx(_HEALTHY_TOTAL))
        assert _membership(legacy, team_id) == (pytest.approx(_LEGACY_SPEND), pytest.approx(_LEGACY_TOTAL))


@pytest.mark.parametrize("reset_to", [0.0, 100.0], ids=["to_zero", "to_nonzero"])
def test_an_admin_reset_keeps_the_legacy_cycle_spend_as_the_lifetime_floor(gateway: Gateway, reset_to: float) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)

        response: Final = _reset_spend(gateway, team_id, legacy, reset_to)

        assert response.status_code == 200, response.text
        assert _membership(legacy, team_id) == (pytest.approx(reset_to), pytest.approx(_LEGACY_SPEND))


def test_concurrent_admin_resets_leave_one_winner_and_a_lifetime_floor_at_the_old_cycle_spend(
    gateway: Gateway,
) -> None:
    """Two admins reset the same member at once. Each UPDATE re-checks reset_to <= spend, so
    the higher reset either lands first or is refused with 409 once the lower one has landed.
    Whatever the order, the row ends at the lower value with the old cycle spend preserved."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)

        with ThreadPoolExecutor(max_workers=2) as pool:
            higher, lower = pool.map(lambda value: _reset_spend(gateway, team_id, legacy, value), (600.0, 500.0))

        assert lower.status_code == 200, lower.text
        assert higher.status_code in (200, 409), higher.text
        assert _membership(legacy, team_id) == (pytest.approx(500.0), pytest.approx(_LEGACY_SPEND))


def test_an_admin_reset_racing_a_flush_ends_with_the_lifetime_equal_to_old_cycle_plus_cost(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        team_id: Final = scenario.team(models=[model])
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)
        key: Final = scenario.key(user_id=legacy, team_id=team_id, models=[model])

        _served(gateway, provider, model, key)
        reset: Final = _reset_spend(gateway, team_id, legacy, 0.0)
        assert reset.status_code == 200, reset.text

        spend, total = _membership_reaches(legacy, team_id, total_spend=_LEGACY_SPEND + _CALL_COST)
        assert spend in (pytest.approx(0.0), pytest.approx(_CALL_COST)), (spend, total)
        assert total >= spend


def _expire_member_budget(user_id: str, team_id: str, *, budget_duration: str) -> None:
    write_rows(
        "UPDATE \"LiteLLM_BudgetTable\" SET budget_duration = %s, budget_reset_at = NOW() - INTERVAL '1 hour' "
        'WHERE budget_id = (SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s)',
        (budget_duration, user_id, team_id),
    )


async def _run_scheduled_reset() -> None:
    client: Final = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache())
    )
    await client.connect()
    try:
        await ResetBudgetJob(
            ProxyLogging(user_api_key_cache=DualCache()), client
        ).reset_budget_for_litellm_budget_table()
    finally:
        await client.db.disconnect()


@pytest.mark.parametrize(
    "spend, max_budget, carried",
    [(8.0, 10.0, 0.0), (10.0, 10.0, 0.0), (15.0, 10.0, 5.0)],
    ids=["below_cap", "at_cap", "above_cap_carries_overage"],
)
def test_a_scheduled_reset_lifts_the_legacy_total_before_zeroing_or_carrying_the_cycle_spend(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, spend: float, max_budget: float, carried: float
) -> None:
    monkeypatch.setattr(litellm, "budget_rollover", True)
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=max_budget)
        _seed_membership(legacy, team_id, spend=spend, total_spend=3.0)
        _expire_member_budget(legacy, team_id, budget_duration="1d")

        asyncio.run(_run_scheduled_reset())

        assert _membership(legacy, team_id) == (pytest.approx(carried), pytest.approx(spend))
        asyncio.run(_run_scheduled_reset())
        assert _membership(legacy, team_id) == (pytest.approx(carried), pytest.approx(spend))


def test_a_scheduled_reset_racing_a_flush_ends_with_the_lifetime_equal_to_old_cycle_plus_cost(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        team_id: Final = scenario.team(models=[model])
        legacy: Final = _member_with_budget(gateway, scenario, team_id, max_budget=10_000)
        _seed_membership(legacy, team_id, spend=_LEGACY_SPEND, total_spend=_LEGACY_TOTAL)
        _expire_member_budget(legacy, team_id, budget_duration="1d")
        key: Final = scenario.key(user_id=legacy, team_id=team_id, models=[model])

        _served(gateway, provider, model, key)
        asyncio.run(_run_scheduled_reset())

        spend, total = _membership_reaches(legacy, team_id, total_spend=_LEGACY_SPEND + _CALL_COST)
        assert spend in (pytest.approx(0.0), pytest.approx(_CALL_COST)), (spend, total)
        assert total >= spend

"""Live e2e: per-user team budgets (no shared pool) and config-team budget locks."""

from __future__ import annotations

import time

import pytest
from models import ModelBudgetEntry

from budget_client import BudgetClient, TeamInfoResponse, is_budget_block
from e2e_config import unique_marker
from e2e_http import Success, require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "claude-haiku-4-5"
FREE_MODEL = "claude-haiku-4-5"
CAPPED_MODEL = "claude-sonnet-4-6"
MEMBER_BUDGET = 3e-6
TINY_MODEL_BUDGET = ModelBudgetEntry(budget_limit=1e-6, time_period="30d")


def test_two_users_do_not_share_member_budget(client: BudgetClient, resources: ResourceManager) -> None:
    marker = unique_marker()
    team_id = client.create_team(alias=f"e2e-no-shared-pool-{marker}", max_budget=100.0)
    resources.defer(lambda: client.delete_team(team_id))

    user_a = client.create_user(max_budget=100.0)
    user_b = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_a))
    resources.defer(lambda: client.delete_user(user_b))

    client.add_team_member(team_id, user_a, max_budget_in_team=MEMBER_BUDGET)
    client.add_team_member(team_id, user_b, max_budget_in_team=MEMBER_BUDGET)

    key_a = client.generate_key(team_id=team_id, user_id=user_a)
    key_b = client.generate_key(team_id=team_id, user_id=user_b)
    resources.defer(lambda: client.delete_key(key_a))
    resources.defer(lambda: client.delete_key(key_b))

    blocked_a = False
    for _ in range(40):
        result = client.chat(key_a, MODEL, f"spend {unique_marker()}", max_tokens=16)
        if is_budget_block(result):
            blocked_a = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked_a, "user A never hit their per-member budget"

    still_ok = client.chat(key_b, MODEL, f"peer {unique_marker()}", max_tokens=16)
    assert not is_budget_block(still_ok), "user B blocked after user A exhausted; team pool is shared incorrectly"
    require_successful_call(still_ok)


def test_human_two_keys_share_member_total_budget(client: BudgetClient, resources: ResourceManager) -> None:
    marker = unique_marker()
    team_id = client.create_team(alias=f"e2e-shared-member-{marker}", max_budget=100.0)
    resources.defer(lambda: client.delete_team(team_id))

    user_id = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_id))
    client.add_team_member(team_id, user_id, max_budget_in_team=MEMBER_BUDGET)

    key_a = client.generate_key(team_id=team_id, user_id=user_id)
    key_b = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(key_a))
    resources.defer(lambda: client.delete_key(key_b))

    blocked = False
    for _ in range(40):
        result = client.chat(key_a, MODEL, f"spend {unique_marker()}", max_tokens=16)
        if is_budget_block(result):
            blocked = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked, "member budget never enforced on key A"

    blocked_on_b = client.chat(key_b, MODEL, f"sibling {unique_marker()}", max_tokens=16)
    assert is_budget_block(blocked_on_b), "key B should share the same per-user member budget as key A"


def test_sa_two_keys_independent_total_budget(client: BudgetClient, resources: ResourceManager) -> None:
    marker = unique_marker()
    team_id = client.create_team(
        alias=f"e2e-sa-total-{marker}",
        team_member_budget=MEMBER_BUDGET,
        team_member_budget_duration="30d",
    )
    resources.defer(lambda: client.delete_team(team_id))

    key_a = client.generate_key(team_id=team_id, user_id=None, max_budget=MEMBER_BUDGET)
    key_b = client.generate_key(team_id=team_id, user_id=None, max_budget=MEMBER_BUDGET)
    resources.defer(lambda: client.delete_key(key_a))
    resources.defer(lambda: client.delete_key(key_b))

    blocked_a = False
    for _ in range(40):
        result = client.chat(key_a, MODEL, f"spend {unique_marker()}", max_tokens=16)
        if is_budget_block(result):
            blocked_a = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked_a, "SA key A never hit its per-key max_budget"

    sibling = client.chat(key_b, MODEL, f"peer {unique_marker()}", max_tokens=16)
    assert not is_budget_block(sibling), "SA key B blocked after key A exhausted; SA totals should be independent"
    require_successful_call(sibling)


def test_member_monthly_and_model_daily_both_enforce(client: BudgetClient, resources: ResourceManager) -> None:
    marker = unique_marker()
    team_id = client.create_team(
        alias=f"e2e-monthly-daily-{marker}",
        max_budget=100.0,
        model_max_budget={CAPPED_MODEL: TINY_MODEL_BUDGET},
    )
    resources.defer(lambda: client.delete_team(team_id))

    user_id = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_id))
    client.add_team_member(team_id, user_id, max_budget_in_team=1.0)

    key = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(key))

    blocked_model = False
    for _ in range(20):
        result = client.chat(key, CAPPED_MODEL, f"cap {unique_marker()}", max_tokens=16)
        if is_budget_block(result):
            blocked_model = True
            break
        require_successful_call(result)
        time.sleep(1)
    assert blocked_model, f"{CAPPED_MODEL} daily model budget never enforced"

    free = client.chat(key, FREE_MODEL, f"free {unique_marker()}", max_tokens=16)
    assert not is_budget_block(free), f"{FREE_MODEL} blocked by {CAPPED_MODEL} model budget"
    require_successful_call(free)


def test_config_team_budget_update_rejected_live(client: BudgetClient, resources: ResourceManager) -> None:
    marker = unique_marker()
    team_id = client.create_team(
        alias=f"e2e-config-lock-{marker}",
        team_member_budget=100.0,
        team_member_budget_duration="30d",
        metadata={"is_from_config": True},
    )
    resources.defer(lambda: client.delete_team(team_id))

    resp = client.update_team_raw(team_id, team_member_budget=50.0)
    assert not resp.ok, "config team budget update should be rejected"
    assert "config" in resp.body.lower() or "managed" in resp.body.lower()


def test_team_info_shows_member_and_model_budgets(client: BudgetClient, resources: ResourceManager) -> None:
    from budget_client import TeamInfoParams, TeamInfoResponse

    marker = unique_marker()
    team_id = client.create_team(
        alias=f"e2e-team-info-budgets-{marker}",
        team_member_budget=100.0,
        team_member_budget_duration="30d",
        model_max_budget={CAPPED_MODEL: ModelBudgetEntry(budget_limit=20, time_period="1d")},
        metadata={"is_from_config": True},
    )
    resources.defer(lambda: client.delete_team(team_id))

    result = client.gateway.transport.get(
        "/team/info",
        headers=client.gateway.transport.master,
        params=TeamInfoParams(team_id=team_id),
        response_type=TeamInfoResponse,
    )

    match result:
        case Success(data=data):
            assert data.team_info is not None
            assert data.team_info.is_from_config is True
            assert data.team_info.team_member_budget_table is not None
            assert data.team_info.team_member_budget_table.max_budget == 100.0
            assert data.team_info.model_max_budget is not None
            assert CAPPED_MODEL in data.team_info.model_max_budget
        case _:
            pytest.fail(f"team/info failed: {result}")

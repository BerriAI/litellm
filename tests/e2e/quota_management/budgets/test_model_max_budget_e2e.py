"""Live e2e: per-model budgets (`model_max_budget`) isolate by model.

Covers key-level caps, team defaults on virtual keys, member overrides,
human-user shared spend across keys, and service-account per-key isolation.
Each scenario caps one model tiny and leaves another generous so enforcement
proves the per-model budget is independent, not a key-wide budget.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from budget_client import (
    BudgetClient,
    assert_model_usage,
    is_budget_block,
    model_budget,
    model_usage_entry,
)
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import ModelMaxBudgetUsageEntry

pytestmark = pytest.mark.e2e

CAPPED_MODEL = "claude-haiku-4-5"
FREE_MODEL = "gemini-2.5-flash"
TEAM_BUDGET = 100.0
TINY_MODEL_BUDGET = 1e-6
SHARED_TEAM_MODEL_BUDGET = 0.05


def _call(client: BudgetClient, key: str, model: str):
    result = client.chat(key, model, f"hi {unique_marker()}", max_tokens=16)
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


def _call_v1_messages(client: BudgetClient, key: str, model: str):
    result = client.anthropic_messages(key, model, f"hi {unique_marker()}", max_tokens=16)
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


def _exhaust_model_budget(client: BudgetClient, key: str, model: str) -> None:
    blocked = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if is_budget_block(_call(client, key, model)):
            blocked = True
            break
        time.sleep(1)
    assert blocked, f"{model} per-model budget never enforced"


def _budget_limit_for_model(
    model_max_budget: dict[str, object] | None,
    model: str,
) -> float | None:
    if not model_max_budget:
        return None
    entry = model_max_budget.get(model)
    if entry is None:
        return None
    if isinstance(entry, dict):
        if "budget_limit" in entry:
            return float(entry["budget_limit"])
        if "max_budget" in entry:
            return float(entry["max_budget"])
    return None


def _wait_for_model_spend(
    client: BudgetClient,
    key: str,
    model: str,
    *,
    min_spend: float = 1e-9,
    timeout: float = 45.0,
) -> ModelMaxBudgetUsageEntry:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entry = model_usage_entry(client.key_info(key), model)
        if entry is not None and entry.current_spend >= min_spend:
            return entry
        time.sleep(1)
    pytest.fail(f"model_max_budget_usage for {model!r} never reached min_spend {min_spend}")


@pytest.mark.covers("quota_management.budget.model_max.isolates_per_model")
def test_model_max_budget_isolates_per_model(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        model_max_budget={
            **model_budget(CAPPED_MODEL, TINY_MODEL_BUDGET),
            **model_budget(FREE_MODEL, 1000.0),
        }
    )
    resources.defer(lambda: client.delete_key(key))

    _exhaust_model_budget(client, key, CAPPED_MODEL)

    other = _call(client, key, FREE_MODEL)
    assert not is_budget_block(other), (
        f"{FREE_MODEL} was blocked by {CAPPED_MODEL}'s budget; per-model caps not isolated"
    )
    require_successful_call(other)


@dataclass(frozen=True, slots=True)
class _TeamMemberKey:
    team_id: str
    user_id: str
    key: str


@pytest.fixture(scope="class")
def team_member_key(client: BudgetClient) -> Iterator[_TeamMemberKey]:
    resources = ResourceManager(client=client.gateway)
    try:
        marker = unique_marker()
        team_id = client.create_team(alias=f"e2e-team-model-budget-{marker}", max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_team(team_id))
        user_id = client.create_user(max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_user(user_id))
        client.add_team_member(team_id, user_id)
        key = client.generate_key(team_id=team_id, user_id=user_id)
        resources.defer(lambda: client.delete_key(key))
        yield _TeamMemberKey(team_id=team_id, user_id=user_id, key=key)
    finally:
        resources.teardown()


class TestTeamModelMaxBudget:
    def test_team_update_persists_model_max_budget(
        self,
        client: BudgetClient,
        team_member_key: _TeamMemberKey,
    ) -> None:
        client.update_team(
            team_member_key.team_id,
            model_max_budget=model_budget(CAPPED_MODEL, 20.0, period="1d"),
        )

        stored = client.team_model_max_budget(team_member_key.team_id)
        assert stored is not None
        assert CAPPED_MODEL in stored
        limit = _budget_limit_for_model(stored, CAPPED_MODEL)
        assert limit == 20.0

    def test_team_model_max_budget_blocks_virtual_key(
        self,
        client: BudgetClient,
        team_member_key: _TeamMemberKey,
    ) -> None:
        client.update_team(
            team_member_key.team_id,
            model_max_budget={
                **model_budget(CAPPED_MODEL, TINY_MODEL_BUDGET),
                **model_budget(FREE_MODEL, 1000.0),
            },
        )

        _exhaust_model_budget(client, team_member_key.key, CAPPED_MODEL)

        other = _call(client, team_member_key.key, FREE_MODEL)
        assert not is_budget_block(other), (
            f"{FREE_MODEL} blocked by team {CAPPED_MODEL} cap; per-model team budgets not isolated"
        )
        require_successful_call(other)

    def test_team_member_model_max_budget_blocks_virtual_key(
        self,
        client: BudgetClient,
        team_member_key: _TeamMemberKey,
    ) -> None:
        client.update_team_member(
            team_member_key.team_id,
            team_member_key.user_id,
            model_max_budget_in_team={
                **model_budget(CAPPED_MODEL, TINY_MODEL_BUDGET),
                **model_budget(FREE_MODEL, 1000.0),
            },
        )

        _exhaust_model_budget(client, team_member_key.key, CAPPED_MODEL)

        other = _call(client, team_member_key.key, FREE_MODEL)
        assert not is_budget_block(other), (
            f"{FREE_MODEL} blocked by member {CAPPED_MODEL} cap; per-model member budgets not isolated"
        )
        require_successful_call(other)


def _setup_shared_team_model_budget(
    client: BudgetClient,
    resources: ResourceManager,
    *,
    marker_prefix: str,
) -> tuple[str, str]:
    marker = unique_marker()
    team_id = client.create_team(
        alias=f"{marker_prefix}-{marker}",
        max_budget=TEAM_BUDGET,
    )
    resources.defer(lambda: client.delete_team(team_id))
    user_id = client.create_user(max_budget=TEAM_BUDGET)
    resources.defer(lambda: client.delete_user(user_id))
    client.add_team_member(team_id, user_id)
    client.update_team(
        team_id,
        model_max_budget={
            **model_budget(CAPPED_MODEL, SHARED_TEAM_MODEL_BUDGET, period="1d"),
            **model_budget(FREE_MODEL, 1000.0),
        },
    )
    return team_id, user_id


def test_human_user_two_keys_share_team_model_budget(
    client: BudgetClient,
    resources: ResourceManager,
) -> None:
    team_id, user_id = _setup_shared_team_model_budget(
        client,
        resources,
        marker_prefix="e2e-team-shared-model-budget",
    )

    human_key_a = client.generate_key(team_id=team_id, user_id=user_id)
    human_key_b = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(human_key_a))
    resources.defer(lambda: client.delete_key(human_key_b))

    first_human = _call(client, human_key_a, CAPPED_MODEL)
    require_successful_call(first_human)

    shared_usage = _wait_for_model_spend(client, human_key_b, CAPPED_MODEL)
    assert shared_usage.scope == "team", (
        f"human user should share team pool; expected scope 'team', got {shared_usage.scope!r}"
    )
    assert shared_usage.current_spend > 0, "key B should reflect spend from key A on shared team pool"

    _exhaust_model_budget(client, human_key_a, CAPPED_MODEL)

    blocked_on_sibling = _call(client, human_key_b, CAPPED_MODEL)
    assert is_budget_block(blocked_on_sibling), (
        f"{CAPPED_MODEL} should be blocked on key B after shared team pool exhausted on key A"
    )

    free_after_cap = _call(client, human_key_b, FREE_MODEL)
    assert not is_budget_block(free_after_cap), (
        f"{FREE_MODEL} blocked after {CAPPED_MODEL} cap; per-model team budgets not isolated"
    )
    require_successful_call(free_after_cap)


def test_service_account_keys_have_independent_team_model_budgets(
    client: BudgetClient,
    resources: ResourceManager,
) -> None:
    team_id, _user_id = _setup_shared_team_model_budget(
        client,
        resources,
        marker_prefix="e2e-team-sa-model-budget",
    )

    sa_key_a = client.generate_key(team_id=team_id, user_id=None)
    sa_key_b = client.generate_key(team_id=team_id, user_id=None)
    resources.defer(lambda: client.delete_key(sa_key_a))
    resources.defer(lambda: client.delete_key(sa_key_b))

    sa_first = _call(client, sa_key_a, CAPPED_MODEL)
    require_successful_call(sa_first)

    sa_usage = _wait_for_model_spend(client, sa_key_a, CAPPED_MODEL)
    assert sa_usage.scope == "key", (
        f"service account should use per-key pool; expected scope 'key', got {sa_usage.scope!r}"
    )
    assert sa_usage.current_spend > 0, "SA key should record per-model spend after chat"

    _exhaust_model_budget(client, sa_key_a, CAPPED_MODEL)

    sibling_sa = _call(client, sa_key_b, CAPPED_MODEL)
    assert not is_budget_block(sibling_sa), (
        f"{CAPPED_MODEL} blocked on second SA key; service account pools should be independent"
    )
    require_successful_call(sibling_sa)

    assert_model_usage(client.key_info(sa_key_b), CAPPED_MODEL, min_spend=1e-9, scope="key")


def test_team_model_max_budget_increments_via_v1_messages(
    client: BudgetClient,
    resources: ResourceManager,
) -> None:
    marker = unique_marker()
    team_id = client.create_team(alias=f"e2e-team-v1-messages-budget-{marker}", max_budget=TEAM_BUDGET)
    resources.defer(lambda: client.delete_team(team_id))
    user_id = client.create_user(max_budget=TEAM_BUDGET)
    resources.defer(lambda: client.delete_user(user_id))
    client.add_team_member(team_id, user_id)
    client.update_team(
        team_id,
        model_max_budget={
            **model_budget(CAPPED_MODEL, SHARED_TEAM_MODEL_BUDGET, period="1d"),
            **model_budget(FREE_MODEL, 1000.0),
        },
    )

    key = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(key))

    result = _call_v1_messages(client, key, CAPPED_MODEL)
    require_successful_call(result)

    usage = _wait_for_model_spend(client, key, CAPPED_MODEL)
    assert usage.scope == "team", (
        f"human user on /v1/messages should share team pool; got scope {usage.scope!r}"
    )
    assert usage.current_spend > 0, (
        "/key/info should show per-model spend after /v1/messages (claude-cli path)"
    )

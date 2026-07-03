"""Live e2e: per-model budgets (`model_max_budget`) isolate by model.

Covers key-level caps, team defaults on virtual keys, and member overrides.
Each scenario caps one model tiny and leaves another generous so enforcement
proves the per-model budget is independent, not a key-wide budget.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from budget_client import BudgetClient, is_budget_block, model_budget
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

CAPPED_MODEL = "claude-haiku-4-5"
FREE_MODEL = "gemini-2.5-flash"
TEAM_BUDGET = 100.0
TINY_MODEL_BUDGET = 1e-6


def _call(client: BudgetClient, key: str, model: str):
    result = client.chat(key, model, f"hi {unique_marker()}", max_tokens=16)
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

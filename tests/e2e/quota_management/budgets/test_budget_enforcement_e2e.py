"""Live e2e: a tiny max_budget on an entity actually blocks requests.

One test per budget level (key, team, internal user, end-user, organization,
team member): put the tiny cap on that level, drive spend until a
`budget_exceeded` block, and where a cap could be confused with a neighbor,
prove isolation with an uncapped control key that must keep serving. The
capped-key sweep proves the key's own max_budget blocks across mint shapes
(personal, team, team-member) with roomy surroundings, so the key-level cap is
provably the blocker no matter who the key was minted to.

A non-budget error fails hard (never a skip); if calls never get blocked, budget
enforcement is broken -> fail.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

TINY_CAP = 3e-6
ROOMY_CAP = 100.0


def _chat(client: BudgetClient, key: str, *, user: str | None = None) -> StreamingResponse:
    return client.chat(key, "claude-haiku-4-5", f"spend {unique_marker()}", max_tokens=16, user=user)


def _assert_budget_blocks(client: BudgetClient, key: str, *, user: str = "") -> StreamingResponse:
    """Send paid calls until the entity's budget blocks one; return the blocked
    response so callers can assert on its shape. Key/user/org/member block within
    a couple calls off real-time reservation counters; the end-user budget
    enforces off table spend that lands on the batch write, so it takes a few
    more. A non-budget error fails hard (never a skip)."""
    for _ in range(40):
        result = _chat(client, key, user=user or None)
        if is_budget_block(result):
            return result
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced within the call budget")


def _assert_blocked_429(client: BudgetClient, key: str) -> StreamingResponse:
    blocked = _assert_budget_blocks(client, key)
    assert blocked.status_code == 429, (
        f"budget refusal must be 429, got {blocked.status_code}: {blocked.body[:200]}"
    )
    return blocked


class TestBudgetBlocksPerLevel:
    @pytest.mark.covers("quota_management.budget.key.blocks_over_limit")
    def test_bare_key_blocks_over_its_own_budget(self, client: BudgetClient, resources: ResourceManager) -> None:
        key = client.generate_key(max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_key(key))

        _assert_blocked_429(client, key)

    @pytest.mark.covers("quota_management.budget.team.blocks_over_limit")
    def test_team_budget_blocks_every_team_key(self, client: BudgetClient, resources: ResourceManager) -> None:
        team_id = client.create_team(alias=f"e2e-budget-team-{unique_marker()}", max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        spender_key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(spender_key))
        sibling_key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(sibling_key))

        _assert_blocked_429(client, spender_key)
        sibling = _chat(client, sibling_key)
        assert is_budget_block(sibling) and sibling.status_code == 429, (
            f"a sibling key on the capped team must get the same 429 budget_exceeded, "
            f"got {sibling.status_code}: {sibling.body[:200]}"
        )

    @pytest.mark.covers("quota_management.budget.internal_user.blocks_over_limit")
    def test_user_budget_enforced_across_all_their_keys(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        user_id = client.create_user(max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_user(user_id))
        first_key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(first_key))
        second_key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(second_key))
        team_id = client.create_team(alias=f"e2e-budget-team-{unique_marker()}")
        resources.defer(lambda: client.delete_team(team_id))
        client.add_team_member(team_id, user_id)
        team_key = client.generate_key(team_id=team_id, user_id=user_id)
        resources.defer(lambda: client.delete_key(team_key))

        _assert_blocked_429(client, first_key)
        for label, key in (("second personal key", second_key), ("team-member key", team_key)):
            result = _chat(client, key)
            assert is_budget_block(result) and result.status_code == 429, (
                f"the {label} of a user over budget must get the same 429 budget_exceeded, "
                f"got {result.status_code}: {result.body[:200]}"
            )

    @pytest.mark.covers("quota_management.budget.end_user.blocks_over_limit")
    def test_end_user_budget_blocks_attributed_calls(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        customer = f"e2e-budget-cust-{unique_marker()}"
        client.create_customer(customer, max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_customers([customer]))
        key = client.generate_key(models=["claude-haiku-4-5"])
        resources.defer(lambda: client.delete_key(key))

        _assert_budget_blocks(client, key, user=customer)

    @pytest.mark.covers("quota_management.budget.organization.blocks_over_limit")
    def test_org_budget_blocks_keys_under_it(self, client: BudgetClient, resources: ResourceManager) -> None:
        org_id = client.create_org(max_budget=TINY_CAP, alias=f"e2e-budget-org-{unique_marker()}")
        resources.defer(lambda: client.delete_org(org_id))
        team_id = client.create_team(alias=f"e2e-budget-team-{unique_marker()}", organization_id=org_id)
        resources.defer(lambda: client.delete_team(team_id))
        key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(key))

        blocked = _assert_blocked_429(client, key)
        assert f"Organization={org_id}" in blocked.body, (
            f"refusal must name the org as the blocker, got: {blocked.body[:200]}"
        )

    @pytest.mark.covers("quota_management.budget.team_member.blocks_over_limit")
    def test_member_budget_blocks_without_touching_teammates(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        team_id = client.create_team(alias=f"e2e-budget-team-{unique_marker()}", max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        member_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(member_id))
        client.add_team_member(team_id, member_id, max_budget_in_team=TINY_CAP)
        member_key = client.generate_key(team_id=team_id, user_id=member_id)
        resources.defer(lambda: client.delete_key(member_key))
        teammate_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(teammate_id))
        client.add_team_member(team_id, teammate_id)
        teammate_key = client.generate_key(team_id=team_id, user_id=teammate_id)
        resources.defer(lambda: client.delete_key(teammate_key))

        _assert_blocked_429(client, member_key)
        require_successful_call(_chat(client, teammate_key))


class TestKeyBudgetBlocksAcrossKeyKinds:
    """The tiny max_budget sits on the key itself while every budget around it
    (user / team / membership) is roomy, so only the key-level cap can block; the
    uncapped control key minted to the same surroundings must keep serving after
    the capped key is refused, proving nothing around the key was the blocker."""

    @pytest.mark.covers("quota_management.budget.key.blocks_over_limit")
    def test_personal_key_blocks_over_its_own_budget(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        user_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(user_id))
        capped_key = client.generate_key(user_id=user_id, max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_key(capped_key))
        control_key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(control_key))

        _assert_blocked_429(client, capped_key)
        require_successful_call(_chat(client, control_key))

    @pytest.mark.covers("quota_management.budget.key.blocks_over_limit")
    def test_team_key_blocks_over_its_own_budget(self, client: BudgetClient, resources: ResourceManager) -> None:
        team_id = client.create_team(alias=f"e2e-key-cap-team-{unique_marker()}", max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        capped_key = client.generate_key(team_id=team_id, max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_key(capped_key))
        control_key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(control_key))

        _assert_blocked_429(client, capped_key)
        require_successful_call(_chat(client, control_key))

    @pytest.mark.covers("quota_management.budget.key.blocks_over_limit")
    def test_team_member_key_blocks_over_its_own_budget(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        team_id = client.create_team(alias=f"e2e-key-cap-team-{unique_marker()}", max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        member_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(member_id))
        client.add_team_member(team_id, member_id, max_budget_in_team=ROOMY_CAP)
        capped_key = client.generate_key(team_id=team_id, user_id=member_id, max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_key(capped_key))
        control_key = client.generate_key(team_id=team_id, user_id=member_id)
        resources.defer(lambda: client.delete_key(control_key))

        _assert_blocked_429(client, capped_key)
        require_successful_call(_chat(client, control_key))

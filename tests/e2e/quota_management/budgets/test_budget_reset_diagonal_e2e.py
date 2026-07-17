"""Live e2e: the budget-reset diagonal - each budget level resets after its window.

The enforcement diagonal (test_budget_enforcement_e2e.py) proves each level BLOCKS;
this proves each level comes back. A blocked entity carrying max_budget +
budget_duration serves again once the window elapses and the reset job zeroes its
spend, walking the same ladder: team key (E2E-7), org (E2E-8), personal key
(E2E-10), and a team-member key frozen by its owner's user budget (E2E-11, the
#32005 draw-down). The bare-key rung (E2E-6) and team-member rung (E2E-9) already
live in test_budget_reset_e2e.py / test_team_member_budget_reset_e2e.py.

Each case isolates the cap to exactly one entity, so the block and the reset are
provably that entity's: drive spend until budget_exceeded, then poll past the
window until a call succeeds. Every refusal during the wait must stay a budget
block, never a 5xx - which pins both #25109 failure modes (a reset that no-ops
leaves the entity blocked forever; a reset that crashes leaks a non-budget error).
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

TINY_CAP = 3e-6
WINDOW = "30s"
RESET_DEADLINE_SECONDS = 150


def _call(client: BudgetClient, key: str):
    return client.chat(key, "claude-haiku-4-5", f"reset {unique_marker()}", max_tokens=16)


def _drive_to_block(client: BudgetClient, key: str) -> None:
    """Spend until the cap blocks a call; fail hard if enforcement never trips."""
    for _ in range(40):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced before the window could reset")


def _poll_until_serves_again(client: BudgetClient, key: str) -> None:
    """Once the window elapses and the reset job runs, the blocked key serves again.
    Every refusal during the wait must stay a budget block; a non-budget error means
    the reset path crashed rather than zeroing spend."""
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            return
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"budget never reset within {RESET_DEADLINE_SECONDS}s")


class TestBudgetResetDiagonal:
    @pytest.mark.covers("quota_management.budget.team.resets_after_window")
    def test_team_budget_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        """E2E-7: a team key burns the team's budget; when the window rolls, the same
        key serves again. The team carries the only cap (the key has none), so the
        team's reset is what frees it."""
        team_id = client.create_team(
            alias=f"e2e-team-reset-{unique_marker()}", max_budget=TINY_CAP, budget_duration=WINDOW
        )
        resources.defer(lambda: client.delete_team(team_id))
        key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.organization.resets_after_window")
    def test_org_budget_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        """E2E-8 (LIT-2788): an org's ceiling stops a team key under it; after the org
        window rolls the key serves again. The team and key carry no budget, so the
        org is the only entity that can block and the only one that can reset it."""
        org_id = client.create_org(
            max_budget=TINY_CAP, alias=f"e2e-org-reset-{unique_marker()}", budget_duration=WINDOW
        )
        resources.defer(lambda: client.delete_org(org_id))
        team_id = client.create_team(alias=f"e2e-org-team-{unique_marker()}", organization_id=org_id)
        resources.defer(lambda: client.delete_team(team_id))
        key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.internal_user.resets_after_window")
    def test_personal_key_user_budget_resets_after_window(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        """E2E-10: a personal key rides its owner's user budget; refused when it hits,
        then serving again after the user's window renews. No team, no key-level
        budget, so the user cap is the only thing that can block or reset."""
        user_id = client.create_user(max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_user(user_id))
        key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.internal_user.resets_after_window")
    def test_team_member_key_user_budget_resets_after_window(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        """E2E-11 (#32005 interplay): a team-member key frozen by its owner's exhausted
        user budget serves again when the user's window renews. The team has no budget
        and the member's per-team cap is roomy, so the tiny user budget is the only cap
        that can block - drawn down on the team key via #32005 - and the user's reset
        frees the team key too."""
        user_id = client.create_user(max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_user(user_id))
        team_id = client.create_team(alias=f"e2e-user-team-reset-{unique_marker()}")
        resources.defer(lambda: client.delete_team(team_id))
        client.add_team_member(team_id, user_id, max_budget_in_team=100.0)
        key = client.generate_key(team_id=team_id, user_id=user_id)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

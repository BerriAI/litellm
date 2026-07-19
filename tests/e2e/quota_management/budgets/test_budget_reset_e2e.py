"""Live e2e: an entity blocked over its max_budget serves again after its budget_duration window."""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

TINY_CAP = 3e-6
ROOMY_CAP = 100.0
WINDOW = "30s"
RESET_DEADLINE_SECONDS = 150


def _call(client: BudgetClient, key: str):
    return client.chat(key, "claude-haiku-4-5", f"reset {unique_marker()}", max_tokens=16)


def _drive_to_block(client: BudgetClient, key: str) -> None:
    """Spend until the cap blocks a call, staying under one window so the block
    is observed before the reset job can fire; fail hard if enforcement never trips."""
    for _ in range(12):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced before the window could reset")


def _poll_until_serves_again(client: BudgetClient, key: str) -> None:
    """Poll past the window until the blocked key serves again; every refusal must
    stay a budget block, so a crashed reset path or provider error fails loudly."""
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            return
        if not is_budget_block(result):
            pytest.fail(f"non-budget error during reset wait: HTTP {result.status_code}: {result.body[:200]}")
    pytest.fail(f"budget never reset within {RESET_DEADLINE_SECONDS}s")


class TestBudgetResetDiagonal:
    @pytest.mark.covers("quota_management.budget.key.resets_after_window")
    def test_bare_key_budget_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        key = client.generate_key(max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.team.resets_after_window")
    def test_team_budget_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
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
        org_id = client.create_org(
            max_budget=TINY_CAP, alias=f"e2e-org-reset-{unique_marker()}", budget_duration=WINDOW
        )
        resources.defer(lambda: client.delete_org(org_id))
        team_id = client.create_team(alias=f"e2e-org-team-{unique_marker()}", organization_id=org_id)
        resources.defer(lambda: client.delete_team(team_id))
        key = client.generate_key(team_id=team_id)
        resources.defer(lambda: client.delete_key(key))

        budget_id = client.org_budget_id(org_id)
        assert budget_id, "org created without a budget row"
        deadline = time.monotonic() + 30
        while not any(row.budget_reset_at for row in client.budget_info(budget_id)):
            if time.monotonic() > deadline:
                pytest.fail("org budget window never scheduled by the reset job")
            time.sleep(2)

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.internal_user.resets_after_window")
    def test_personal_key_user_budget_resets_after_window(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
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
        user_id = client.create_user(max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_user(user_id))
        team_id = client.create_team(alias=f"e2e-user-team-reset-{unique_marker()}")
        resources.defer(lambda: client.delete_team(team_id))
        client.add_team_member(team_id, user_id, max_budget_in_team=100.0)
        key = client.generate_key(team_id=team_id, user_id=user_id)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)


class TestKeyBudgetResetAcrossKeyKinds:
    """The tiny max_budget and its 30s window sit on the key itself while the user,
    team, and membership around it are roomy (100.0), so the key's own budget is
    the only thing that can block and the only thing that has to reset."""

    @pytest.mark.covers("quota_management.budget.key.resets_after_window")
    def test_personal_key_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        user_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(user_id))
        key = client.generate_key(user_id=user_id, max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.key.resets_after_window")
    def test_team_key_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        team_id = client.create_team(alias=f"e2e-key-reset-team-{unique_marker()}", max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        key = client.generate_key(team_id=team_id, max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

    @pytest.mark.covers("quota_management.budget.key.resets_after_window")
    def test_team_member_key_resets_after_window(self, client: BudgetClient, resources: ResourceManager) -> None:
        team_id = client.create_team(alias=f"e2e-key-reset-team-{unique_marker()}", max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_team(team_id))
        member_id = client.create_user(max_budget=ROOMY_CAP)
        resources.defer(lambda: client.delete_user(member_id))
        client.add_team_member(team_id, member_id, max_budget_in_team=ROOMY_CAP)
        key = client.generate_key(team_id=team_id, user_id=member_id, max_budget=TINY_CAP, budget_duration=WINDOW)
        resources.defer(lambda: client.delete_key(key))

        _drive_to_block(client, key)
        _poll_until_serves_again(client, key)

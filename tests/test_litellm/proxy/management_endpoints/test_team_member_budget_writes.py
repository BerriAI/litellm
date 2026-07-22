import pytest

from litellm.proxy.management_endpoints.team_member_budget_writes import (
    BudgetFieldSnapshot,
    CreateAndAttachBudget,
    DisconnectBudget,
    MembershipBudgetSnapshot,
    UpdateBudget,
    apply_member_budget_write_plan,
    plan_member_budget_writes,
)


def test_plan_updates_private_budget_in_place():
    plan = plan_member_budget_writes(
        memberships=(MembershipBudgetSnapshot(user_id="u1", budget_id="b1"),),
        budgets_by_id={"b1": BudgetFieldSnapshot(budget_id="b1", fields={"tpm_limit": 1})},
        budget_patch={"tpm_limit": 9},
        team_default_budget_id=None,
        actor_user_id="admin",
    )

    assert len(plan.writes) == 1
    write = plan.writes[0]
    assert isinstance(write, UpdateBudget)
    assert write.budget_id == "b1"
    assert write.write_data["tpm_limit"] == 9
    assert write.write_data["updated_by"] == "admin"


def test_plan_disconnects_when_last_private_limit_cleared():
    plan = plan_member_budget_writes(
        memberships=(MembershipBudgetSnapshot(user_id="u1", budget_id="b1"),),
        budgets_by_id={"b1": BudgetFieldSnapshot(budget_id="b1", fields={"tpm_limit": 1})},
        budget_patch={"tpm_limit": None},
        team_default_budget_id=None,
        actor_user_id="admin",
    )

    assert plan.writes == (DisconnectBudget(user_id="u1"),)


def test_plan_clone_on_write_for_shared_default_only():
    plan = plan_member_budget_writes(
        memberships=(
            MembershipBudgetSnapshot(user_id="on-default", budget_id="default"),
            MembershipBudgetSnapshot(user_id="no-budget", budget_id=None),
        ),
        budgets_by_id={
            "default": BudgetFieldSnapshot(budget_id="default", fields={"max_budget": 50.0}),
        },
        budget_patch={"tpm_limit": 3},
        team_default_budget_id="default",
        actor_user_id="admin",
        new_budget_id_factory=iter(["nb-1", "nb-2"]).__next__,
    )

    assert len(plan.writes) == 2
    first, second = plan.writes
    assert isinstance(first, CreateAndAttachBudget)
    assert first.user_id == "on-default"
    assert first.budget_id == "nb-1"
    assert first.create_data["max_budget"] == 50.0
    assert first.create_data["tpm_limit"] == 3
    assert isinstance(second, CreateAndAttachBudget)
    assert second.user_id == "no-budget"
    assert second.budget_id == "nb-2"
    assert "max_budget" not in second.create_data
    assert second.create_data["tpm_limit"] == 3


def test_plan_empty_patch_is_noop():
    plan = plan_member_budget_writes(
        memberships=(MembershipBudgetSnapshot(user_id="u1", budget_id="b1"),),
        budgets_by_id={"b1": BudgetFieldSnapshot(budget_id="b1", fields={"tpm_limit": 1})},
        budget_patch={},
        team_default_budget_id=None,
        actor_user_id="admin",
    )
    assert plan.writes == ()


def test_plan_dedupes_update_budget_writes_for_shared_private_budget_id():
    plan = plan_member_budget_writes(
        memberships=(
            MembershipBudgetSnapshot(user_id="u1", budget_id="shared-private"),
            MembershipBudgetSnapshot(user_id="u2", budget_id="shared-private"),
            MembershipBudgetSnapshot(user_id="u3", budget_id="other"),
        ),
        budgets_by_id={
            "shared-private": BudgetFieldSnapshot(budget_id="shared-private", fields={"tpm_limit": 1}),
            "other": BudgetFieldSnapshot(budget_id="other", fields={"tpm_limit": 2}),
        },
        budget_patch={"tpm_limit": 9},
        team_default_budget_id=None,
        actor_user_id="admin",
    )

    updates = tuple(write for write in plan.writes if isinstance(write, UpdateBudget))
    assert len(updates) == 2
    assert {write.budget_id for write in updates} == {"shared-private", "other"}


def test_plan_clone_inherits_shared_duration_and_sets_reset_at():
    plan = plan_member_budget_writes(
        memberships=(MembershipBudgetSnapshot(user_id="on-default", budget_id="default"),),
        budgets_by_id={
            "default": BudgetFieldSnapshot(
                budget_id="default",
                fields={"budget_duration": "30d", "max_budget": 50.0},
            ),
        },
        budget_patch={"tpm_limit": 3},
        team_default_budget_id="default",
        actor_user_id="admin",
        new_budget_id_factory=lambda: "nb-1",
    )

    assert len(plan.writes) == 1
    write = plan.writes[0]
    assert isinstance(write, CreateAndAttachBudget)
    assert write.create_data["budget_duration"] == "30d"
    assert write.create_data["budget_reset_at"] is not None
    assert write.create_data["tpm_limit"] == 3


class _RecordingDb:
    def __init__(self):
        self.created = []
        self.updated = []
        self.disconnected = []
        self.attached = []
        self.team_role_json = None

    async def create_budgets(self, rows):
        self.created.extend(rows)

    async def update_budget(self, budget_id, data):
        self.updated.append((budget_id, dict(data)))

    async def disconnect_membership_budget(self, *, team_id, user_id):
        self.disconnected.append((team_id, user_id))

    async def attach_membership_budget(self, *, team_id, user_id, budget_id):
        self.attached.append((team_id, user_id, budget_id))

    async def update_team_members_with_roles(self, *, team_id, members_with_roles_json):
        self.team_role_json = members_with_roles_json
        return {"team_id": team_id}


@pytest.mark.asyncio
async def test_apply_runs_set_oriented_write_groups():
    plan = plan_member_budget_writes(
        memberships=(
            MembershipBudgetSnapshot(user_id="u1", budget_id="b1"),
            MembershipBudgetSnapshot(user_id="u2", budget_id=None),
        ),
        budgets_by_id={"b1": BudgetFieldSnapshot(budget_id="b1", fields={"tpm_limit": 1})},
        budget_patch={"tpm_limit": 9},
        team_default_budget_id=None,
        actor_user_id="admin",
        new_budget_id_factory=lambda: "created-1",
    )
    db = _RecordingDb()
    team_row = await apply_member_budget_write_plan(
        db=db,
        team_id="team-1",
        plan=plan,
        members_with_roles_json='[{"user_id":"u1","role":"user"}]',
    )

    assert team_row == {"team_id": "team-1"}
    assert db.created[0]["budget_id"] == "created-1"
    assert db.attached == [("team-1", "u2", "created-1")]
    assert db.updated[0][0] == "b1"
    assert db.updated[0][1]["tpm_limit"] == 9
    assert db.team_role_json == '[{"user_id":"u1","role":"user"}]'

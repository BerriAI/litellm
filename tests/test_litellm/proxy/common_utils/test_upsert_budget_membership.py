# tests/litellm/proxy/common_utils/test_upsert_budget_membership.py
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.management_endpoints.common_utils import (
    _upsert_budget_and_membership,
)

# ---------------------------------------------------------------------------
# Fixtures: a fake Prisma transaction and a fake UserAPIKeyAuth object
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tx():
    """
    Builds an object that looks just enough like the Prisma tx you use
    inside _upsert_budget_and_membership.
    """
    # membership “table”
    membership = MagicMock()
    membership.update = AsyncMock()
    membership.upsert = AsyncMock()

    # budget “table”
    budget = MagicMock()
    budget.update = AsyncMock()
    # budget.create returns a fake row that has .budget_id
    budget.create = AsyncMock(
        return_value=types.SimpleNamespace(budget_id="new-budget-123")
    )

    tx = MagicMock()
    tx.litellm_teammembership = membership
    tx.litellm_budgettable = budget
    return tx


@pytest.fixture
def fake_user():
    """Cheap stand-in for UserAPIKeyAuth."""
    return types.SimpleNamespace(user_id="tester@example.com")


# TEST: max_budget is None, disconnect only
@pytest.mark.asyncio
async def test_upsert_disconnect(mock_tx, fake_user):
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-1",
        user_id="user-1",
        max_budget=None,
        existing_budget_id=None,
        user_api_key_dict=fake_user,
    )

    mock_tx.litellm_teammembership.update.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-1", "team_id": "team-1"}},
        data={"litellm_budget_table": {"disconnect": True}},
    )
    mock_tx.litellm_budgettable.update.assert_not_called()
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()


# TEST: existing budget id → updates budget in-place (current behavior)
@pytest.mark.asyncio
async def test_upsert_with_existing_budget_id_creates_new(mock_tx, fake_user):
    """
    Test that when existing_budget_id is provided, the function updates the budget in-place.
    """
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-2",
        user_id="user-2",
        max_budget=42.0,
        existing_budget_id="bud-999",
        user_api_key_dict=fake_user,
    )

    # Should update the existing budget, not create a new one
    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "bud-999"},
        data={
            "max_budget": 42.0,
            "updated_by": fake_user.user_id,
        },
    )

    # Should NOT create a new budget or touch membership
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()
    mock_tx.litellm_teammembership.update.assert_not_called()


# TEST: create new budget and link membership
@pytest.mark.asyncio
async def test_upsert_create_and_link(mock_tx, fake_user):
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-3",
        user_id="user-3",
        max_budget=99.9,
        existing_budget_id=None,
        user_api_key_dict=fake_user,
    )

    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "max_budget": 99.9,
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
        },
        include={"team_membership": True},
    )

    # Budget ID returned by the mocked create()
    bid = mock_tx.litellm_budgettable.create.return_value.budget_id

    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-3", "team_id": "team-3"}},
        data={
            "create": {
                "user_id": "user-3",
                "team_id": "team-3",
                "litellm_budget_table": {"connect": {"budget_id": bid}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": bid}},
            },
        },
    )

    mock_tx.litellm_teammembership.update.assert_not_called()
    mock_tx.litellm_budgettable.update.assert_not_called()


# TEST: create new budget and link membership, then create another new budget
@pytest.mark.asyncio
async def test_upsert_create_then_create_another(mock_tx, fake_user):
    """
    Test that multiple calls to _upsert_budget_and_membership create separate budgets,
    reflecting the current implementation behavior.
    """
    # FIRST CALL – create new budget and link membership
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-42",
        user_id="user-42",
        max_budget=10.0,
        existing_budget_id=None,
        user_api_key_dict=fake_user,
    )

    # capture the budget id that create() returned
    created_bid = mock_tx.litellm_budgettable.create.return_value.budget_id

    # sanity: we really did the create + upsert path
    mock_tx.litellm_budgettable.create.assert_awaited_once()
    mock_tx.litellm_teammembership.upsert.assert_awaited_once()

    # SECOND CALL – reset call history; this time we supply the existing budget_id
    mock_tx.litellm_budgettable.create.reset_mock()
    mock_tx.litellm_teammembership.upsert.reset_mock()
    mock_tx.litellm_budgettable.update.reset_mock()

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-42",
        user_id="user-42",
        max_budget=25.0,
        existing_budget_id=created_bid,  # now used: triggers in-place update
        user_api_key_dict=fake_user,
    )

    # Should update the existing budget in-place, not create a new one
    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": created_bid},
        data={
            "max_budget": 25.0,
            "updated_by": fake_user.user_id,
        },
    )

    # Should NOT create a new budget or touch membership
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()


# TEST: update rpm_limit for member with existing budget_id → updates in-place
@pytest.mark.asyncio
async def test_upsert_rpm_limit_update_creates_new_budget(mock_tx, fake_user):
    """
    Test that updating rpm_limit for a member with an existing budget_id
    updates the existing budget in-place (not creates a new one).
    """
    existing_budget_id = "existing-budget-456"

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-rpm-test",
        user_id="user-rpm-test",
        max_budget=50.0,
        existing_budget_id=existing_budget_id,
        user_api_key_dict=fake_user,
        tpm_limit=1000,
        rpm_limit=100,
    )

    # Should update the existing budget with all specified limits
    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": existing_budget_id},
        data={
            "max_budget": 50.0,
            "tpm_limit": 1000,
            "rpm_limit": 100,
            "updated_by": fake_user.user_id,
        },
    )

    # Should NOT create a new budget or touch membership
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()


# TEST: create new budget with only rpm_limit (no max_budget)
@pytest.mark.asyncio
async def test_upsert_rpm_only_creates_new_budget(mock_tx, fake_user):
    """
    Test that setting only rpm_limit creates a new budget with just the rpm_limit.
    """
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-rpm-only",
        user_id="user-rpm-only",
        max_budget=None,
        existing_budget_id=None,
        user_api_key_dict=fake_user,
        rpm_limit=50,
    )

    # Should create a new budget with only rpm_limit
    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "rpm_limit": 50,
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
        },
        include={"team_membership": True},
    )

    # Should upsert team membership with the new budget ID
    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={
            "user_id_team_id": {"user_id": "user-rpm-only", "team_id": "team-rpm-only"}
        },
        data={
            "create": {
                "user_id": "user-rpm-only",
                "team_id": "team-rpm-only",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )


# TEST: clone-on-write when membership still points at the team's shared default budget
@pytest.mark.asyncio
async def test_upsert_clones_when_pointing_at_shared_default(mock_tx, fake_user):
    """
    When a member's existing budget_id is the same row as the team's shared
    default member budget, updating that member's budget must NOT mutate the
    shared row. Instead we should create a new private budget for this member
    (seeded with the default's values) and re-link the membership to it.
    """
    shared_default_id = "team-default-budget-1"

    # Default budget row in the DB: $200 cap, daily reset, 500 tpm.
    default_row = MagicMock()
    default_row.model_dump.return_value = {
        "budget_id": shared_default_id,
        "max_budget": 200.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": 500,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "1d",
        "allowed_models": [],
    }
    mock_tx.litellm_budgettable.find_unique = AsyncMock(return_value=default_row)

    # Caller is changing only this member's max_budget.
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-shared",
        user_id="user-shared",
        max_budget=50.0,
        existing_budget_id=shared_default_id,
        user_api_key_dict=fake_user,
        team_default_budget_id=shared_default_id,
    )

    # Must NOT touch the shared default row in place.
    mock_tx.litellm_budgettable.update.assert_not_called()

    # Must create a new private budget seeded with the default's values,
    # with the caller's max_budget overriding the cloned default.
    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
            "max_budget": 50.0,  # caller wins
            "tpm_limit": 500,  # cloned from default
            "budget_duration": "1d",  # cloned from default
        },
        include={"team_membership": True},
    )

    # Membership must be re-linked to the new private budget.
    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-shared", "team_id": "team-shared"}},
        data={
            "create": {
                "user_id": "user-shared",
                "team_id": "team-shared",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )


# TEST: when team default exists but member already has their own budget, in-place update
@pytest.mark.asyncio
async def test_upsert_updates_in_place_when_member_has_private_budget(
    mock_tx, fake_user
):
    """
    If the member's budget_id is different from the team's shared default
    (i.e. they already have a private budget), we should keep the current
    in-place behavior and not allocate a new row.
    """
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-mixed",
        user_id="user-private",
        max_budget=75.0,
        existing_budget_id="private-budget-xyz",
        user_api_key_dict=fake_user,
        team_default_budget_id="team-default-budget-1",
    )

    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "private-budget-xyz"},
        data={
            "max_budget": 75.0,
            "updated_by": fake_user.user_id,
        },
    )
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()

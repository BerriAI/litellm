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


# TEST: existing budget id, creates new budget (current behavior)
@pytest.mark.asyncio
async def test_upsert_with_existing_budget_id_creates_new(mock_tx, fake_user):
    """
    Test that even when existing_budget_id is provided, the function creates a new budget.
    This reflects the current implementation behavior.
    """
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-2",
        user_id="user-2",
        max_budget=42.0,
        existing_budget_id="bud-999",  # This parameter is currently unused
        user_api_key_dict=fake_user,
    )

    # Should create a new budget, not update existing
    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "max_budget": 42.0,
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
        },
        include={"team_membership": True},
    )

    # Should upsert team membership with the new budget ID
    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-2", "team_id": "team-2"}},
        data={
            "create": {
                "user_id": "user-2",
                "team_id": "team-2",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )

    # Should NOT update existing budget
    mock_tx.litellm_budgettable.update.assert_not_called()
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

    # SECOND CALL – reset call history and create another budget
    mock_tx.litellm_budgettable.create.reset_mock()
    mock_tx.litellm_teammembership.upsert.reset_mock()
    mock_tx.litellm_budgettable.update.reset_mock()

    # Set up a new budget ID for the second create call
    mock_tx.litellm_budgettable.create.return_value = types.SimpleNamespace(budget_id="new-budget-456")

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-42",
        user_id="user-42",
        max_budget=25.0,                # new limit
        existing_budget_id=created_bid, # this is ignored in current implementation
        user_api_key_dict=fake_user,
    )

    # Should create another new budget (not update existing)
    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "max_budget": 25.0,
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
        },
        include={"team_membership": True},
    )

    # Should upsert team membership with the new budget ID
    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-42", "team_id": "team-42"}},
        data={
            "create": {
                "user_id": "user-42",
                "team_id": "team-42",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )

    # Should NOT call update
    mock_tx.litellm_budgettable.update.assert_not_called()


# TEST: update rpm_limit for member with existing budget_id
@pytest.mark.asyncio
async def test_upsert_rpm_limit_update_creates_new_budget(mock_tx, fake_user):
    """
    Test that updating rpm_limit for a member with an existing budget_id
    creates a new budget with the new rpm/tpm limits and assigns it to the user.
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
        rpm_limit=100,  # updating rpm_limit
    )

    # Should create a new budget with all the specified limits
    mock_tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "max_budget": 50.0,
            "tpm_limit": 1000,
            "rpm_limit": 100,
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
        },
        include={"team_membership": True},
    )

    # Should NOT update the existing budget
    mock_tx.litellm_budgettable.update.assert_not_called()

    # Should upsert team membership with the new budget ID
    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-rpm-test", "team_id": "team-rpm-test"}},
        data={
            "create": {
                "user_id": "user-rpm-test",
                "team_id": "team-rpm-test",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )


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
        where={"user_id_team_id": {"user_id": "user-rpm-only", "team_id": "team-rpm-only"}},
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

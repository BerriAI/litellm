# tests/litellm/proxy/common_utils/test_upsert_budget_membership.py
import types
import pytest
from unittest.mock import AsyncMock, MagicMock

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


# TEST: existing budget id, update only
@pytest.mark.asyncio
async def test_upsert_update_existing(mock_tx, fake_user):
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-2",
        user_id="user-2",
        max_budget=42.0,
        existing_budget_id="bud-999",
        user_api_key_dict=fake_user,
    )

    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "bud-999"},
        data={"max_budget": 42.0},
    )
    mock_tx.litellm_teammembership.update.assert_not_called()
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()


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


# TEST: create new budget and link membership, then update
@pytest.mark.asyncio
async def test_upsert_create_then_update(mock_tx, fake_user):
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

    # SECOND CALL – pretend the same membership already exists, and
    # reset call history so the next assertions are clear
    mock_tx.litellm_budgettable.create.reset_mock()
    mock_tx.litellm_teammembership.upsert.reset_mock()
    mock_tx.litellm_budgettable.update.reset_mock()

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-42",
        user_id="user-42",
        max_budget=25.0,                # new limit
        existing_budget_id=created_bid, # now we say it exists
        user_api_key_dict=fake_user,
    )

    # Now we expect ONLY an update to fire
    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": created_bid},
        data={"max_budget": 25.0},
    )
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()

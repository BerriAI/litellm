# tests/litellm/proxy/common_utils/test_upsert_budget_membership.py
import types
from datetime import datetime, timezone
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
    membership = MagicMock()
    membership.update = AsyncMock()
    membership.upsert = AsyncMock()

    budget = MagicMock()
    budget.update = AsyncMock()
    budget.find_unique = AsyncMock(return_value=None)
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


def budget_row(**fields):
    """A fake litellm_budgettable row whose model_dump returns the given fields."""
    row = MagicMock()
    row.model_dump.return_value = fields
    return row


def assert_future_reset_time(value):
    """A budget_reset_at must be a timezone-aware datetime in the future, so the
    member's budget actually rolls over and the UI shows a reset date instead of
    waiting for the reset cron to backfill it."""
    assert isinstance(value, datetime)
    assert value.tzinfo is not None
    assert value > datetime.now(timezone.utc)


# TEST: an empty patch (caller sent no budget fields) leaves everything alone.
# This is the merge-patch contract: absent != clear. Updating only a member's
# role must not silently wipe their budget.
@pytest.mark.asyncio
async def test_empty_patch_is_noop(mock_tx, fake_user):
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-1",
        user_id="user-1",
        existing_budget_id="bud-1",
        user_api_key_dict=fake_user,
        budget_patch={},
    )

    mock_tx.litellm_teammembership.update.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()
    mock_tx.litellm_budgettable.update.assert_not_called()
    mock_tx.litellm_budgettable.create.assert_not_called()


# TEST: clearing every limit on a member's private budget disconnects it, so the
# member falls back to the team default instead of keeping an empty private row.
@pytest.mark.asyncio
async def test_clearing_all_limits_disconnects(mock_tx, fake_user):
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(max_budget=100.0)
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-1",
        user_id="user-1",
        existing_budget_id="bud-1",
        user_api_key_dict=fake_user,
        budget_patch={"max_budget": None},
    )

    mock_tx.litellm_teammembership.update.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-1", "team_id": "team-1"}},
        data={"litellm_budget_table": {"disconnect": True}},
    )
    mock_tx.litellm_budgettable.update.assert_not_called()
    mock_tx.litellm_budgettable.create.assert_not_called()


# TEST: clearing one field on a budget that still has another limit updates in
# place (clears just that column + its reset time) and does NOT disconnect.
@pytest.mark.asyncio
async def test_clear_one_field_keeps_others(mock_tx, fake_user):
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(max_budget=100.0, budget_duration="24h")
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-1",
        user_id="user-1",
        existing_budget_id="bud-1",
        user_api_key_dict=fake_user,
        budget_patch={"budget_duration": None},
    )

    mock_tx.litellm_teammembership.update.assert_not_called()
    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "bud-1"},
        data={
            "updated_by": fake_user.user_id,
            "budget_duration": None,
            "budget_reset_at": None,
        },
    )


# TEST: setting budget_duration in place writes the duration AND a future
# budget_reset_at, so the budget rolls over without waiting for the reset cron.
@pytest.mark.asyncio
async def test_update_in_place_seeds_reset_at(mock_tx, fake_user):
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(max_budget=20.0)
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-dur",
        user_id="user-dur",
        existing_budget_id="bud-dur",
        user_api_key_dict=fake_user,
        budget_patch={"budget_duration": "30d"},
    )

    mock_tx.litellm_budgettable.update.assert_awaited_once()
    call = mock_tx.litellm_budgettable.update.await_args
    assert call.kwargs["where"] == {"budget_id": "bud-dur"}
    data = call.kwargs["data"]
    assert data["budget_duration"] == "30d"
    assert data["updated_by"] == fake_user.user_id
    assert_future_reset_time(data["budget_reset_at"])
    mock_tx.litellm_budgettable.create.assert_not_called()


# TEST: updating a single limit in place only writes that field; an untouched
# budget_duration must not get a (re)computed reset time.
@pytest.mark.asyncio
async def test_update_in_place_single_field_leaves_reset_at_alone(mock_tx, fake_user):
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(max_budget=50.0)
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-rpm",
        user_id="user-rpm",
        existing_budget_id="bud-rpm",
        user_api_key_dict=fake_user,
        budget_patch={"rpm_limit": 100},
    )

    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "bud-rpm"},
        data={"updated_by": fake_user.user_id, "rpm_limit": 100},
    )
    mock_tx.litellm_budgettable.create.assert_not_called()


# TEST: with no existing budget, a duration-only patch creates a budget carrying
# the duration and a future reset time, then links the membership.
@pytest.mark.asyncio
async def test_create_seeds_reset_at_and_links(mock_tx, fake_user):
    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-new",
        user_id="user-new",
        existing_budget_id=None,
        user_api_key_dict=fake_user,
        budget_patch={"budget_duration": "7d"},
    )

    mock_tx.litellm_budgettable.create.assert_awaited_once()
    data = mock_tx.litellm_budgettable.create.await_args.kwargs["data"]
    assert data["budget_duration"] == "7d"
    assert data["created_by"] == fake_user.user_id
    assert data["updated_by"] == fake_user.user_id
    assert_future_reset_time(data["budget_reset_at"])

    new_budget_id = mock_tx.litellm_budgettable.create.return_value.budget_id
    mock_tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-new", "team_id": "team-new"}},
        data={
            "create": {
                "user_id": "user-new",
                "team_id": "team-new",
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_budget_id}},
            },
        },
    )


# TEST: clone-on-write when the membership still points at the team's shared
# default budget. Editing this member must fork a private budget instead of
# mutating the shared row, and cloning a duration must seed a fresh reset time.
@pytest.mark.asyncio
async def test_clone_on_write_from_shared_default(mock_tx, fake_user):
    shared_default_id = "team-default-budget-1"
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(
            budget_id=shared_default_id,
            max_budget=200.0,
            soft_budget=None,
            max_parallel_requests=None,
            tpm_limit=500,
            rpm_limit=None,
            model_max_budget=None,
            budget_duration="1d",
            allowed_models=[],
        )
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-shared",
        user_id="user-shared",
        existing_budget_id=shared_default_id,
        user_api_key_dict=fake_user,
        budget_patch={"max_budget": 50.0},
        team_default_budget_id=shared_default_id,
    )

    mock_tx.litellm_budgettable.update.assert_not_called()
    mock_tx.litellm_budgettable.create.assert_awaited_once()
    create_data = mock_tx.litellm_budgettable.create.await_args.kwargs["data"]
    assert_future_reset_time(create_data.pop("budget_reset_at"))
    assert create_data == {
        "created_by": fake_user.user_id,
        "updated_by": fake_user.user_id,
        "max_budget": 50.0,  # caller wins
        "tpm_limit": 500,  # cloned from default
        "budget_duration": "1d",  # cloned from default
    }

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


# TEST: forking the shared default while clearing its duration must drop the
# duration (and not carry a reset time) on the new private budget.
@pytest.mark.asyncio
async def test_clone_on_write_clears_duration(mock_tx, fake_user):
    shared_default_id = "team-default-budget-1"
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(
            budget_id=shared_default_id,
            max_budget=200.0,
            tpm_limit=500,
            budget_duration="1d",
            allowed_models=[],
        )
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-shared",
        user_id="user-shared",
        existing_budget_id=shared_default_id,
        user_api_key_dict=fake_user,
        budget_patch={"budget_duration": None},
        team_default_budget_id=shared_default_id,
    )

    mock_tx.litellm_budgettable.update.assert_not_called()
    create_data = mock_tx.litellm_budgettable.create.await_args.kwargs["data"]
    assert create_data == {
        "created_by": fake_user.user_id,
        "updated_by": fake_user.user_id,
        "max_budget": 200.0,
        "tpm_limit": 500,
        "budget_duration": None,
    }
    assert "budget_reset_at" not in create_data


# TEST: when the member already has their own private budget (different from the
# team default), we update it in place rather than forking another row.
@pytest.mark.asyncio
async def test_private_budget_updates_in_place(mock_tx, fake_user):
    mock_tx.litellm_budgettable.find_unique = AsyncMock(
        return_value=budget_row(max_budget=10.0)
    )

    await _upsert_budget_and_membership(
        mock_tx,
        team_id="team-mixed",
        user_id="user-private",
        existing_budget_id="private-budget-xyz",
        user_api_key_dict=fake_user,
        budget_patch={"max_budget": 75.0},
        team_default_budget_id="team-default-budget-1",
    )

    mock_tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "private-budget-xyz"},
        data={"max_budget": 75.0, "updated_by": fake_user.user_id},
    )
    mock_tx.litellm_budgettable.create.assert_not_called()
    mock_tx.litellm_teammembership.upsert.assert_not_called()

"""Regression tests for LIT-3359: shared budget_id mutation when multiple
LiteLLM_TeamMembership rows reference the same budget_id.

Before the ref-count guard in _upsert_budget_and_membership, the buggy
in-place update path mutated a budget row that was shared by other team
memberships, so changing one members max_budget changed every member who
happened to share that budget_id.
"""
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.management_endpoints.common_utils import (
    _upsert_budget_and_membership,
)


@pytest.fixture
def fake_user():
    return types.SimpleNamespace(user_id="tester@example.com")


def _make_tx(*, sharing_count: int, existing_row_max_budget: float = 200.0):
    membership = MagicMock()
    membership.update = AsyncMock()
    membership.upsert = AsyncMock()
    membership.count = AsyncMock(return_value=sharing_count)

    budget = MagicMock()
    budget.update = AsyncMock()
    budget.create = AsyncMock(
        return_value=types.SimpleNamespace(budget_id="new-priv-1")
    )

    shared_row = MagicMock()
    shared_row.model_dump.return_value = {
        "budget_id": "shared-bid",
        "max_budget": existing_row_max_budget,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": 500,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "1d",
        "allowed_models": [],
    }
    budget.find_unique = AsyncMock(return_value=shared_row)

    tx = MagicMock()
    tx.litellm_teammembership = membership
    tx.litellm_budgettable = budget
    return tx


@pytest.mark.asyncio
async def test_clone_on_write_when_budget_shared_no_team_default(fake_user):
    """LIT-3359 repro: 3 memberships share a budget_id and team_default_budget_id
    is None (customer never used the team_member_budget setter, or the metadata
    has been cleared). Updating one member must clone-on-write, not mutate the
    shared row.
    """
    tx = _make_tx(sharing_count=3)

    await _upsert_budget_and_membership(
        tx,
        team_id="team-X",
        user_id="user-1",
        max_budget=50.0,
        existing_budget_id="shared-bid",
        user_api_key_dict=fake_user,
        team_default_budget_id=None,
    )

    tx.litellm_budgettable.update.assert_not_called()

    tx.litellm_budgettable.create.assert_awaited_once_with(
        data={
            "created_by": fake_user.user_id,
            "updated_by": fake_user.user_id,
            "max_budget": 50.0,
            "tpm_limit": 500,
            "budget_duration": "1d",
        },
        include={"team_membership": True},
    )

    new_bid = tx.litellm_budgettable.create.return_value.budget_id
    tx.litellm_teammembership.upsert.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "user-1", "team_id": "team-X"}},
        data={
            "create": {
                "user_id": "user-1",
                "team_id": "team-X",
                "litellm_budget_table": {"connect": {"budget_id": new_bid}},
            },
            "update": {
                "litellm_budget_table": {"connect": {"budget_id": new_bid}},
            },
        },
    )


@pytest.mark.asyncio
async def test_in_place_update_when_budget_is_private(fake_user):
    """Counterpart: when sharing_count is 1 (private budget) AND
    team_default_budget_id is None, in-place behavior is preserved.
    """
    tx = _make_tx(sharing_count=1)

    await _upsert_budget_and_membership(
        tx,
        team_id="team-Y",
        user_id="user-2",
        max_budget=75.0,
        existing_budget_id="private-bid",
        user_api_key_dict=fake_user,
        team_default_budget_id=None,
    )

    tx.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "private-bid"},
        data={
            "max_budget": 75.0,
            "updated_by": fake_user.user_id,
        },
    )
    tx.litellm_budgettable.create.assert_not_called()
    tx.litellm_teammembership.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_count_probe_failure_falls_back_to_in_place(fake_user):
    """If the count probe raises, fall back to the previous behavior."""
    tx = _make_tx(sharing_count=1)
    tx.litellm_teammembership.count = AsyncMock(side_effect=RuntimeError("no count"))

    await _upsert_budget_and_membership(
        tx,
        team_id="team-Z",
        user_id="user-3",
        max_budget=10.0,
        existing_budget_id="some-bid",
        user_api_key_dict=fake_user,
        team_default_budget_id=None,
    )

    tx.litellm_budgettable.update.assert_awaited_once()
    tx.litellm_budgettable.create.assert_not_called()

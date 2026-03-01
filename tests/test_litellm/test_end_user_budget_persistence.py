"""
Tests for persisting default end-user budget_id to database.

Regression tests for https://github.com/BerriAI/litellm/issues/22019

Problem: When max_end_user_budget_id is configured, implicitly created
end users get the budget applied in-memory only.  The DB row keeps
budget_id=NULL, so the budget-reset job never resets their spend.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_BudgetTable
from litellm.proxy.auth.auth_checks import (
    _apply_default_budget_to_end_user,
    _persist_end_user_budget_id,
)


# ────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────


def _make_end_user(user_id: str = "user-1", budget_table=None) -> LiteLLM_EndUserTable:
    return LiteLLM_EndUserTable(
        user_id=user_id,
        blocked=False,
        litellm_budget_table=budget_table,
        spend=0.0,
    )


def _make_budget(budget_id: str = "default-budget") -> LiteLLM_BudgetTable:
    return LiteLLM_BudgetTable(
        budget_id=budget_id,
        max_budget=100.0,
    )


def _mock_prisma_client():
    pc = MagicMock()
    pc.db = MagicMock()
    pc.db.litellm_endusertable = MagicMock()
    pc.db.litellm_endusertable.update = AsyncMock(return_value=None)
    pc.db.litellm_budgettable = MagicMock()
    pc.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=MagicMock(
            dict=lambda: _make_budget("default-budget").__dict__,
        )
    )
    return pc


def _mock_cache():
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_set_cache = AsyncMock(return_value=None)
    return cache


# ────────────────────────────────────────────────────
#  Tests: _persist_end_user_budget_id
# ────────────────────────────────────────────────────


class TestPersistEndUserBudgetId:
    @pytest.mark.asyncio
    async def test_updates_database_row(self):
        pc = _mock_prisma_client()
        await _persist_end_user_budget_id(
            prisma_client=pc,
            user_id="user-1",
            budget_id="default-budget",
        )
        pc.db.litellm_endusertable.update.assert_awaited_once_with(
            where={"user_id": "user-1"},
            data={"budget_id": "default-budget"},
        )

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        pc = _mock_prisma_client()
        pc.db.litellm_endusertable.update = AsyncMock(
            side_effect=Exception("DB down")
        )
        # Must not raise
        await _persist_end_user_budget_id(
            prisma_client=pc,
            user_id="user-1",
            budget_id="default-budget",
        )


# ────────────────────────────────────────────────────
#  Tests: _apply_default_budget_to_end_user
# ────────────────────────────────────────────────────


class TestApplyDefaultBudgetToEndUser:
    @pytest.mark.asyncio
    async def test_skips_if_budget_already_assigned(self):
        end_user = _make_end_user(budget_table=_make_budget())
        pc = _mock_prisma_client()
        cache = _mock_cache()

        result = await _apply_default_budget_to_end_user(
            end_user_obj=end_user,
            prisma_client=pc,
            user_api_key_cache=cache,
        )
        assert result.litellm_budget_table is not None
        # No DB update should be triggered
        pc.db.litellm_endusertable.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_if_no_default_configured(self):
        original = litellm.max_end_user_budget_id
        try:
            litellm.max_end_user_budget_id = None
            end_user = _make_end_user()
            pc = _mock_prisma_client()
            cache = _mock_cache()

            result = await _apply_default_budget_to_end_user(
                end_user_obj=end_user,
                prisma_client=pc,
                user_api_key_cache=cache,
            )
            assert result.litellm_budget_table is None
        finally:
            litellm.max_end_user_budget_id = original

    @pytest.mark.asyncio
    async def test_applies_budget_and_persists_to_db(self):
        original = litellm.max_end_user_budget_id
        try:
            litellm.max_end_user_budget_id = "default-budget"
            end_user = _make_end_user()
            pc = _mock_prisma_client()
            cache = _mock_cache()

            result = await _apply_default_budget_to_end_user(
                end_user_obj=end_user,
                prisma_client=pc,
                user_api_key_cache=cache,
            )
            # Budget applied in-memory
            assert result.litellm_budget_table is not None

            # Let the event loop run the fire-and-forget task
            await asyncio.sleep(0)

            # Budget persisted to DB via background task
            pc.db.litellm_endusertable.update.assert_awaited_once_with(
                where={"user_id": "user-1"},
                data={"budget_id": "default-budget"},
            )
        finally:
            litellm.max_end_user_budget_id = original

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_break_in_memory_budget(self):
        original = litellm.max_end_user_budget_id
        try:
            litellm.max_end_user_budget_id = "default-budget"
            end_user = _make_end_user()
            pc = _mock_prisma_client()
            pc.db.litellm_endusertable.update = AsyncMock(
                side_effect=Exception("DB down")
            )
            cache = _mock_cache()

            result = await _apply_default_budget_to_end_user(
                end_user_obj=end_user,
                prisma_client=pc,
                user_api_key_cache=cache,
            )
            # In-memory budget still applied
            assert result.litellm_budget_table is not None

            # Let the event loop run the fire-and-forget task
            await asyncio.sleep(0)
        finally:
            litellm.max_end_user_budget_id = original

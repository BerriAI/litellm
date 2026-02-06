"""
Simplified tests for default end user budget feature.

Tests the core scenarios where litellm.max_end_user_budget_id applies
a default budget to end users without explicit budgets.
"""

import sys
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_EndUserTable
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.caching import DualCache


@pytest.mark.asyncio
async def test_default_budget_applied_to_end_user_without_budget():
    """
    Core scenario: End user without explicit budget gets default budget applied.
    This is the main use case - applying limits to all unbudgeted end users.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    default_budget_id = str(uuid.uuid4())
    litellm.max_end_user_budget_id = default_budget_id
    
    default_budget = LiteLLM_BudgetTable(
        budget_id=default_budget_id,
        max_budget=10.0,
        rpm_limit=2,
        tpm_limit=10,
    )
    
    # Mock end user in DB without budget
    mock_end_user_data = {
        "user_id": end_user_id,
        "spend": 1.0,
        "litellm_budget_table": None,
        "alias": None,
        "allowed_model_region": None,
        "default_model": None,
        "blocked": False,
    }
    
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: mock_end_user_data)
    )
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: default_budget.dict())
    )
    
    mock_cache = AsyncMock(spec=DualCache)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()
    
    result = await get_end_user_object(
        end_user_id=end_user_id,
        prisma_client=mock_prisma_client,
        user_api_key_cache=mock_cache,
        route="/chat/completions",
    )
    
    # Verify default budget was applied
    assert result is not None
    assert result.litellm_budget_table is not None
    assert result.litellm_budget_table.budget_id == default_budget_id
    assert result.litellm_budget_table.max_budget == 10.0
    assert result.litellm_budget_table.rpm_limit == 2
    assert result.litellm_budget_table.tpm_limit == 10
    
    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_explicit_budget_not_overridden_by_default():
    """
    Core scenario: End users with explicit budgets keep their budgets.
    The default should not override user-specific configurations.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    explicit_budget_id = str(uuid.uuid4())
    default_budget_id = str(uuid.uuid4())
    litellm.max_end_user_budget_id = default_budget_id
    
    explicit_budget = LiteLLM_BudgetTable(
        budget_id=explicit_budget_id,
        max_budget=100.0,
        rpm_limit=50,
    )
    
    # Mock end user with explicit budget
    mock_end_user_data = {
        "user_id": end_user_id,
        "spend": 10.0,
        "litellm_budget_table": explicit_budget.dict(),
        "alias": None,
        "allowed_model_region": None,
        "default_model": None,
        "blocked": False,
    }
    
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: mock_end_user_data)
    )
    
    mock_cache = AsyncMock(spec=DualCache)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()
    
    result = await get_end_user_object(
        end_user_id=end_user_id,
        prisma_client=mock_prisma_client,
        user_api_key_cache=mock_cache,
        route="/chat/completions",
    )
    
    # Verify explicit budget is kept (not replaced with default)
    assert result is not None
    assert result.litellm_budget_table.budget_id == explicit_budget_id
    assert result.litellm_budget_table.max_budget == 100.0
    assert result.litellm_budget_table.rpm_limit == 50
    
    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_budget_enforcement_blocks_over_budget_users():
    """
    Core scenario: Budget limits are actually enforced.
    Users who exceed their budget should be blocked.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    default_budget_id = str(uuid.uuid4())
    litellm.max_end_user_budget_id = default_budget_id
    
    default_budget = LiteLLM_BudgetTable(
        budget_id=default_budget_id,
        max_budget=10.0,
        rpm_limit=2,
    )
    
    # Mock end user who has already spent more than budget
    mock_end_user_data = {
        "user_id": end_user_id,
        "spend": 15.0,  # Exceeds budget of 10.0
        "litellm_budget_table": None,
        "alias": None,
        "allowed_model_region": None,
        "default_model": None,
        "blocked": False,
    }
    
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: mock_end_user_data)
    )
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: default_budget.dict())
    )
    
    mock_cache = AsyncMock(spec=DualCache)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()
    
    # Should raise BudgetExceededError
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            route="/chat/completions",
        )
    
    assert "ExceededBudget" in str(exc_info.value)
    assert end_user_id in str(exc_info.value)
    
    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_system_works_without_default_budget_configured():
    """
    Core scenario: System continues to work when no default budget is configured.
    This ensures backward compatibility.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    litellm.max_end_user_budget_id = None  # Not configured
    
    # Mock end user without budget
    mock_end_user_data = {
        "user_id": end_user_id,
        "spend": 5.0,
        "litellm_budget_table": None,
        "alias": None,
        "allowed_model_region": None,
        "default_model": None,
        "blocked": False,
    }
    
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=MagicMock(dict=lambda: mock_end_user_data)
    )
    
    mock_cache = AsyncMock(spec=DualCache)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()
    
    result = await get_end_user_object(
        end_user_id=end_user_id,
        prisma_client=mock_prisma_client,
        user_api_key_cache=mock_cache,
        route="/chat/completions",
    )
    
    # Should work fine, just without budget limits
    assert result is not None
    assert result.user_id == end_user_id
    assert result.litellm_budget_table is None  # No budget applied


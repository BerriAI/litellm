"""
Test for issue #19105: Team member budget reset

Tests that team member budgets reset based on the team's budget_duration,
not just their individual budget table's duration.
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob


@pytest.mark.asyncio
async def test_team_member_budget_resets_with_team():
    """
    Test that team member budgets reset when the team's budget resets,
    even if the member's budget table doesn't have its own budget_duration.

    This tests the fix for issue #19105.
    """
    # Mock Prisma client
    prisma_client = MagicMock(spec=PrismaClient)
    prisma_client.db = MagicMock()

    # Mock ProxyLogging
    proxy_logging_obj = MagicMock(spec=ProxyLogging)
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    # Create test data
    now = datetime.utcnow()
    past_time = now - timedelta(hours=1)

    # Team with budget_duration that needs reset
    team = MagicMock()
    team.team_id = "team_123"
    team.max_budget = 10.0
    team.budget_duration = "1d"
    team.budget_reset_at = past_time  # In the past, needs reset
    team.spend = 5.0

    # Mock get_data to return the team
    async def mock_get_data(table_name, query_type, **kwargs):
        if table_name == "team" and "reset_at" in kwargs:
            return [team]
        return None

    prisma_client.get_data = AsyncMock(side_effect=mock_get_data)

    # Mock update_many for team membership
    update_result = MagicMock()
    update_result.count = 2  # 2 team members updated
    prisma_client.db.litellm_teammembership.update_many = AsyncMock(
        return_value=update_result
    )

    # Create reset job
    reset_job = ResetBudgetJob(
        proxy_logging_obj=proxy_logging_obj, prisma_client=prisma_client
    )

    # Run the team member reset
    await reset_job.reset_budget_for_team_members_by_team()

    # Verify that update_many was called with correct parameters
    prisma_client.db.litellm_teammembership.update_many.assert_called_once()
    call_args = prisma_client.db.litellm_teammembership.update_many.call_args

    # Check the where clause includes the team_id
    assert call_args.kwargs["where"]["team_id"] == {"in": ["team_123"]}

    # Check that spend is reset to 0
    assert call_args.kwargs["data"]["spend"] == 0

    # Verify success hook was called
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_called_once()


@pytest.mark.asyncio
async def test_team_member_budget_reset_no_teams():
    """
    Test that the function handles the case where no teams need resetting.
    """
    # Mock Prisma client
    prisma_client = MagicMock(spec=PrismaClient)
    prisma_client.db = MagicMock()

    # Mock ProxyLogging
    proxy_logging_obj = MagicMock(spec=ProxyLogging)
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()

    # Mock get_data to return empty list (no teams need reset)
    prisma_client.get_data = AsyncMock(return_value=[])

    # Mock update_many
    prisma_client.db.litellm_teammembership.update_many = AsyncMock()

    # Create reset job
    reset_job = ResetBudgetJob(
        proxy_logging_obj=proxy_logging_obj, prisma_client=prisma_client
    )

    # Run the team member reset
    await reset_job.reset_budget_for_team_members_by_team()

    # Verify that update_many was NOT called (no teams to process)
    prisma_client.db.litellm_teammembership.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_team_member_budget_reset_multiple_teams():
    """
    Test that team member budgets are reset for multiple teams.
    """
    # Mock Prisma client
    prisma_client = MagicMock(spec=PrismaClient)
    prisma_client.db = MagicMock()

    # Mock ProxyLogging
    proxy_logging_obj = MagicMock(spec=ProxyLogging)
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()

    # Create test data - multiple teams
    now = datetime.utcnow()
    past_time = now - timedelta(hours=1)

    teams = []
    for i in range(3):
        team = MagicMock()
        team.team_id = f"team_{i}"
        team.max_budget = 10.0
        team.budget_duration = "1d"
        team.budget_reset_at = past_time
        teams.append(team)

    # Mock get_data to return multiple teams
    prisma_client.get_data = AsyncMock(return_value=teams)

    # Mock update_many
    update_result = MagicMock()
    update_result.count = 5  # 5 team members updated across all teams
    prisma_client.db.litellm_teammembership.update_many = AsyncMock(
        return_value=update_result
    )

    # Create reset job
    reset_job = ResetBudgetJob(
        proxy_logging_obj=proxy_logging_obj, prisma_client=prisma_client
    )

    # Run the team member reset
    await reset_job.reset_budget_for_team_members_by_team()

    # Verify that update_many was called with all team IDs
    call_args = prisma_client.db.litellm_teammembership.update_many.call_args
    assert set(call_args.kwargs["where"]["team_id"]["in"]) == {
        "team_0",
        "team_1",
        "team_2",
    }

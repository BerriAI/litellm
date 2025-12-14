"""
Tests for organization budget enforcement.

These tests verify that organization-level budgets are properly enforced during
request authentication. When an organization's spend exceeds its max_budget,
requests should fail with BudgetExceededError.

This prevents teams within an organization from collectively exceeding the
organization's budget limit.
"""

import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../"))

import litellm
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy.utils import ProxyLogging


@pytest.mark.asyncio
async def test_organization_budget_exceeded_blocks_request():
    """
    Bug: Organization budget is retrieved but NEVER enforced.

    When organization spend >= organization_max_budget, requests should fail
    with BudgetExceededError. Currently this passes because no check exists.
    """
    org_id = "test-org-budget-exceeded"

    # Organization with max_budget of 100, but spend is 150
    org_object = LiteLLM_OrganizationTable(
        organization_id=org_id,
        budget_id="org-budget-1",
        spend=150.0,  # Over budget!
        models=["gpt-4"],
        created_by="test",
        updated_by="test",
        litellm_budget_table=LiteLLM_BudgetTable(
            max_budget=100.0,  # Budget is 100
        ),
    )

    # Team within the organization (team itself is under budget)
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        organization_id=org_id,
        max_budget=50.0,  # Team budget is 50
        spend=10.0,       # Team spend is only 10 - under budget
        models=["gpt-4"],
    )

    # Valid token with organization info
    valid_token = UserAPIKeyAuth(
        token="sk-test-123",
        team_id="test-team-1",
        org_id=org_id,
        organization_max_budget=100.0,  # This is set but never checked!
    )

    mock_request = MagicMock()
    mock_request.url.path = "/v1/chat/completions"

    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = AsyncMock()

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        with patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache:
            with patch("litellm.proxy.auth.auth_checks.get_org_object", new_callable=AsyncMock) as mock_get_org:
                mock_get_org.return_value = org_object

                # BUG: This should raise BudgetExceededError but currently passes
                with pytest.raises(litellm.BudgetExceededError) as exc_info:
                    await common_checks(
                        request_body={"model": "gpt-4"},
                        team_object=team_object,
                        user_object=None,
                        end_user_object=None,
                        global_proxy_spend=None,
                        general_settings={},
                        route="/v1/chat/completions",
                        llm_router=None,
                        proxy_logging_obj=mock_proxy_logging,
                        valid_token=valid_token,
                        request=mock_request,
                    )

                assert "Organization" in str(exc_info.value.message)
                assert exc_info.value.current_cost == 150.0
                assert exc_info.value.max_budget == 100.0


@pytest.mark.asyncio
async def test_multiple_teams_exceed_organization_budget():
    """
    Test that organization budget is enforced even when individual teams are under budget.

    Scenario:
    - Organization max_budget = $5000, spend = $5000 (at limit)
    - Team A spend = $1500 (under team budget of $2000)
    - Request via Team A should FAIL because org is at budget limit

    Expected: Request fails with BudgetExceededError
    """
    org_id = "multi-team-org"

    # Organization at budget limit
    org_object = LiteLLM_OrganizationTable(
        organization_id=org_id,
        budget_id="org-budget-2",
        spend=5000.0,  # At $5000 limit
        models=["gpt-4"],
        created_by="test",
        updated_by="test",
        litellm_budget_table=LiteLLM_BudgetTable(
            max_budget=5000.0,  # Org budget is $5000
        ),
    )

    # Team A - under its own budget, but org is almost at limit
    team_a = LiteLLM_TeamTable(
        team_id="team-a",
        organization_id=org_id,
        max_budget=2000.0,
        spend=1500.0,  # Team A has spent $1500 of its $2000 budget
        models=["gpt-4"],
    )

    valid_token = UserAPIKeyAuth(
        token="sk-team-a-key",
        team_id="team-a",
        org_id=org_id,
        organization_max_budget=5000.0,  # Set but never enforced
    )

    mock_request = MagicMock()
    mock_request.url.path = "/v1/chat/completions"

    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = AsyncMock()

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        with patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache:
            with patch("litellm.proxy.auth.auth_checks.get_org_object", new_callable=AsyncMock) as mock_get_org:
                mock_get_org.return_value = org_object

                # Org is at budget limit, should raise BudgetExceededError
                with pytest.raises(litellm.BudgetExceededError) as exc_info:
                    await common_checks(
                        request_body={"model": "gpt-4"},
                        team_object=team_a,
                        user_object=None,
                        end_user_object=None,
                        global_proxy_spend=None,
                        general_settings={},
                        route="/v1/chat/completions",
                        llm_router=None,
                        proxy_logging_obj=mock_proxy_logging,
                        valid_token=valid_token,
                        request=mock_request,
                    )

                # Verify the error message mentions organization
                assert "Organization" in str(exc_info.value.message)
                assert exc_info.value.current_cost == 5000.0
                assert exc_info.value.max_budget == 5000.0


@pytest.mark.asyncio
async def test_organization_budget_fields_are_checked():
    """
    Verify that organization_max_budget is populated in UserAPIKeyAuth
    and BudgetExceededError is raised when organization is over budget.
    """
    # Token has org budget info
    valid_token = UserAPIKeyAuth(
        token="sk-test",
        team_id="test-team",
        org_id="test-org",
        organization_max_budget=100.0,  # Budget is $100
    )

    # Verify the field exists and is set
    assert valid_token.organization_max_budget == 100.0
    assert valid_token.org_id == "test-org"

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        organization_id="test-org",
        max_budget=None,
        spend=0.0,
        models=["gpt-4"],
    )

    mock_request = MagicMock()
    mock_request.url.path = "/v1/chat/completions"

    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = AsyncMock()

    # Organization is over budget
    org_over_budget = LiteLLM_OrganizationTable(
        organization_id="test-org",
        budget_id="budget-1",
        spend=150.0,  # Over $100 budget
        models=["gpt-4"],
        created_by="test",
        updated_by="test",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        with patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache:
            with patch("litellm.proxy.auth.auth_checks.get_org_object", new_callable=AsyncMock) as mock_get_org:
                mock_get_org.return_value = org_over_budget

                # Organization is over budget, should raise BudgetExceededError
                with pytest.raises(litellm.BudgetExceededError) as exc_info:
                    await common_checks(
                        request_body={"model": "gpt-4"},
                        team_object=team_object,
                        user_object=None,
                        end_user_object=None,
                        global_proxy_spend=None,
                        general_settings={},
                        route="/v1/chat/completions",
                        llm_router=None,
                        proxy_logging_obj=mock_proxy_logging,
                        valid_token=valid_token,
                        request=mock_request,
                    )

                assert exc_info.value.current_cost == 150.0
                assert exc_info.value.max_budget == 100.0


@pytest.mark.asyncio
async def test_both_team_and_org_budget_enforced():
    """
    Verify that both team budget and organization budget are enforced consistently.

    This test verifies:
    1. Team over budget raises BudgetExceededError
    2. Organization over budget also raises BudgetExceededError
    """
    mock_request = MagicMock()
    mock_request.url.path = "/v1/chat/completions"

    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = AsyncMock()

    # Scenario A: Team over budget - should raise BudgetExceededError
    team_over_budget = LiteLLM_TeamTable(
        team_id="team-over",
        max_budget=100.0,
        spend=150.0,  # Over budget
        models=["gpt-4"],
    )

    valid_token_team = UserAPIKeyAuth(
        token="sk-team-test",
        team_id="team-over",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        with patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache:
            with pytest.raises(litellm.BudgetExceededError) as exc_info:
                await common_checks(
                    request_body={"model": "gpt-4"},
                    team_object=team_over_budget,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route="/v1/chat/completions",
                    llm_router=None,
                    proxy_logging_obj=mock_proxy_logging,
                    valid_token=valid_token_team,
                    request=mock_request,
                )
            assert "Team" in str(exc_info.value.message)

    # Scenario B: Org over budget - should also raise BudgetExceededError
    org_over_budget = LiteLLM_OrganizationTable(
        organization_id="org-over",
        budget_id="budget-1",
        spend=150.0,  # Over $100 budget
        models=["gpt-4"],
        created_by="test",
        updated_by="test",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    team_under_budget = LiteLLM_TeamTable(
        team_id="team-under",
        organization_id="org-over",
        max_budget=50.0,
        spend=10.0,  # Team is fine
        models=["gpt-4"],
    )

    valid_token_org = UserAPIKeyAuth(
        token="sk-org-test",
        team_id="team-under",
        org_id="org-over",
        organization_max_budget=100.0,
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        with patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache:
            with patch("litellm.proxy.auth.auth_checks.get_org_object", new_callable=AsyncMock) as mock_get_org:
                mock_get_org.return_value = org_over_budget

                # Organization is over budget, should raise BudgetExceededError
                with pytest.raises(litellm.BudgetExceededError) as exc_info:
                    await common_checks(
                        request_body={"model": "gpt-4"},
                        team_object=team_under_budget,
                        user_object=None,
                        end_user_object=None,
                        global_proxy_spend=None,
                        general_settings={},
                        route="/v1/chat/completions",
                        llm_router=None,
                        proxy_logging_obj=mock_proxy_logging,
                        valid_token=valid_token_org,
                        request=mock_request,
                    )

                assert "Organization" in str(exc_info.value.message)
                assert exc_info.value.current_cost == 150.0
                assert exc_info.value.max_budget == 100.0

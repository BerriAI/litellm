"""
Unit tests for team member budget checks in common_checks.
These tests verify the team member budget enforcement without requiring a proxy server.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request

import litellm
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks, get_team_membership


@pytest.mark.asyncio
async def test_team_member_budget_check_exceeds_budget():
    """Test that common_checks raises BudgetExceededError when team member spend exceeds budget."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # Create team object
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        team_alias="Test Team",
        spend=0.0,
        max_budget=None,
    )

    # Create user object
    user_object = LiteLLM_UserTable(
        user_id="test-user-1",
        spend=0.0,
        max_budget=None,
    )

    # Create valid token
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    # Create team membership with budget exceeded
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user-1",
        team_id="test-team-1",
        spend=0.0000002,  # Exceeds budget
        litellm_budget_table=LiteLLM_BudgetTable(
            max_budget=0.0000001,  # Very small budget
        ),
    )

    mock_request = MagicMock(spec=Request)
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # Mock get_team_membership to return our team membership
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        new_callable=AsyncMock,
        return_value=team_membership,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    ):
        # Should raise BudgetExceededError
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await common_checks(
                request_body=request_body,
                team_object=team_object,
                user_object=user_object,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=mock_proxy_logging_obj,
                valid_token=valid_token,
                request=mock_request,
            )

        # Verify error message contains expected text
        assert "Budget has been exceeded" in str(exc_info.value)
        assert "test-user-1" in str(exc_info.value)
        assert "test-team-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_team_member_budget_check_within_budget():
    """Test that common_checks passes when team member spend is within budget."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # Create team object
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        team_alias="Test Team",
        spend=0.0,
        max_budget=None,
    )

    # Create user object
    user_object = LiteLLM_UserTable(
        user_id="test-user-1",
        spend=0.0,
        max_budget=None,
    )

    # Create valid token
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    # Create team membership within budget
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user-1",
        team_id="test-team-1",
        spend=0.00000005,  # Within budget
        litellm_budget_table=LiteLLM_BudgetTable(
            max_budget=0.0000001,
        ),
    )

    mock_request = MagicMock(spec=Request)
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # Mock get_team_membership to return our team membership
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        new_callable=AsyncMock,
        return_value=team_membership,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    ):
        # Should not raise an exception
        result = await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=mock_proxy_logging_obj,
            valid_token=valid_token,
            request=mock_request,
        )

        assert result is True


@pytest.mark.asyncio
async def test_team_member_budget_check_no_budget_set():
    """Test that common_checks passes when team member has no budget set."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # Create team object
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        team_alias="Test Team",
        spend=0.0,
        max_budget=None,
    )

    # Create user object
    user_object = LiteLLM_UserTable(
        user_id="test-user-1",
        spend=0.0,
        max_budget=None,
    )

    # Create valid token
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    # Create team membership without budget
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user-1",
        team_id="test-team-1",
        spend=0.0,
        litellm_budget_table=None,  # No budget set
    )

    mock_request = MagicMock(spec=Request)
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # Mock get_team_membership to return our team membership
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        new_callable=AsyncMock,
        return_value=team_membership,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    ):
        # Should not raise an exception (no budget means no limit)
        result = await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=mock_proxy_logging_obj,
            valid_token=valid_token,
            request=mock_request,
        )

        assert result is True


@pytest.mark.asyncio
async def test_team_member_budget_check_no_team_membership():
    """Test that common_checks passes when team membership doesn't exist."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # Create team object
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        team_alias="Test Team",
        spend=0.0,
        max_budget=None,
    )

    # Create user object
    user_object = LiteLLM_UserTable(
        user_id="test-user-1",
        spend=0.0,
        max_budget=None,
    )

    # Create valid token
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    mock_request = MagicMock(spec=Request)
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # Mock get_team_membership to return None (no membership)
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    ):
        # Should not raise an exception (no membership means no budget check)
        result = await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=mock_proxy_logging_obj,
            valid_token=valid_token,
            request=mock_request,
        )

        assert result is True


@pytest.mark.asyncio
async def test_team_member_budget_check_personal_key_not_team():
    """Test that team member budget check is skipped for personal keys (no team)."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # No team object (personal key)
    team_object = None

    # Create user object
    user_object = LiteLLM_UserTable(
        user_id="test-user-1",
        spend=0.0,
        max_budget=None,
    )

    # Create valid token without team
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        team_id=None,  # Personal key
        models=["gpt-3.5-turbo"],
    )

    mock_request = MagicMock(spec=Request)
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # get_team_membership should not be called for personal keys
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        new_callable=AsyncMock,
    ) as mock_get_team_membership, patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    ):
        result = await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=mock_proxy_logging_obj,
            valid_token=valid_token,
            request=mock_request,
        )

        # Should pass and get_team_membership should not be called
        assert result is True
        mock_get_team_membership.assert_not_called()

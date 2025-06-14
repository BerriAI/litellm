from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager


@pytest.mark.asyncio
async def test_map_user_to_teams_user_already_in_team():
    """Test that no action is taken when user is already in team"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(
        team_id="test_team_1",
        members_with_roles=[Member(user_id="test_user_1", role="user")],
    )

    # Mock team_member_add to ensure it's not called
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
    ) as mock_add:
        await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)
        mock_add.assert_not_called()


@pytest.mark.asyncio
async def test_map_user_to_teams_add_new_user():
    """Test that new user is added to team"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Mock team_member_add
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
    ) as mock_add:
        await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)
        mock_add.assert_called_once()
        # Verify the correct data was passed to team_member_add
        call_args = mock_add.call_args[1]["data"]
        assert call_args.member.user_id == "test_user_1"
        assert call_args.member.role == "user"
        assert call_args.team_id == "test_team_1"


@pytest.mark.asyncio
async def test_map_user_to_teams_handles_already_in_team_exception():
    """Test that team_member_already_in_team exception is handled gracefully"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Create a ProxyException with team_member_already_in_team error type
    already_in_team_exception = ProxyException(
        message="User test_user_1 already in team",
        type=ProxyErrorTypes.team_member_already_in_team,
        param="user_id",
        code="400",
    )

    # Mock team_member_add to raise the exception
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        side_effect=already_in_team_exception,
    ) as mock_add:
        with patch("litellm.proxy.auth.handle_jwt.verbose_proxy_logger") as mock_logger:
            # This should not raise an exception
            result = await JWTAuthManager.map_user_to_teams(
                user_object=user, team_object=team
            )

            # Verify the method completed successfully
            assert result is None
            mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_map_user_to_teams_reraises_other_proxy_exceptions():
    """Test that other ProxyException types are re-raised"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Create a ProxyException with a different error type
    other_exception = ProxyException(
        message="Some other error",
        type=ProxyErrorTypes.internal_server_error,
        param="some_param",
        code="500",
    )

    # Mock team_member_add to raise the exception
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        side_effect=other_exception,
    ) as mock_add:
        # This should re-raise the exception
        with pytest.raises(ProxyException) as exc_info:
            await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)


@pytest.mark.asyncio
async def test_map_user_to_teams_null_inputs():
    """Test that method handles null inputs gracefully"""
    # Test with null user
    await JWTAuthManager.map_user_to_teams(
        user_object=None, team_object=LiteLLM_TeamTable(team_id="test_team_1")
    )

    # Test with null team
    await JWTAuthManager.map_user_to_teams(
        user_object=LiteLLM_UserTable(user_id="test_user_1"), team_object=None
    )

    # Test with both null
    await JWTAuthManager.map_user_to_teams(user_object=None, team_object=None)

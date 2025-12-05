
import pytest
from fastapi import HTTPException
from litellm.proxy.management_endpoints.team_endpoints import team_member_add
from litellm.proxy._types import TeamMemberAddRequest, Member, UserAPIKeyAuth

@pytest.mark.asyncio
async def test_team_member_add_org_validation_failure(mocker):
    """
    Test that adding a user to an org-scoped team fails if the user is not in the org.
    """
    # Mock dependencies
    mock_prisma_client = mocker.MagicMock()
    
    # Mock team with organization_id
    mock_team_data = {
        "team_id": "org-team-id",
        "organization_id": "test-org-id",
        "team_alias": "test-team",
        "admins": [],
        "members": [],
        "metadata": {}
    }
    mock_team_row = mocker.MagicMock()
    mock_team_row.model_dump.return_value = mock_team_data
    
    # Mock get_team_object to return our team
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        return_value=mock_team_row
    )
    
    # Mock validation checks
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints.team_call_validation_checks")
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints._validate_team_member_add_permissions")
    
    # Mock organization membership check to return EMPTY list (user not in org)
    mock_prisma_client.db.litellm_organizationmembership.find_many = mocker.AsyncMock(return_value=[])
    
    # Patch prisma client
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    # Request data
    data = TeamMemberAddRequest(
        team_id="org-team-id",
        member=Member(user_id="outsider-user", role="user")
    )
    
    # Call endpoint and expect 403
    with pytest.raises(HTTPException) as exc_info:
        await team_member_add(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role="proxy_admin")
        )
    
    assert exc_info.value.status_code == 403
    assert "is not a member of organization" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_team_member_add_org_validation_success(mocker):
    """
    Test that adding a user to an org-scoped team succeeds if the user is in the org.
    """
    # Mock dependencies
    mock_prisma_client = mocker.MagicMock()
    
    # Mock team with organization_id
    mock_team_data = {
        "team_id": "org-team-id",
        "organization_id": "test-org-id",
        "team_alias": "test-team",
        "admins": [],
        "members": [],
        "metadata": {}
    }
    mock_team_row = mocker.MagicMock()
    mock_team_row.model_dump.return_value = mock_team_data
    
    # Mock get_team_object
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        return_value=mock_team_row
    )
    
    # Mock validation checks
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints.team_call_validation_checks")
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints._validate_team_member_add_permissions")
    
    # Mock organization membership check to return match
    mock_membership = mocker.MagicMock()
    mock_membership.user_id = "insider-user"
    mock_prisma_client.db.litellm_organizationmembership.find_many = mocker.AsyncMock(return_value=[mock_membership])
    
    # Mock successful addition
    mock_updated_team = mocker.MagicMock()
    mock_updated_team.model_dump.return_value = mock_team_data
    
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        return_value=(mock_updated_team, [], [])
    )
    
    # Patch prisma client
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    # Request data
    data = TeamMemberAddRequest(
        team_id="org-team-id",
        member=Member(user_id="insider-user", role="user")
    )
    
    # Call endpoint
    response = await team_member_add(
        data=data,
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role="proxy_admin")
    )
    
    assert response.team_id == "org-team-id"

@pytest.mark.asyncio
async def test_team_member_add_no_org_validation_skipped(mocker):
    """
    Test that organization validation is skipped for teams without organization_id.
    """
    # Mock dependencies
    mock_prisma_client = mocker.MagicMock()
    
    # Mock team WITHOUT organization_id
    mock_team_data = {
        "team_id": "standalone-team-id",
        "organization_id": None,
        "team_alias": "test-team",
        "admins": [],
        "members": [],
        "metadata": {}
    }
    mock_team_row = mocker.MagicMock()
    mock_team_row.model_dump.return_value = mock_team_data
    
    # Mock get_team_object
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        return_value=mock_team_row
    )
    
    # Mock validation checks
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints.team_call_validation_checks")
    mocker.patch("litellm.proxy.management_endpoints.team_endpoints._validate_team_member_add_permissions")
    
    # Mock organization membership check (Should NOT be called, but if it is, let's fail to prove it wasn't used for validation)
    mock_prisma_client.db.litellm_organizationmembership.find_many = mocker.AsyncMock(return_value=[])
    
    # Mock successful addition
    mock_updated_team = mocker.MagicMock()
    mock_updated_team.model_dump.return_value = mock_team_data
    
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        return_value=(mock_updated_team, [], [])
    )
    
    # Patch prisma client
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    # Request data
    data = TeamMemberAddRequest(
        team_id="standalone-team-id",
        member=Member(user_id="any-user", role="user")
    )
    
    # Call endpoint
    response = await team_member_add(
        data=data,
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role="proxy_admin")
    )
    
    # Assert success despite empty membership return
    assert response.team_id == "standalone-team-id"


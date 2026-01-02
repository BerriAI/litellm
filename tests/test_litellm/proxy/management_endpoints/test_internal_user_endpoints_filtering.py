
import pytest
from litellm.proxy.management_endpoints.internal_user_endpoints import ui_view_users
from litellm.proxy._types import UserAPIKeyAuth

@pytest.mark.asyncio
async def test_ui_view_users_filter_by_organization(mocker):
    """
    Test that /user/filter/ui endpoint correctly filters users by organization_id
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()
    
    # Setup mock organization memberships
    mock_membership_1 = mocker.MagicMock()
    mock_membership_1.user_id = "user-in-org-1"
    
    mock_membership_2 = mocker.MagicMock()
    mock_membership_2.user_id = "user-in-org-2"

    async def mock_find_many_memberships(*args, **kwargs):
        # Verify we're looking for the correct organization
        assert kwargs.get("where", {}).get("organization_id") == "test-org-id"
        return [mock_membership_1, mock_membership_2]

    mock_prisma_client.db.litellm_organizationmembership.find_many = mock_find_many_memberships

    # Setup mock users
    mock_user_1 = mocker.MagicMock()
    mock_user_1.model_dump.return_value = {
        "user_id": "user-in-org-1",
        "user_email": "user1@example.com",
        "user_role": "internal_user",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    mock_user_2 = mocker.MagicMock()
    mock_user_2.model_dump.return_value = {
        "user_id": "user-in-org-2",
        "user_email": "user2@example.com",
        "user_role": "internal_user",
        "created_at": "2024-01-01T00:00:00Z"
    }

    async def mock_find_many_users(*args, **kwargs):
        # Verify the user filtering logic
        where = kwargs.get("where", {})
        
        # Should filter by user_id IN [org_members]
        assert "user_id" in where
        assert "in" in where["user_id"]
        assert set(where["user_id"]["in"]) == {"user-in-org-1", "user-in-org-2"}
        
        return [mock_user_1, mock_user_2]

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many_users

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call ui_view_users function with organization_id
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        user_id=None,
        user_email=None,
        organization_id="test-org-id",
        page=1,
        page_size=50
    )

    # Verify response
    assert len(response) == 2
    user_ids = [user.user_id for user in response]
    assert "user-in-org-1" in user_ids
    assert "user-in-org-2" in user_ids

@pytest.mark.asyncio
async def test_ui_view_users_filter_by_organization_empty(mocker):
    """
    Test that /user/filter/ui endpoint returns empty list when organization has no members
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()
    
    # Return empty list for memberships
    async def mock_find_many_memberships(*args, **kwargs):
        return []

    mock_prisma_client.db.litellm_organizationmembership.find_many = mock_find_many_memberships
    
    # Mock users find_many (should not be called if no memberships, but just in case)
    mock_prisma_client.db.litellm_usertable.find_many = mocker.AsyncMock(return_value=[])

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call ui_view_users function
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        user_id=None,
        user_email=None,
        organization_id="empty-org-id",
        page=1,
        page_size=50
    )

    assert response == []

@pytest.mark.asyncio
async def test_ui_view_users_filter_by_organization_and_user_id(mocker):
    """
    Test that /user/filter/ui endpoint correctly combines organization filter with user_id search
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()
    
    # Setup mock organization memberships
    mock_membership_1 = mocker.MagicMock()
    mock_membership_1.user_id = "user-in-org-1"
    
    mock_membership_2 = mocker.MagicMock()
    mock_membership_2.user_id = "user-in-org-2"

    async def mock_find_many_memberships(*args, **kwargs):
        return [mock_membership_1, mock_membership_2]

    mock_prisma_client.db.litellm_organizationmembership.find_many = mock_find_many_memberships

    # Setup mock users
    mock_user_1 = mocker.MagicMock()
    mock_user_1.model_dump.return_value = {
        "user_id": "user-in-org-1",
        "user_email": "user1@example.com",
        "user_role": "internal_user",
        "created_at": "2024-01-01T00:00:00Z"
    }

    async def mock_find_many_users(*args, **kwargs):
        # Verify the user filtering logic
        where = kwargs.get("where", {})
        
        assert "user_id" in where
        user_id_condition = where["user_id"]
        
        # Check combined conditions
        assert "in" in user_id_condition
        assert set(user_id_condition["in"]) == {"user-in-org-1", "user-in-org-2"}
        
        assert "contains" in user_id_condition
        assert user_id_condition["contains"] == "user-in-org-1"
        
        return [mock_user_1]

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many_users

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call ui_view_users function with organization_id AND user_id search
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        user_id="user-in-org-1",
        user_email=None,
        organization_id="test-org-id",
        page=1,
        page_size=50
    )

    # Verify response
    assert len(response) == 1
    assert response[0].user_id == "user-in-org-1"

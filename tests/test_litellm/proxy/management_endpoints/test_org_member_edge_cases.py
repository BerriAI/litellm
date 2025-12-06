"""
Test cases specifically for edge cases around organization membership validation
with varying numbers of members (1 vs 2+) to catch object vs list handling bugs.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmUserRoles,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_team_member_add_org_with_single_member_reject():
    """
    Test rejection when org has EXACTLY 1 member and we try to add someone else.

    This catches bugs where single-item lists might be handled differently than multi-item lists.

    Scenario:
    - Organization has exactly 1 member: "only-member"
    - Team belongs to that organization
    - Try to add "different-user" who is NOT the only member
    - Expected: 403 Forbidden
    """
    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        LiteLLM_TeamMembership,
        LiteLLM_UserTable,
        Member,
    )
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    team_admin = UserAPIKeyAuth(
        user_id="only-member",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="single-member-org-team",
    )

    # Try to add a user who is NOT the single org member
    request_data = TeamMemberAddRequest(
        team_id="single-member-org-team",
        member=Member(
            user_id="different-user",
            role="user",
        ),
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = "single-member-org-team"
    mock_team.organization_id = "single-member-org"
    mock_team.members_with_roles = [
        Member(user_id="only-member", role="admin"),
    ]
    mock_team.metadata = None
    mock_team.model_dump.return_value = {
        "team_id": "single-member-org-team",
        "organization_id": "single-member-org",
        "members_with_roles": [{"user_id": "only-member", "role": "admin"}],
        "metadata": None,
        "spend": 0.0,
    }

    # Mock Membership finding: No members found for "different-user"
    mock_memberships = []

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj"), patch(
        "litellm.proxy.proxy_server.premium_user", True
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        return_value=True,
    ):
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_organizationmembership.find_many = AsyncMock(
            return_value=mock_memberships
        )
        mock_prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])

        # Expected: Should reject because different-user is not in the single-member org
        with pytest.raises(HTTPException) as exc_info:
            await team_member_add(
                data=request_data,
                user_api_key_dict=team_admin,
            )
        assert exc_info.value.status_code == 403
        assert "organization" in str(exc_info.value.detail).lower()
        assert "different-user" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_team_member_add_org_with_single_member_allow():
    """
    Test success when org has EXACTLY 1 member and we add that same member to another team.

    This catches bugs where single-item lists might be handled differently than multi-item lists.

    Scenario:
    - Organization has exactly 1 member: "only-member"
    - Team belongs to that organization
    - Try to add "only-member" (the same user)
    - Expected: Success
    """
    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        LiteLLM_TeamMembership,
        LiteLLM_UserTable,
        Member,
    )
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    team_admin = UserAPIKeyAuth(
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Try to add the only member of the org to a team
    request_data = TeamMemberAddRequest(
        team_id="single-member-org-team",
        member=Member(
            user_id="only-member",
            role="user",
        ),
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = "single-member-org-team"
    mock_team.organization_id = "single-member-org"
    mock_team.members_with_roles = []
    mock_team.metadata = None
    mock_team.model_dump.return_value = {
        "team_id": "single-member-org-team",
        "organization_id": "single-member-org",
        "members_with_roles": [],
        "metadata": None,
        "spend": 0.0,
    }

    # Mock Membership finding: "only-member" IS a member
    mock_membership_record = MagicMock()
    mock_membership_record.user_id = "only-member"
    mock_memberships = [mock_membership_record]

    mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_updated_team.team_id = "single-member-org-team"
    mock_updated_team.organization_id = "single-member-org"
    mock_updated_team.model_dump.return_value = {
        "team_id": "single-member-org-team",
        "organization_id": "single-member-org",
        "spend": 0.0,
    }

    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_id = "only-member"
    mock_user.model_dump.return_value = {"user_id": "only-member"}

    mock_membership = MagicMock(spec=LiteLLM_TeamMembership)
    mock_membership.model_dump.return_value = {
        "user_id": "only-member",
        "team_id": "single-member-org-team",
    }

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj"), patch(
        "litellm.proxy.proxy_server.premium_user", True
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        return_value=True,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new_callable=AsyncMock,
        return_value=(mock_updated_team, [mock_user], [mock_membership]),
    ):
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_organizationmembership.find_many = AsyncMock(
            return_value=mock_memberships
        )

        # Should succeed - only-member is in the single-member org
        result = await team_member_add(
            data=request_data,
            user_api_key_dict=team_admin,
        )

        assert result.team_id == "single-member-org-team"
        assert len(result.updated_users) == 1


@pytest.mark.asyncio
async def test_team_member_add_org_with_empty_users_list():
    """
    Test rejection when org exists but has NO members (empty users list).
    """
    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        Member,
    )
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    team_admin = UserAPIKeyAuth(
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = TeamMemberAddRequest(
        team_id="empty-org-team",
        member=Member(
            user_id="some-user",
            role="user",
        ),
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = "empty-org-team"
    mock_team.organization_id = "empty-org"
    mock_team.members_with_roles = []
    mock_team.metadata = None
    mock_team.model_dump.return_value = {
        "team_id": "empty-org-team",
        "organization_id": "empty-org",
        "members_with_roles": [],
        "metadata": None,
        "spend": 0.0,
    }

    # Mock Membership finding: No members
    mock_memberships = []

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj"), patch(
        "litellm.proxy.proxy_server.premium_user", True
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        return_value=True,
    ):
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_organizationmembership.find_many = AsyncMock(
            return_value=mock_memberships
        )
        mock_prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])

        # Expected: Should reject because org has no members matching the request
        with pytest.raises(HTTPException) as exc_info:
            await team_member_add(
                data=request_data,
                user_api_key_dict=team_admin,
            )
        assert exc_info.value.status_code == 403
        assert "organization" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_team_member_add_org_with_email_validation():
    """
    Test that email-based member addition validates against organization membership correctly.
    """
    from litellm.proxy._types import (
        LiteLLM_TeamMembership,
        LiteLLM_UserTable,
        Member,
    )
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    team_admin = UserAPIKeyAuth(
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Try to add a user by email
    request_data = TeamMemberAddRequest(
        team_id="email-org-team",
        member=Member(
            user_email="valid@example.com",
            role="user",
        ),
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = "email-org-team"
    mock_team.organization_id = "email-org"
    mock_team.members_with_roles = []
    mock_team.metadata = None
    mock_team.model_dump.return_value = {
        "team_id": "email-org-team",
        "organization_id": "email-org",
        "members_with_roles": [],
        "metadata": None,
        "spend": 0.0,
    }

    # Mock finding user by email in org
    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_email = "valid@example.com"
    mock_user.user_id = "valid-user-id"
    mock_user.model_dump.return_value = {
        "user_id": "valid-user-id",
        "user_email": "valid@example.com",
    }

    mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_updated_team.team_id = "email-org-team"
    mock_updated_team.organization_id = "email-org"
    mock_updated_team.model_dump.return_value = {"team_id": "email-org-team"}

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj"), patch(
        "litellm.proxy.proxy_server.premium_user", True
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        return_value=True,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new_callable=AsyncMock,
        return_value=(mock_updated_team, [mock_user], []),
    ):
        mock_prisma.db = MagicMock()
        # ID check returns empty
        mock_prisma.db.litellm_organizationmembership.find_many = AsyncMock(
            return_value=[]
        )
        # Email check returns the user
        mock_prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[mock_user])

        result = await team_member_add(
            data=request_data,
            user_api_key_dict=team_admin,
        )
        assert result.team_id == "email-org-team"

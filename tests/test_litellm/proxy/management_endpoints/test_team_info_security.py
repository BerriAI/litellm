"""
Test for team info access control - ensures only team admins can view team info
"""
import pytest
from fastapi import HTTPException
from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_TeamTable, Member, LitellmUserRoles
from litellm.proxy.management_endpoints.team_endpoints import validate_team_admin


@pytest.mark.parametrize(
    "user_role,expected_pass",
    [
        (LitellmUserRoles.PROXY_ADMIN.value, True),
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, True),
    ],
)
def test_validate_team_admin_proxy_admins(user_role, expected_pass):
    """Proxy admins should always have access"""
    user_api_key_dict = UserAPIKeyAuth(user_role=user_role)
    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        members_with_roles=[]
    )
    
    if expected_pass:
        validate_team_admin(user_api_key_dict, team_table)
    else:
        with pytest.raises(HTTPException):
            validate_team_admin(user_api_key_dict, team_table)


@pytest.mark.parametrize(
    "user_role,member_role,should_pass",
    [
        (LitellmUserRoles.INTERNAL_USER.value, "admin", True),   # Team admin can access
        (LitellmUserRoles.INTERNAL_USER.value, "user", False),   # Regular user blocked
    ],
)
def test_validate_team_admin_member_roles(user_role, member_role, should_pass):
    """Test different team member roles"""
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test-user",
        user_role=user_role
    )
    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        members_with_roles=[
            Member(user_id="test-user", role=member_role)
        ]
    )
    
    if should_pass:
        validate_team_admin(user_api_key_dict, team_table)
    else:
        with pytest.raises(HTTPException) as exc_info:
            validate_team_admin(user_api_key_dict, team_table)
        assert exc_info.value.status_code == 403
        assert "must be team admin" in str(exc_info.value.detail)


def test_validate_team_admin_non_member():
    """Non-members should be blocked"""
    user_api_key_dict = UserAPIKeyAuth(
        user_id="outsider-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value
    )
    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        members_with_roles=[
            Member(user_id="admin-user", role="admin")
        ]
    )
    
    with pytest.raises(HTTPException) as exc_info:
        validate_team_admin(user_api_key_dict, team_table)
    
    assert exc_info.value.status_code == 403
    assert "not authorized" in str(exc_info.value.detail)


def test_validate_team_admin_team_key_blocked():
    """Team keys accessing other teams should be blocked"""
    user_api_key_dict = UserAPIKeyAuth(
        user_id=None,
        team_id="other-team",
        user_role=LitellmUserRoles.INTERNAL_USER.value
    )
    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        members_with_roles=[]
    )
    
    with pytest.raises(HTTPException) as exc_info:
        validate_team_admin(user_api_key_dict, team_table)
    
    assert exc_info.value.status_code == 403
    assert "Team keys cannot access other teams" in str(exc_info.value.detail)

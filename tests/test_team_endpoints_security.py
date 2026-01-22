"""
Unit tests for team endpoint security fix
Tests that non-admin users cannot access sensitive team information
"""
import pytest
from litellm.proxy._types import (
    UserAPIKeyAuth,
    LitellmUserRoles,
    LiteLLM_TeamTable,
    Member,
)
from litellm.proxy.management_endpoints.common_utils import (
    _user_has_admin_view,
    _is_user_team_admin,
)


class TestTeamEndpointsSecurity:
    """Test team endpoints security"""

    @pytest.fixture
    def mock_team(self):
        """Mock team with sensitive data"""
        return LiteLLM_TeamTable(
            team_id="test_team_id",
            team_alias="test_team",
            members_with_roles=[
                Member(user_id="admin_user", role="admin"),
                Member(user_id="regular_user", role="user"),
                Member(user_id="other_user", role="user"),
            ],
            team_member_permissions=["read", "write"],
            max_budget=1000.0,
            spend=0.0,
        )

    @pytest.fixture
    def admin_user(self):
        """Admin user with PROXY_ADMIN role"""
        return UserAPIKeyAuth(
            user_id="admin_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
        )

    @pytest.fixture
    def regular_user(self):
        """Regular user with INTERNAL_USER role"""
        return UserAPIKeyAuth(
            user_id="regular_user",
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key="sk-regular",
        )

    @pytest.fixture
    def team_admin_user(self):
        """Team admin user (admin role in team, but INTERNAL_USER system role)"""
        return UserAPIKeyAuth(
            user_id="admin_user",
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key="sk-team-admin",
        )

    @pytest.fixture
    def org_admin_user(self):
        """Organization admin user"""
        return UserAPIKeyAuth(
            user_id="org_admin",
            user_role=LitellmUserRoles.ORG_ADMIN,
            api_key="sk-org-admin",
        )

    @pytest.mark.parametrize(
        "user_fixture,expected_authorized",
        [
            ("admin_user", True),
            ("org_admin_user", True),
            ("team_admin_user", True),
            ("regular_user", False),
        ],
    )
    def test_authorization_logic(
        self, user_fixture, expected_authorized, mock_team, request
    ):
        """Test authorization logic for different user roles"""
        user = request.getfixturevalue(user_fixture)
        
        is_authorized = (
            _user_has_admin_view(user) or 
            user.user_role == LitellmUserRoles.ORG_ADMIN or
            _is_user_team_admin(user, mock_team)
        )
        
        assert is_authorized == expected_authorized

    @pytest.mark.parametrize(
        "user_fixture,should_see_data",
        [
            ("admin_user", True),
            ("org_admin_user", True),
            ("team_admin_user", True),
            ("regular_user", False),
        ],
    )
    def test_convert_teams_to_response(
        self, user_fixture, should_see_data, mock_team, request
    ):
        """Test _convert_teams_to_response filters correctly based on user role"""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _convert_teams_to_response,
        )

        user = request.getfixturevalue(user_fixture)
        teams = [mock_team]
        result = _convert_teams_to_response(teams, False, user)

        assert len(result) == 1
        
        if should_see_data:
            # Authorized users should see all fields
            assert len(result[0].members_with_roles) == 3
            assert result[0].team_member_permissions == ["read", "write"]
        else:
            # Non-authorized users should NOT see sensitive fields
            assert result[0].members_with_roles == []
            assert result[0].team_member_permissions is None or result[0].team_member_permissions == []

    @pytest.mark.parametrize(
        "user_fixture,should_retain_fields",
        [
            ("admin_user", True),
            ("regular_user", False),
        ],
    )
    def test_field_filtering_logic(
        self, user_fixture, should_retain_fields, mock_team, request
    ):
        """Test that fields are properly filtered based on authorization"""
        user = request.getfixturevalue(user_fixture)
        team_dict = mock_team.model_dump()
        
        # Simulate the authorization check
        is_authorized = (
            _user_has_admin_view(user) or 
            user.user_role == LitellmUserRoles.ORG_ADMIN or
            _is_user_team_admin(user, mock_team)
        )
        
        # Apply the filtering logic
        if not is_authorized:
            team_dict.pop("members_with_roles", None)
            team_dict.pop("team_member_permissions", None)
        
        if should_retain_fields:
            # Authorized users should retain fields
            assert "members_with_roles" in team_dict
            assert len(team_dict["members_with_roles"]) == 3
            assert "team_member_permissions" in team_dict
        else:
            # Non-authorized users should have fields removed
            assert "members_with_roles" not in team_dict
            assert "team_member_permissions" not in team_dict


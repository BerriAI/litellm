"""
Tests for org admin access to team management endpoints.

Covers:
- _is_user_org_admin_for_team helper
- validate_membership allowing org admins
- _user_is_org_admin route-level check (no privilege escalation)
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.proxy._types import (
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_team(team_id="team-1", organization_id="org-1") -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        team_alias="Test Team",
        organization_id=organization_id,
        members_with_roles=[
            Member(user_id="direct-member", role="user"),
            Member(user_id="team-admin", role="admin"),
        ],
    )


def _make_user_key(
    user_id="org-admin-user", role=LitellmUserRoles.INTERNAL_USER.value
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, user_role=role)


def _make_membership(user_id, org_id, role="org_admin"):
    return LiteLLM_OrganizationMembershipTable(
        user_id=user_id,
        organization_id=org_id,
        user_role=role,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_caller_user(
    user_id="org-admin-user", org_id="org-1", org_role="org_admin"
) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        organization_memberships=[_make_membership(user_id, org_id, org_role)],
    )


def _patch_org_admin_deps(get_user_return):
    """Context manager that patches the lazy imports inside _is_user_org_admin_for_team."""
    return (
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            new_callable=AsyncMock,
            return_value=get_user_return,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock(), create=True),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock(), create=True),
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock(), create=True
        ),
    )


# ---------------------------------------------------------------------------
# _is_user_org_admin_for_team
# ---------------------------------------------------------------------------


class TestIsUserOrgAdminForTeam:
    """Tests for the reusable _is_user_org_admin_for_team helper."""

    @pytest.mark.asyncio
    async def test_org_admin_for_teams_org_returns_true(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id="org-admin-user")
        caller = _make_caller_user(user_id="org-admin-user", org_id="org-1")

        p1, p2, p3, p4 = _patch_org_admin_deps(caller)
        with p1, p2, p3, p4:
            result = await _is_user_org_admin_for_team(
                user_api_key_dict=key, team_obj=team
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_org_admin_different_org_returns_false(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id="other-admin")
        caller = _make_caller_user(user_id="other-admin", org_id="org-2")

        p1, p2, p3, p4 = _patch_org_admin_deps(caller)
        with p1, p2, p3, p4:
            result = await _is_user_org_admin_for_team(
                user_api_key_dict=key, team_obj=team
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_team_without_org_returns_false(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = _make_team(organization_id=None)
        key = _make_user_key()
        result = await _is_user_org_admin_for_team(user_api_key_dict=key, team_obj=team)
        assert result is False

    @pytest.mark.asyncio
    async def test_org_member_not_admin_returns_false(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id="regular")
        caller = _make_caller_user(user_id="regular", org_id="org-1", org_role="user")

        p1, p2, p3, p4 = _patch_org_admin_deps(caller)
        with p1, p2, p3, p4:
            result = await _is_user_org_admin_for_team(
                user_api_key_dict=key, team_obj=team
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_no_user_id_returns_false(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id=None)
        result = await _is_user_org_admin_for_team(user_api_key_dict=key, team_obj=team)
        assert result is False


# ---------------------------------------------------------------------------
# validate_membership
# ---------------------------------------------------------------------------


class TestValidateMembership:
    """Tests for validate_membership with org admin support."""

    @pytest.mark.asyncio
    async def test_proxy_admin_allowed(self):
        from litellm.proxy.management_endpoints.team_endpoints import (
            validate_membership,
        )

        team = _make_team()
        key = _make_user_key(user_id="admin", role=LitellmUserRoles.PROXY_ADMIN.value)
        await validate_membership(user_api_key_dict=key, team_table=team)

    @pytest.mark.asyncio
    async def test_direct_team_member_allowed(self):
        from litellm.proxy.management_endpoints.team_endpoints import (
            validate_membership,
        )

        team = _make_team()
        key = _make_user_key(user_id="direct-member")
        await validate_membership(user_api_key_dict=key, team_table=team)

    @pytest.mark.asyncio
    async def test_org_admin_for_team_org_allowed(self):
        from litellm.proxy.management_endpoints.team_endpoints import (
            validate_membership,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id="org-admin-user")
        caller = _make_caller_user(user_id="org-admin-user", org_id="org-1")

        p1, p2, p3, p4 = _patch_org_admin_deps(caller)
        with p1, p2, p3, p4:
            await validate_membership(user_api_key_dict=key, team_table=team)

    @pytest.mark.asyncio
    async def test_non_member_non_org_admin_rejected(self):
        from fastapi import HTTPException
        from litellm.proxy.management_endpoints.team_endpoints import (
            validate_membership,
        )

        team = _make_team(organization_id="org-1")
        key = _make_user_key(user_id="random-user")
        caller = _make_caller_user(
            user_id="random-user", org_id="org-2", org_role="user"
        )

        p1, p2, p3, p4 = _patch_org_admin_deps(caller)
        with p1, p2, p3, p4:
            with pytest.raises(HTTPException) as exc_info:
                await validate_membership(user_api_key_dict=key, team_table=team)
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_team_key_matches_team_allowed(self):
        from litellm.proxy.management_endpoints.team_endpoints import (
            validate_membership,
        )

        team = _make_team(team_id="team-1")
        key = UserAPIKeyAuth(
            team_id="team-1", user_role=LitellmUserRoles.INTERNAL_USER.value
        )
        await validate_membership(user_api_key_dict=key, team_table=team)


# ---------------------------------------------------------------------------
# _user_is_org_admin (route-level) — no privilege escalation
# ---------------------------------------------------------------------------


class TestUserIsOrgAdminRouteCheck:
    """
    Verify that _user_is_org_admin does NOT grant blanket access
    when no organization_id is in the request body.
    """

    def test_no_candidate_org_ids_returns_false(self):
        from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin

        user = LiteLLM_UserTable(
            user_id="org-admin-user",
            organization_memberships=[_make_membership("org-admin-user", "org-1")],
        )
        result = _user_is_org_admin(request_data={}, user_object=user)
        assert result is False, "Must NOT grant blanket access when no org in request"

    def test_matching_org_id_returns_true(self):
        from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin

        user = LiteLLM_UserTable(
            user_id="org-admin-user",
            organization_memberships=[_make_membership("org-admin-user", "org-1")],
        )
        result = _user_is_org_admin(
            request_data={"organization_id": "org-1"}, user_object=user
        )
        assert result is True

    def test_non_matching_org_id_returns_false(self):
        from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin

        user = LiteLLM_UserTable(
            user_id="org-admin-user",
            organization_memberships=[_make_membership("org-admin-user", "org-1")],
        )
        result = _user_is_org_admin(
            request_data={"organization_id": "org-99"}, user_object=user
        )
        assert result is False

    def test_organizations_list_field(self):
        from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin

        user = LiteLLM_UserTable(
            user_id="org-admin-user",
            organization_memberships=[_make_membership("org-admin-user", "org-1")],
        )
        result = _user_is_org_admin(
            request_data={"organizations": ["org-1"]}, user_object=user
        )
        assert result is True

    def test_none_user_object_returns_false(self):
        from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin

        result = _user_is_org_admin(request_data={}, user_object=None)
        assert result is False

    def test_user_list_in_self_managed_routes(self):
        """Verify /user/list is in self_managed_routes so org admins can reach it."""
        from litellm.proxy._types import LiteLLMRoutes

        assert "/user/list" in LiteLLMRoutes.self_managed_routes.value

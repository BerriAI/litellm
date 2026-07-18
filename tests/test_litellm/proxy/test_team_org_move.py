"""
Tests for moving teams to organizations.

Covers the SSO/Entra scenario where:
- Proxy admins can move teams freely; missing members are auto-added to the org.
- Non-proxy-admins (team admins) must have all team members pre-added to the org,
  preserving the original security model (no privilege escalation via team move).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import (
    LiteLLM_OrganizationTableWithMembers,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    OrgMember,
    SpecialProxyStrings,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    _auto_add_team_members_to_organization,
    validate_team_org_change,
)
from litellm.router import Router


def _make_org(organization_id="org-1", members=None, models=None):
    from datetime import datetime

    return LiteLLM_OrganizationTableWithMembers(
        organization_id=organization_id,
        organization_alias="test-org",
        budget_id="budget-test",
        spend=0.0,
        metadata={},
        models=models or [],
        created_by="default_user_id",
        updated_by="default_user_id",
        members=members or [],
        teams=[],
        litellm_budget_table=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def _make_team(team_id="team-1", member_ids=None, organization_id=None):
    members = [
        Member(user_id=uid, role="user") for uid in (member_ids or [])
    ]
    members.append(Member(user_id=SpecialProxyStrings.default_user_id.value, role="admin"))
    return LiteLLM_TeamTable(
        team_id=team_id,
        team_alias="test-team",
        organization_id=organization_id,
        admins=[],
        members=[],
        members_with_roles=members,
        metadata={},
        models=[],
        blocked=False,
        spend=0.0,
    )


def _make_org_membership(user_id):
    from datetime import datetime

    return LiteLLM_OrganizationMembershipTable(
        user_id=user_id,
        organization_id="org-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        spend=0.0,
        budget_id=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


class TestValidateTeamOrgChange:
    def test_proxy_admin_not_blocked_when_members_not_in_org(self):
        """Proxy admins bypass the membership check — auto-add handles it instead."""
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["sso-user-001", "sso-user-002"])
        org = _make_org(members=[])

        result = validate_team_org_change(
            team=team, organization=org, llm_router=router, is_proxy_admin=True
        )
        assert result is True

    def test_non_admin_blocked_when_members_not_in_org(self):
        """Team admins (non-proxy-admin) must have all members pre-added to the org."""
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["sso-user-001"])
        org = _make_org(members=[])

        with pytest.raises(Exception) as exc_info:
            validate_team_org_change(
                team=team, organization=org, llm_router=router, is_proxy_admin=False
            )
        assert "403" in str(exc_info.value) or "not a member" in str(exc_info.value)

    def test_non_admin_passes_when_all_members_in_org(self):
        """Team admin move succeeds when all team members are already org members."""
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["u1"])
        org = _make_org(members=[_make_org_membership("u1")])

        result = validate_team_org_change(
            team=team, organization=org, llm_router=router, is_proxy_admin=False
        )
        assert result is True

    def test_same_org_short_circuits(self):
        """Moving to the same org is always a no-op, regardless of role."""
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["u1"], organization_id="org-1")
        org = _make_org(organization_id="org-1")

        assert validate_team_org_change(
            team=team, organization=org, llm_router=router, is_proxy_admin=False
        ) is True
        assert validate_team_org_change(
            team=team, organization=org, llm_router=router, is_proxy_admin=True
        ) is True

    def test_default_user_excluded_from_membership_check(self):
        """default_user_id is never checked for org membership."""
        router = MagicMock(spec=Router)
        # Team has only default_user_id (added by _make_team)
        team = _make_team(member_ids=[])
        org = _make_org(members=[])

        # Should not raise even for non-proxy-admin
        result = validate_team_org_change(
            team=team, organization=org, llm_router=router, is_proxy_admin=False
        )
        assert result is True


class TestAutoAddTeamMembersToOrg:
    @pytest.mark.asyncio
    async def test_adds_missing_members(self):
        team = _make_team(member_ids=["sso-user-001", "sso-user-002"])
        org = _make_org(members=[])

        mock_add = AsyncMock()
        import litellm.proxy.management_endpoints.team_endpoints as te
        original = te.add_member_to_organization
        te.add_member_to_organization = mock_add

        try:
            await _auto_add_team_members_to_organization(
                team=team,
                organization=org,
                prisma_client=MagicMock(),
            )
        finally:
            te.add_member_to_organization = original

        assert mock_add.call_count == 2
        called_user_ids = {
            call.kwargs["member"].user_id for call in mock_add.call_args_list
        }
        assert called_user_ids == {"sso-user-001", "sso-user-002"}

    @pytest.mark.asyncio
    async def test_skips_existing_org_members(self):
        team = _make_team(member_ids=["u1", "u2"])
        org = _make_org(members=[_make_org_membership("u1")])

        mock_add = AsyncMock()
        import litellm.proxy.management_endpoints.team_endpoints as te
        original = te.add_member_to_organization
        te.add_member_to_organization = mock_add

        try:
            await _auto_add_team_members_to_organization(
                team=team,
                organization=org,
                prisma_client=MagicMock(),
            )
        finally:
            te.add_member_to_organization = original

        assert mock_add.call_count == 1
        assert mock_add.call_args.kwargs["member"].user_id == "u2"

    @pytest.mark.asyncio
    async def test_skips_default_user(self):
        """default_user_id should never be added as an org member."""
        team = _make_team(member_ids=[])
        org = _make_org(members=[])

        mock_add = AsyncMock()
        import litellm.proxy.management_endpoints.team_endpoints as te
        original = te.add_member_to_organization
        te.add_member_to_organization = mock_add

        try:
            await _auto_add_team_members_to_organization(
                team=team,
                organization=org,
                prisma_client=MagicMock(),
            )
        finally:
            te.add_member_to_organization = original

        assert mock_add.call_count == 0

    @pytest.mark.asyncio
    async def test_logs_and_continues_on_error(self):
        """Errors must not propagate — they are logged at DEBUG and skipped."""
        team = _make_team(member_ids=["u1"])
        org = _make_org(members=[])

        mock_add = AsyncMock(side_effect=Exception("duplicate key"))
        import litellm.proxy.management_endpoints.team_endpoints as te
        original = te.add_member_to_organization
        te.add_member_to_organization = mock_add

        try:
            await _auto_add_team_members_to_organization(
                team=team,
                organization=org,
                prisma_client=MagicMock(),
            )
        finally:
            te.add_member_to_organization = original

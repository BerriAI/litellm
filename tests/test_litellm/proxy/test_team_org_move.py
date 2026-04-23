"""
Tests for moving teams to organizations.

Covers the SSO/Entra scenario where team members are not pre-added as org members
and the move should auto-add them rather than blocking.
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
    def test_no_block_when_team_members_not_in_org(self):
        """The membership check must NOT block the move."""
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["sso-user-001", "sso-user-002"])
        org = _make_org(members=[])  # org has zero members

        # Should not raise
        result = validate_team_org_change(team=team, organization=org, llm_router=router)
        assert result is True

    def test_same_org_short_circuits(self):
        router = MagicMock(spec=Router)
        team = _make_team(member_ids=["u1"], organization_id="org-1")
        org = _make_org(organization_id="org-1")

        assert validate_team_org_change(team=team, organization=org, llm_router=router) is True


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

        # Should have been called once per non-default SSO user
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

        # Only u2 should be added; u1 is already in org
        assert mock_add.call_count == 1
        assert mock_add.call_args.kwargs["member"].user_id == "u2"

    @pytest.mark.asyncio
    async def test_skips_default_user(self):
        """default_user_id should never be added as an org member."""
        team = _make_team(member_ids=[])  # only has default_user_id from _make_team
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
    async def test_silently_ignores_errors(self):
        """Errors (e.g. duplicate key) must not propagate."""
        team = _make_team(member_ids=["u1"])
        org = _make_org(members=[])

        mock_add = AsyncMock(side_effect=Exception("duplicate key"))
        import litellm.proxy.management_endpoints.team_endpoints as te
        original = te.add_member_to_organization
        te.add_member_to_organization = mock_add

        try:
            # Should not raise
            await _auto_add_team_members_to_organization(
                team=team,
                organization=org,
                prisma_client=MagicMock(),
            )
        finally:
            te.add_member_to_organization = original

"""
Tests for org-admin team-list scoping (LIT-3723).

A user who is org admin of one org must still see teams they belong to in
*other* orgs (where they are only a member). PR #25904 clamped a self-query to
the caller's org-admin orgs, dropping those memberships. The fix is strictly
additive: a self/bare org-admin query unions org teams with the caller's own
memberships, while a cross-user org-admin query keeps the org-boundary
intersection (#25904).

These tests are self-contained — they mock `get_user_object`, `TeamRepository`,
and the access-gate helpers rather than touching the shared seed fixture in
tests/proxy_behavior/management/ (which ~8 suites parametrize over).
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.proxy._types import (
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

_MOD = "litellm.proxy.management_endpoints.team_endpoints"
_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeTeam:
    """Lightweight stand-in for a Prisma team row.

    `members_with_roles` is a list of plain dicts so `m.get("user_id")` (the
    production filter) works without pydantic coercion.
    """

    def __init__(self, team_id, organization_id, member_user_ids):
        self.team_id = team_id
        self.organization_id = organization_id
        self.members_with_roles = [{"user_id": uid, "role": "user"} for uid in member_user_ids]


def _user_with_teams(user_id="x", teams=None) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(user_id=user_id, teams=teams or [])


def _org_admin_caller(user_id="x", org_id="org-a") -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id=user_id,
                organization_id=org_id,
                user_role=LitellmUserRoles.ORG_ADMIN.value,
                created_at=_NOW,
                updated_at=_NOW,
            )
        ],
    )


def _internal_key(user_id="x") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER.value)


def _make_team_repo(all_teams):
    """Return a TeamRepository factory whose `.table.find_many` honors `where`.

    No `where` (self-query full scan) returns every team; an
    `organization_id: {in: [...]}` filter returns only teams in those orgs.
    """

    async def _find_many(*args, **kwargs):
        where = kwargs.get("where")
        if not where:
            return list(all_teams)
        org_ids = where.get("organization_id", {}).get("in", [])
        return [t for t in all_teams if t.organization_id in org_ids]

    instance = MagicMock()
    instance.table.find_many = AsyncMock(side_effect=_find_many)
    return MagicMock(return_value=instance)


async def _build_where(**overrides):
    from litellm.proxy.management_endpoints.team_endpoints import (
        _build_team_list_where_conditions,
    )

    kwargs = dict(
        prisma_client=MagicMock(),
        team_id=None,
        team_alias=None,
        organization_id=None,
        user_id="x",
        use_deleted_table=False,
        search=None,
        org_admin_org_ids=["org-a"],
        union_org_admin_membership=True,
        user_api_key_cache=MagicMock(),
        proxy_logging_obj=MagicMock(),
    )
    kwargs.update(overrides)
    return await _build_team_list_where_conditions(**kwargs)


# ---------------------------------------------------------------------------
# _build_team_list_where_conditions — union semantics
# ---------------------------------------------------------------------------


class TestBuildWhereOrgAdminUnion:
    @pytest.mark.asyncio
    async def test_self_query_emits_union(self):
        """Self/bare org admin → OR(org teams, memberships), no sibling org clamp."""
        user = _user_with_teams("x", ["team-2"])  # team-2 lives in org-b
        with patch(f"{_MOD}.get_user_object", new_callable=AsyncMock, return_value=user):
            where = await _build_where()

        assert where is not None
        assert where["OR"] == [
            {"organization_id": {"in": ["org-a"]}},
            {"team_id": {"in": ["team-2"]}},
        ]
        # No sibling organization_id clamp — that would re-AND the intersection.
        assert "organization_id" not in where

    @pytest.mark.asyncio
    async def test_self_query_no_memberships_keeps_org_teams(self):
        """Org admin with no memberships still sees org teams; does NOT return None."""
        user = _user_with_teams("x", [])
        with patch(f"{_MOD}.get_user_object", new_callable=AsyncMock, return_value=user):
            where = await _build_where()

        assert where is not None
        assert where["OR"] == [{"organization_id": {"in": ["org-a"]}}]

    @pytest.mark.asyncio
    async def test_cross_user_query_keeps_intersection(self):
        """union flag off → sibling org clamp AND membership filter (the #25904 boundary)."""
        target = _user_with_teams("target", ["team-2"])
        with patch(f"{_MOD}.get_user_object", new_callable=AsyncMock, return_value=target):
            where = await _build_where(user_id="target", union_org_admin_membership=False)

        assert where is not None
        assert where["organization_id"] == {"in": ["org-a"]}
        assert where["team_id"] == {"in": ["team-2"]}
        assert "OR" not in where

    @pytest.mark.asyncio
    async def test_search_co_exists_with_union(self):
        """search OR and union OR are both preserved, ANDed at the top level."""
        user = _user_with_teams("x", ["team-2"])
        with patch(f"{_MOD}.get_user_object", new_callable=AsyncMock, return_value=user):
            where = await _build_where(search="foo")

        assert where is not None
        assert where["AND"] == [
            {
                "OR": [
                    {"team_id": "foo"},
                    {"team_alias": {"contains": "foo", "mode": "insensitive"}},
                ]
            },
            {
                "OR": [
                    {"organization_id": {"in": ["org-a"]}},
                    {"team_id": {"in": ["team-2"]}},
                ]
            },
        ]
        assert "OR" not in where


# ---------------------------------------------------------------------------
# _enforce_list_team_v2_access — flag + boundary gates
# ---------------------------------------------------------------------------


class TestEnforceListTeamV2Access:
    async def _call(self, user_id, organization_id=None, org_admin_org_ids=["org-a"]):
        from litellm.proxy.management_endpoints.team_endpoints import (
            _enforce_list_team_v2_access,
        )

        with patch(
            f"{_MOD}._get_org_admin_org_ids",
            new_callable=AsyncMock,
            return_value=org_admin_org_ids,
        ):
            return await _enforce_list_team_v2_access(
                user_api_key_dict=_internal_key("x"),
                user_id=user_id,
                organization_id=organization_id,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_bare_self_query_sets_flag(self):
        uid, orgs, union = await self._call(user_id=None)
        assert (uid, orgs, union) == ("x", ["org-a"], True)

    @pytest.mark.asyncio
    async def test_explicit_self_query_sets_flag(self):
        uid, orgs, union = await self._call(user_id="x")
        assert (uid, orgs, union) == ("x", ["org-a"], True)

    @pytest.mark.asyncio
    async def test_cross_user_query_clears_flag(self):
        uid, orgs, union = await self._call(user_id="other")
        assert (uid, orgs, union) == ("other", ["org-a"], False)

    @pytest.mark.asyncio
    async def test_foreign_org_filter_is_403(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await self._call(user_id=None, organization_id="org-z")
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_non_org_admin_bare_is_401(self):
        from fastapi import HTTPException

        with patch(f"{_MOD}.allowed_route_check_inside_route", return_value=False):
            with pytest.raises(HTTPException) as exc:
                await self._call(user_id=None, org_admin_org_ids=None)
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# _authorize_and_filter_teams (v1) — self-query union vs cross-user intersection
# ---------------------------------------------------------------------------


class TestAuthorizeAndFilterTeams:
    async def _call(self, user_id, all_teams):
        from litellm.proxy.management_endpoints.team_endpoints import (
            _authorize_and_filter_teams,
        )

        caller = _org_admin_caller("x", "org-a")
        with (
            patch(f"{_MOD}.get_user_object", new_callable=AsyncMock, return_value=caller),
            patch(f"{_MOD}.TeamRepository", _make_team_repo(all_teams)),
        ):
            return await _authorize_and_filter_teams(
                user_api_key_dict=_internal_key("x"),
                user_id=user_id,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_self_query_includes_cross_org_membership(self):
        """Org admin of org-a, member of a team in org-b, self-query → both teams."""
        team_a = _FakeTeam("team-1", "org-a", member_user_ids=["x"])
        team_b = _FakeTeam("team-2", "org-b", member_user_ids=["x"])
        other = _FakeTeam("team-3", "org-c", member_user_ids=["someone-else"])

        result = await self._call(user_id="x", all_teams=[team_a, team_b, other])

        ids = {t.team_id for t in result}
        assert ids == {"team-1", "team-2"}  # org-b membership survives

    @pytest.mark.asyncio
    async def test_cross_user_query_keeps_org_boundary(self):
        """Cross-user query → only org-a teams the target is a member of."""
        team_a = _FakeTeam("team-1", "org-a", member_user_ids=["target"])
        team_b = _FakeTeam("team-2", "org-b", member_user_ids=["target"])

        result = await self._call(user_id="target", all_teams=[team_a, team_b])

        ids = {t.team_id for t in result}
        assert ids == {"team-1"}  # team-2 (org-b) is NOT leaked

    @pytest.mark.asyncio
    async def test_bare_query_returns_all_org_teams(self):
        """Bare org-admin query (no user_id) → every team in the admin's orgs."""
        team_a1 = _FakeTeam("team-1", "org-a", member_user_ids=["someone"])
        team_a2 = _FakeTeam("team-2", "org-a", member_user_ids=[])
        team_b = _FakeTeam("team-3", "org-b", member_user_ids=["x"])

        result = await self._call(user_id=None, all_teams=[team_a1, team_a2, team_b])

        ids = {t.team_id for t in result}
        assert ids == {"team-1", "team-2"}  # all org-a teams, no org-b

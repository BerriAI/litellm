"""
Unit tests for the VERIA-55 fixes:

- Project update permission must be evaluated against the project's *current*
  team, not a team supplied in the request body.
- Key update may not assign a key to an organization the caller is not a
  member of.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# ---------------------------------------------------------------------------
# /project/update — _check_user_permission_for_project
# ---------------------------------------------------------------------------


def _make_prisma_with_team(team_id: str, admins: list):
    prisma = MagicMock()
    team_row = MagicMock()
    team_row.team_id = team_id
    team_row.admins = admins
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    return prisma


@pytest.mark.asyncio
async def test_project_perm_check_uses_current_team_not_caller_supplied():
    """The permission check must look at the project's existing team. Even
    if the caller is admin of an unrelated team, they must not pass when no
    explicit team_object is forced through."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_user_permission_for_project,
    )

    # Project lives on team-A, caller is admin only of team-B.
    prisma = _make_prisma_with_team(team_id="team-A", admins=["alice"])
    caller = UserAPIKeyAuth(
        user_id="bob",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    has_perm = await _check_user_permission_for_project(
        user_api_key_dict=caller,
        team_id="team-A",
        prisma_client=prisma,
    )
    assert has_perm is False
    prisma.db.litellm_teamtable.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_project_perm_check_allows_team_admin_of_existing_team():
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_user_permission_for_project,
    )

    prisma = _make_prisma_with_team(team_id="team-A", admins=["alice"])
    alice = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    has_perm = await _check_user_permission_for_project(
        user_api_key_dict=alice,
        team_id="team-A",
        prisma_client=prisma,
    )
    assert has_perm is True


@pytest.mark.asyncio
async def test_project_perm_check_proxy_admin_always_allowed():
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_user_permission_for_project,
    )

    prisma = MagicMock()
    admin = UserAPIKeyAuth(
        user_id="root",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    has_perm = await _check_user_permission_for_project(
        user_api_key_dict=admin,
        team_id="team-A",
        prisma_client=prisma,
    )
    assert has_perm is True
    # Admin shortcut should not even hit the DB.
    prisma.db.litellm_teamtable.find_unique.assert_not_called()


# ---------------------------------------------------------------------------
# /key/update — _validate_caller_can_assign_key_org
# ---------------------------------------------------------------------------


def _make_prisma_with_user_orgs(user_id: str, org_ids: list):
    prisma = MagicMock()
    user_row = MagicMock()
    user_row.organization_memberships = [
        MagicMock(organization_id=org_id) for org_id in org_ids
    ]
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)
    return prisma


@pytest.mark.asyncio
async def test_assign_key_org_allows_member():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = _make_prisma_with_user_orgs("alice", ["org-1", "org-2"])
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    # Should not raise.
    await _validate_caller_can_assign_key_org(
        user_api_key_dict=caller,
        organization_id="org-2",
        prisma_client=prisma,
    )


@pytest.mark.asyncio
async def test_assign_key_org_blocks_non_member():
    """The IDOR: caller asks to point a key at an org they don't belong to."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = _make_prisma_with_user_orgs("alice", ["org-1"])
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="someone-elses-org",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
    assert "someone-elses-org" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_assign_key_org_blocks_caller_without_user_id():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = MagicMock()
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_assign_key_org_blocks_caller_with_no_memberships():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = MagicMock()
    user_row = MagicMock()
    user_row.organization_memberships = None
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)

    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403



# ---------------------------------------------------------------------------
# LIT-3214 — _validate_caller_can_assign_key_org transitive team membership
# ---------------------------------------------------------------------------


def _make_prisma_with_user_teams_and_team_orgs(
    user_id: str,
    org_memberships: list,
    team_ids: list,
    team_org_map: dict,
    monkeypatch=None,
):
    """Build a prisma mock where the user belongs to `team_ids` and each
    team in `team_org_map` has the given `organization_id`. Returns
    ``(prisma, get_team_object_mock)``.

    The fix uses ``get_team_object`` (cached lookup) per team_id, so we patch
    that function instead of the underlying prisma call.
    """
    prisma = MagicMock()

    user_row = MagicMock()
    user_row.organization_memberships = [
        MagicMock(organization_id=org_id) for org_id in org_memberships
    ]
    user_row.teams = list(team_ids)
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)

    async def _get_team_object(team_id, **kwargs):
        org = team_org_map.get(team_id)
        if org is None:
            raise Exception(f"team {team_id} not found")
        team_obj = MagicMock()
        team_obj.team_id = team_id
        team_obj.organization_id = org
        return team_obj

    get_team_object_mock = AsyncMock(side_effect=_get_team_object)
    if monkeypatch is not None:
        # Patch the import inside the function under test (the fix imports
        # ``get_team_object`` locally, so we patch the source module).
        import litellm.proxy.auth.auth_checks as _ac
        monkeypatch.setattr(_ac, "get_team_object", get_team_object_mock)
        # Also patch user_api_key_cache import target.
        import litellm.proxy.proxy_server as _ps
        if not hasattr(_ps, "user_api_key_cache"):
            monkeypatch.setattr(_ps, "user_api_key_cache", MagicMock(), raising=False)

    return prisma, get_team_object_mock


@pytest.mark.asyncio
async def test_assign_key_org_allows_transitive_via_team_membership(monkeypatch):
    """LIT-3214 regression: an internal user who is a member of a team that
    belongs to ``org-1`` should be able to create a key for that team even if
    they have NO ``LiteLLM_OrganizationMembership`` row — the enterprise
    inheritance copies ``organization_id`` from the team and the callsite
    runs this check.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma, _ = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=[],
        team_ids=["team-A"],
        team_org_map={"team-A": "org-1"},
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    await _validate_caller_can_assign_key_org(
        user_api_key_dict=caller,
        organization_id="org-1",
        prisma_client=prisma,
    )


@pytest.mark.asyncio
async def test_assign_key_org_blocks_explicit_other_org_even_with_team_membership(monkeypatch):
    """Security guardrail: even though the caller is a team member of team-A
    in org-1, they may NOT explicitly target a different org-2 they have no
    relationship to.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma, _ = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=[],
        team_ids=["team-A"],
        team_org_map={"team-A": "org-1"},
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-2-other",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
    assert "org-2-other" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_assign_key_org_direct_membership_short_circuits_team_lookup(monkeypatch):
    """Direct org membership wins without hitting the team table."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma, get_team_mock = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=["org-1"],
        team_ids=["team-A"],
        team_org_map={"team-A": "org-other"},
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    await _validate_caller_can_assign_key_org(
        user_api_key_dict=caller,
        organization_id="org-1",
        prisma_client=prisma,
    )
    get_team_mock.assert_not_called()


@pytest.mark.asyncio
async def test_assign_key_org_blocks_caller_with_no_teams_and_no_memberships(monkeypatch):
    """Caller belongs to nothing — both branches miss => 403."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma, get_team_mock = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=[],
        team_ids=[],
        team_org_map={},
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
    get_team_mock.assert_not_called()


@pytest.mark.asyncio
async def test_assign_key_org_team_in_different_org_does_not_grant_access(monkeypatch):
    """User is in team-A (org-other). Should NOT be able to target org-1."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma, get_team_mock = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=[],
        team_ids=["team-A"],
        team_org_map={"team-A": "org-other"},
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
    # The cached lookup WAS used (once per team in the user's teams).
    get_team_mock.assert_called_once_with(
        team_id="team-A",
        prisma_client=prisma,
        user_api_key_cache=get_team_mock.call_args.kwargs["user_api_key_cache"],
    )


@pytest.mark.asyncio
async def test_assign_key_org_user_not_found():
    """If the user row vanished from the DB, deny — don't NoneType-error."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = MagicMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403



@pytest.mark.asyncio
async def test_assign_key_org_skips_stale_team_id_that_no_longer_exists(monkeypatch):
    """If a team_id in user_row.teams was deleted, get_team_object raises.
    The helper should skip the stale entry and continue, not 500.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    # team-stale is on the user row but the team table has no such row;
    # team-real points at org-1.
    prisma, get_team_mock = _make_prisma_with_user_teams_and_team_orgs(
        user_id="alice",
        org_memberships=[],
        team_ids=["team-stale", "team-real"],
        team_org_map={"team-real": "org-1"},  # team-stale missing on purpose
        monkeypatch=monkeypatch,
    )
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    # Should NOT raise: team-real grants access to org-1.
    await _validate_caller_can_assign_key_org(
        user_api_key_dict=caller,
        organization_id="org-1",
        prisma_client=prisma,
    )
    assert get_team_mock.call_count == 2

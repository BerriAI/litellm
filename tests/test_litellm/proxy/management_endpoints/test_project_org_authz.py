"""
Unit tests for the VERIA-55 fixes:

- Project update permission must be evaluated against the project's *current*
  team, not a team supplied in the request body.
- Key update may not assign a key to an organization the caller is not a
  member of.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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
# /project/update — object_permission upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_project_upserts_object_permission_as_dict():
    """jsonify_object serializes the nested object_permission dict into a JSON
    string; passing that string straight to the Prisma table raised
    FieldNotFoundError. The shared helper must receive it and upsert a dict."""
    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        UpdateProjectRequest,
    )
    from litellm.proxy.utils import PrismaClient
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        update_project,
    )

    prisma = MagicMock()
    prisma.jsonify_object = lambda data: PrismaClient.jsonify_object(prisma, data)
    prisma.db.litellm_projecttable.find_unique = AsyncMock(
        return_value=MagicMock(team_id=None, budget_id=None, object_permission_id=None)
    )
    prisma.db.litellm_projecttable.update = AsyncMock(return_value=MagicMock())
    perm_table = prisma.db.litellm_objectpermissiontable
    perm_table.find_unique = AsyncMock(return_value=None)
    perm_table.upsert = AsyncMock(return_value=MagicMock(object_permission_id="op-1"))

    admin = UserAPIKeyAuth(user_id="root", user_role=LitellmUserRoles.PROXY_ADMIN.value)
    data = UpdateProjectRequest(
        project_id="proj-1",
        object_permission=LiteLLM_ObjectPermissionBase(mcp_servers=["server-1"]),
    )
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.premium_user", True),
    ):
        await update_project(
            data=data, http_request=MagicMock(), user_api_key_dict=admin
        )

    perm_table.upsert.assert_awaited_once()
    upsert_data = perm_table.upsert.await_args.kwargs["data"]
    assert upsert_data["create"] == {"mcp_servers": ["server-1"]}
    project_update = prisma.db.litellm_projecttable.update.await_args.kwargs["data"]
    assert project_update["object_permission_id"] == "op-1"
    assert "object_permission" not in project_update

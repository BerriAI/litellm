"""Regression tests for LIT-2553.

/team/list must narrow data for internal_user / internal_user_viewer
callers so they only see their own membership, their own keys, and no
other-member or admin-only fields. proxy_admin and org_admin callers
keep full visibility.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamMembership,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    TeamListResponseObject,
    UserAPIKeyAuth,
)


def _build_response_object(
    caller_user_id: str = "alice",
    team_id: str = "team-1",
) -> TeamListResponseObject:
    alice_membership = LiteLLM_TeamMembership(
        user_id=caller_user_id,
        team_id=team_id,
        budget_id="b-alice",
        spend=12.5,
        total_spend=44.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )
    bob_membership = LiteLLM_TeamMembership(
        user_id="bob",
        team_id=team_id,
        budget_id="b-bob",
        spend=99.0,
        total_spend=200.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=500.0),
    )
    alice_key = {
        "token": "sk-alice-hash",
        "key_alias": "alice-key",
        "user_id": caller_user_id,
        "team_id": team_id,
        "spend": 1.0,
    }
    bob_key = {
        "token": "sk-bob-hash",
        "key_alias": "bob-key",
        "user_id": "bob",
        "team_id": team_id,
        "spend": 2.0,
    }
    return TeamListResponseObject(
        team_id=team_id,
        team_alias="my-team",
        organization_id="org-1",
        admins=[caller_user_id, "bob"],
        members=[caller_user_id, "bob"],
        members_with_roles=[
            Member(user_id=caller_user_id, role="user"),
            Member(user_id="bob", role="admin"),
        ],
        team_member_permissions=["/key/generate", "/key/update"],
        metadata={"secret": "internal", "team_default_budget_id": "b-default"},
        models=["gpt-4o"],
        blocked=False,
        max_budget=1000.0,
        soft_budget=800.0,
        budget_duration="30d",
        spend=400.0,
        tpm_limit=100,
        rpm_limit=10,
        router_settings={"timeout": 30},
        team_memberships=[alice_membership, bob_membership],
        keys=[alice_key, bob_key],
    )


def _build_team_row(team_id: str = "team-1", caller: str = "alice"):
    row = MagicMock()
    row.team_id = team_id
    row.organization_id = "org-1"
    row.members_with_roles = [
        {"user_id": caller, "role": "user"},
        {"user_id": "bob", "role": "admin"},
    ]
    row.model_dump = lambda: {
        "team_id": team_id,
        "team_alias": "my-team",
        "organization_id": "org-1",
        "admins": [caller, "bob"],
        "members": [caller, "bob"],
        "members_with_roles": [
            {"user_id": caller, "role": "user"},
            {"user_id": "bob", "role": "admin"},
        ],
        "team_member_permissions": ["/key/generate"],
        "metadata": {"secret": "value"},
        "models": ["gpt-4o"],
        "blocked": False,
        "max_budget": 1000.0,
        "spend": 400.0,
        "tpm_limit": 100,
        "rpm_limit": 10,
        "router_settings": {"timeout": 30},
    }
    return row


def _build_membership(user_id: str, team_id: str = "team-1"):
    row = MagicMock()
    row.team_id = team_id
    row.model_dump = lambda: {
        "user_id": user_id,
        "team_id": team_id,
        "budget_id": f"b-{user_id}",
        "spend": 12.5,
        "total_spend": 44.0,
        "litellm_budget_table": None,
    }
    return row


def _build_key(user_id: str, team_id: str = "team-1"):
    return {
        "token": f"sk-{user_id}-hash",
        "key_alias": f"{user_id}-key",
        "user_id": user_id,
        "team_id": team_id,
        "spend": 0.0,
    }


@pytest.fixture
def mock_request():
    from fastapi import Request

    return MagicMock(spec=Request)


def test_narrowing_helper_filters_to_caller_only():
    from litellm.proxy.management_endpoints.team_endpoints import (
        _narrow_team_list_response_for_internal_user,
    )

    resp = _build_response_object(caller_user_id="alice")
    out = _narrow_team_list_response_for_internal_user(resp, "alice")

    assert out.team_id == "team-1"
    assert out.team_alias == "my-team"
    assert out.organization_id == "org-1"
    assert out.models == ["gpt-4o"]
    assert out.max_budget == 1000.0
    assert out.blocked is False

    assert out.admins == ["alice"]
    assert out.members == ["alice"]
    assert len(out.members_with_roles) == 1
    assert out.members_with_roles[0].user_id == "alice"

    assert len(out.team_memberships) == 1
    assert out.team_memberships[0].user_id == "alice"

    assert len(out.keys) == 1
    assert out.keys[0]["user_id"] == "alice"

    assert out.team_member_permissions is None
    assert out.metadata is None
    assert out.router_settings is None


def test_narrowing_helper_with_missing_caller_user_id_strips_everything():
    from litellm.proxy.management_endpoints.team_endpoints import (
        _narrow_team_list_response_for_internal_user,
    )

    resp = _build_response_object(caller_user_id="alice")
    out = _narrow_team_list_response_for_internal_user(resp, None)

    assert out.admins == []
    assert out.members == []
    assert out.members_with_roles == []
    assert out.team_memberships == []
    assert out.keys == []
    assert out.metadata is None
    assert out.team_member_permissions is None


def test_narrowing_helper_drops_keys_owned_by_other_users():
    from litellm.proxy.management_endpoints.team_endpoints import (
        _narrow_team_list_response_for_internal_user,
    )

    resp = _build_response_object(caller_user_id="alice")
    out = _narrow_team_list_response_for_internal_user(resp, "alice")

    key_owners = [k.get("user_id") for k in out.keys]
    assert key_owners == ["alice"]
    assert "bob" not in key_owners


@pytest.mark.asyncio
async def test_list_team_narrows_response_for_internal_user(mock_request):
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
    )

    team_row = _build_team_row(caller="alice")
    memberships = [_build_membership("alice"), _build_membership("bob")]
    keys = [_build_key("alice"), _build_key("bob")]

    caller_user = LiteLLM_UserTable(
        user_id="alice",
        teams=["team-1"],
        organization_memberships=[],
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = MagicMock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
        mock_db.litellm_teammembership.find_many = AsyncMock(return_value=memberships)
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)

        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=caller_user,
        ):
            result = await list_team(
                http_request=mock_request,
                user_id="alice",
                organization_id=None,
                user_api_key_dict=user_api_key_dict,
            )

    assert len(result) == 1
    team = result[0]

    assert [m.user_id for m in team.members_with_roles] == ["alice"]
    assert team.admins == ["alice"]
    assert team.members == ["alice"]

    assert [tm.user_id for tm in team.team_memberships] == ["alice"]
    key_uids = [
        (k.get("user_id") if isinstance(k, dict) else getattr(k, "user_id", None))
        for k in team.keys
    ]
    assert key_uids == ["alice"]

    assert team.team_member_permissions is None
    assert team.metadata is None
    assert team.router_settings is None

    assert team.team_id == "team-1"
    assert team.team_alias == "my-team"
    assert team.organization_id == "org-1"
    assert team.models == ["gpt-4o"]
    assert team.max_budget == 1000.0


@pytest.mark.asyncio
async def test_list_team_does_not_narrow_for_proxy_admin(mock_request):
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-1",
    )

    team_row = _build_team_row(caller="alice")
    memberships = [_build_membership("alice"), _build_membership("bob")]
    keys = [_build_key("alice"), _build_key("bob")]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = MagicMock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
        mock_db.litellm_teammembership.find_many = AsyncMock(return_value=memberships)
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)

        result = await list_team(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            user_api_key_dict=user_api_key_dict,
        )

    assert len(result) == 1
    team = result[0]
    assert sorted(m.user_id for m in team.members_with_roles) == ["alice", "bob"]
    assert sorted(team.admins) == ["alice", "bob"]
    assert sorted(tm.user_id for tm in team.team_memberships) == ["alice", "bob"]
    key_uids = [
        (k.get("user_id") if isinstance(k, dict) else getattr(k, "user_id", None))
        for k in team.keys
    ]
    assert sorted(key_uids) == ["alice", "bob"]
    assert team.team_member_permissions == ["/key/generate"]
    assert team.metadata == {"secret": "value"}


@pytest.mark.asyncio
async def test_list_team_does_not_narrow_for_org_admin(mock_request):
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
    )

    team_row = _build_team_row(caller="alice")
    memberships = [_build_membership("alice"), _build_membership("bob")]
    keys = [_build_key("alice"), _build_key("bob")]

    org_admin_user = LiteLLM_UserTable(
        user_id="alice",
        teams=["team-1"],
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="alice",
                organization_id="org-1",
                user_role=LitellmUserRoles.ORG_ADMIN.value,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ],
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = MagicMock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
        mock_db.litellm_teammembership.find_many = AsyncMock(return_value=memberships)
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)

        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=org_admin_user,
        ):
            result = await list_team(
                http_request=mock_request,
                user_id="alice",
                organization_id=None,
                user_api_key_dict=user_api_key_dict,
            )

    assert len(result) == 1
    team = result[0]
    assert sorted(m.user_id for m in team.members_with_roles) == ["alice", "bob"]
    assert team.metadata == {"secret": "value"}
    assert team.team_member_permissions == ["/key/generate"]


@pytest.mark.asyncio
async def test_list_team_does_not_narrow_for_proxy_admin_view_only(mock_request):
    """LIT-2553: PROXY_ADMIN_VIEW_ONLY callers also keep full visibility
    (they pass `_user_has_admin_view`, same as PROXY_ADMIN). Regression
    guard from Greptile review on PR #29117."""
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        user_id="viewer-admin",
    )

    team_row = _build_team_row(caller="alice")
    memberships = [_build_membership("alice"), _build_membership("bob")]
    keys = [_build_key("alice"), _build_key("bob")]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = MagicMock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
        mock_db.litellm_teammembership.find_many = AsyncMock(return_value=memberships)
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)

        result = await list_team(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            user_api_key_dict=user_api_key_dict,
        )

    assert len(result) == 1
    team = result[0]
    assert sorted(m.user_id for m in team.members_with_roles) == ["alice", "bob"]
    assert sorted(team.admins) == ["alice", "bob"]
    assert sorted(tm.user_id for tm in team.team_memberships) == ["alice", "bob"]
    key_uids = [
        (k.get("user_id") if isinstance(k, dict) else getattr(k, "user_id", None))
        for k in team.keys
    ]
    assert sorted(key_uids) == ["alice", "bob"]
    assert team.metadata == {"secret": "value"}
    assert team.team_member_permissions == ["/key/generate"]

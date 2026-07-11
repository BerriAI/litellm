import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.management_endpoints.team_endpoints as team_endpoints
from litellm.proxy._types import (
    BulkTeamMemberUpdateRequest,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    TeamMemberBulkUpdateFields,
    TeamMemberUpdateRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    bulk_update_team_members,
    team_member_update,
)


@pytest.mark.asyncio
async def test_ateam_member_update_admin_requires_premium(monkeypatch):
    # Arrange: patch prisma_client and premium_user
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "premium_user", False)

    # Create a request body that tries to set role=admin
    data = TeamMemberUpdateRequest(
        team_id="team-1234",
        user_id="user-1",
        user_email=None,
        role="admin",
        max_budget_in_team=None,
    )
    scope = {"type": "http", "method": "POST", "path": "/team/member_update"}
    request = Request(scope)

    # We don't need a full auth object since premium check happens before auth is used
    auth = object()

    # Act & Assert: expect HTTPException 400 with the exact premium feature message
    with pytest.raises(HTTPException) as exc_info:
        await team_member_update(data, request, auth)

    assert exc_info.value.status_code == 400
    expected_msg = (
        "Assigning team admins is a premium feature. You must be a LiteLLM Enterprise user to use this feature. "
        "If you have a license please set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/#trial. "
        "Pricing: https://www.litellm.ai/#pricing"
    )
    assert exc_info.value.detail == expected_msg


@pytest.fixture
def happy_path_upsert(monkeypatch):
    """Stub out the DB and the budget upsert so a team_member_update call reaches
    _upsert_budget_and_membership, and hand back that mock to inspect the patch."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
        metadata={},
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    prisma_client.db.litellm_teamtable.update = AsyncMock()

    class _FakeTx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    prisma_client.db.tx = MagicMock(return_value=_FakeTx())

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "premium_user", False)
    monkeypatch.setattr(
        team_endpoints,
        "team_info",
        AsyncMock(
            return_value={
                "team_info": team_row,
                "team_memberships": [types.SimpleNamespace(user_id="user-1", budget_id="bud-1")],
            }
        ),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_upsert_budget_and_membership", upsert_mock)
    return upsert_mock


def _member_update_request(**overrides):
    data = TeamMemberUpdateRequest(team_id="team-1234", user_id="user-1", role="user", **overrides)
    request = Request({"type": "http", "method": "POST", "path": "/team/member_update"})
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")
    return data, request, auth


@pytest.mark.asyncio
async def test_team_member_update_sends_provided_fields_as_patch(happy_path_upsert):
    """Fields the request sets must reach _upsert_budget_and_membership as a
    budget patch, otherwise the member budget is never written/reset."""
    data, request, auth = _member_update_request(max_budget_in_team=10.0, budget_duration="30d")

    response = await team_member_update(data, request, auth)

    happy_path_upsert.assert_awaited_once()
    assert happy_path_upsert.await_args.kwargs["budget_patch"] == {
        "max_budget": 10.0,
        "budget_duration": "30d",
    }
    assert response.budget_duration == "30d"


@pytest.mark.asyncio
async def test_team_member_update_explicit_null_clears_field(happy_path_upsert):
    """An explicitly-null field must be forwarded as None so the column is
    cleared, rather than silently dropped."""
    data, request, auth = _member_update_request(budget_duration=None)

    await team_member_update(data, request, auth)

    assert happy_path_upsert.await_args.kwargs["budget_patch"] == {"budget_duration": None}


@pytest.mark.asyncio
async def test_team_member_update_omits_unset_fields_from_patch(happy_path_upsert):
    """A request that touches no budget fields must produce an empty patch so the
    member's existing budget is left untouched."""
    data, request, auth = _member_update_request()

    await team_member_update(data, request, auth)

    assert happy_path_upsert.await_args.kwargs["budget_patch"] == {}


@pytest.mark.parametrize(
    "bad_duration",
    [
        "not-a-duration",  # unparseable garbage
        "10x",  # unsupported unit
        "0d",  # zero-length window
        "999999999999999999999999d",  # overflows datetime math
    ],
)
@pytest.mark.asyncio
async def test_team_member_update_rejects_invalid_budget_duration(monkeypatch, bad_duration):
    """An invalid budget_duration must be rejected with a 400 before any DB
    write, so it can never be persisted and later break the budget reset job."""
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "premium_user", False)
    upsert_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_upsert_budget_and_membership", upsert_mock)

    data = TeamMemberUpdateRequest(
        team_id="team-1234",
        user_id="user-1",
        role="user",
        budget_duration=bad_duration,
    )
    request = Request({"type": "http", "method": "POST", "path": "/team/member_update"})
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")

    with pytest.raises(HTTPException) as exc_info:
        await team_member_update(data, request, auth)

    assert exc_info.value.status_code == 400
    assert "budget_duration" in str(exc_info.value.detail)
    upsert_mock.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_team_member_update_applies_patch_and_returns_member_failures(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(
        team_endpoints,
        "team_info",
        AsyncMock(return_value={"team_info": team_row, "team_memberships": []}),
    )
    update_mock = AsyncMock(
        side_effect=[
            team_endpoints.TeamMemberUpdateResponse(team_id="team-1234", user_id="user-1", tpm_limit=42),
            HTTPException(status_code=404, detail={"error": "User is not a team member"}),
        ]
    )
    monkeypatch.setattr(team_endpoints, "_apply_team_member_update", update_mock)

    response = await bulk_update_team_members(
        data=BulkTeamMemberUpdateRequest(
            team_id="team-1234",
            user_ids=["user-1", "user-2", "user-1"],
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=Request({"type": "http", "method": "POST", "path": "/team/member/bulk_update"}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin"),
    )

    assert response.total_requested == 2
    assert [member.user_id for member in response.successful_updates] == ["user-1"]
    assert response.failed_updates[0].user_id == "user-2"
    # a dict HTTPException detail must surface the nested error string, not a
    # python dict repr like "{'error': 'User is not a team member'}"
    assert response.failed_updates[0].failed_reason == "User is not a team member"
    assert update_mock.await_args_list[0].kwargs["data"].model_dump(exclude_unset=True) == {
        "team_id": "team-1234",
        "user_id": "user-1",
        "tpm_limit": 42,
    }


@pytest.mark.asyncio
async def test_bulk_team_member_update_returns_unexpected_member_failure(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(
        team_endpoints,
        "team_info",
        AsyncMock(return_value={"team_info": team_row, "team_memberships": []}),
    )
    monkeypatch.setattr(
        team_endpoints, "_apply_team_member_update", AsyncMock(side_effect=RuntimeError("database unavailable"))
    )

    response = await bulk_update_team_members(
        data=BulkTeamMemberUpdateRequest(
            team_id="team-1234",
            user_ids=["user-1"],
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=Request({"type": "http", "method": "POST", "path": "/team/member/bulk_update"}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin"),
    )

    assert response.successful_updates == []
    assert response.failed_updates == [
        team_endpoints.FailedTeamMemberUpdate(user_id="user-1", failed_reason="database unavailable")
    ]


@pytest.mark.asyncio
async def test_bulk_team_member_update_resolves_team_info_once(monkeypatch):
    """The whole batch must resolve team_info a single time and still upsert every
    member; resolving it per member re-scans the team, its keys, and all
    memberships on each iteration, which times out large teams."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
            Member(user_id="user-3", role="user"),
        ],
        metadata={},
    )
    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)

    class _FakeTx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    prisma_client.db.tx = MagicMock(return_value=_FakeTx())
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "premium_user", False)

    team_info_mock = AsyncMock(
        return_value={
            "team_info": team_row,
            "team_memberships": [
                types.SimpleNamespace(user_id="user-1", budget_id="bud-1"),
                types.SimpleNamespace(user_id="user-2", budget_id="bud-2"),
                types.SimpleNamespace(user_id="user-3", budget_id="bud-3"),
            ],
        }
    )
    monkeypatch.setattr(team_endpoints, "team_info", team_info_mock)
    upsert_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_upsert_budget_and_membership", upsert_mock)

    response = await bulk_update_team_members(
        data=BulkTeamMemberUpdateRequest(
            team_id="team-1234",
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=Request({"type": "http", "method": "POST", "path": "/team/member/bulk_update"}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin"),
    )

    assert team_info_mock.await_count == 1
    assert upsert_mock.await_count == 3
    assert [member.user_id for member in response.successful_updates] == ["user-1", "user-2", "user-3"]


def test_bulk_team_member_update_requires_exactly_one_member_selector():
    with pytest.raises(ValueError, match="either user_ids or all_members_in_team"):
        BulkTeamMemberUpdateRequest(
            team_id="team-1234",
            user_ids=["user-1"],
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        )

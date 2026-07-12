import json
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.management_endpoints.team_endpoints as team_endpoints
from litellm.proxy._types import (
    BulkTeamMemberUpdateRequest,
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
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
                "team_memberships": [LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="bud-1")],
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


class _RecordedWrites:
    def __init__(self):
        self.budget_update_many: list = []
        self.budget_creates: list = []
        self.team_updates: list = []


class _FakeBatcher:
    def __init__(self, writes: _RecordedWrites):
        self.litellm_budgettable = types.SimpleNamespace(
            update_many=lambda **kwargs: writes.budget_update_many.append(kwargs),
            create=lambda **kwargs: writes.budget_creates.append(kwargs),
        )
        self.litellm_teamtable = types.SimpleNamespace(update=lambda **kwargs: writes.team_updates.append(kwargs))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeBulkDb:
    """Typed fake for the exact prisma surface the bulk endpoint touches, so the
    tests assert the real queries issued (one update_many, batched creates, one
    team update) instead of monkeypatching endpoint internals."""

    def __init__(self, team_row, memberships, default_budget=None):
        self.writes = _RecordedWrites()
        self.membership_find_many_wheres: list = []
        self.budget_find_unique_wheres: list = []

        async def _team_find_unique(where):
            return team_row

        async def _membership_find_many(where):
            self.membership_find_many_wheres.append(where)
            return memberships

        async def _budget_find_unique(where):
            self.budget_find_unique_wheres.append(where)
            return default_budget

        self.litellm_teamtable = types.SimpleNamespace(find_unique=_team_find_unique)
        self.litellm_teammembership = types.SimpleNamespace(find_many=_membership_find_many)
        self.litellm_budgettable = types.SimpleNamespace(find_unique=_budget_find_unique)

    def batch_(self):
        return _FakeBatcher(self.writes)


def _bulk_setup(monkeypatch, team_row, memberships, default_budget=None):
    db = _FakeBulkDb(team_row, memberships, default_budget)
    monkeypatch.setattr(proxy_server, "prisma_client", types.SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "premium_user", False)
    return db


def _bulk_request():
    return Request({"type": "http", "method": "PATCH", "path": "/v2/team/team-1234/members"})


_ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")


@pytest.mark.asyncio
async def test_bulk_update_patches_private_budgets_with_one_update_many(monkeypatch):
    """Members that already own a private budget must be covered by a single
    update_many over their budget ids; a query per member re-introduces the
    n round trips this endpoint exists to avoid."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
            Member(user_id="user-3", role="user"),
        ],
    )
    db = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[
            LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="bud-1"),
            LiteLLM_TeamMembership(user_id="user-2", team_id="team-1234", budget_id="bud-2"),
        ],
    )

    response = await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "user-2", "user-1"],
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert db.writes.budget_update_many == [
        {"where": {"budget_id": {"in": ["bud-1", "bud-2"]}}, "data": {"updated_by": "admin", "tpm_limit": 42}}
    ]
    assert db.writes.budget_creates == []
    assert db.writes.team_updates == []
    assert db.membership_find_many_wheres == [{"team_id": "team-1234", "user_id": {"in": ["user-1", "user-2"]}}]
    assert response.total_requested == 2
    assert [member.user_id for member in response.successful_updates] == ["user-1", "user-2"]


@pytest.mark.asyncio
async def test_bulk_update_clones_default_budget_instead_of_patching_it(monkeypatch):
    """A member on the team's shared default budget must get their own cloned
    budget (default limits + patch); patching the shared row in place would
    change limits for every member outside the request. A member with no
    membership row gets a new budget wired to a created membership."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
        metadata={"team_member_budget_id": "default-bud"},
    )
    db = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="default-bud")],
        default_budget=LiteLLM_BudgetTable(budget_id="default-bud", max_budget=100.0),
    )

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "user-2"],
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert db.writes.budget_update_many == []
    assert db.budget_find_unique_wheres == [{"budget_id": "default-bud"}]
    assert db.writes.budget_creates == [
        {
            "data": {
                "created_by": "admin",
                "updated_by": "admin",
                "max_budget": 100.0,
                "tpm_limit": 42,
                "team_membership": {"connect": [{"user_id_team_id": {"user_id": "user-1", "team_id": "team-1234"}}]},
            }
        },
        {
            "data": {
                "created_by": "admin",
                "updated_by": "admin",
                "max_budget": 100.0,
                "tpm_limit": 42,
                "team_membership": {"create": [{"user_id": "user-2", "team_id": "team-1234"}]},
            }
        },
    ]


@pytest.mark.asyncio
async def test_bulk_update_role_writes_team_row_once(monkeypatch):
    """A role-only bulk update must rewrite members_with_roles in a single team
    update covering every targeted member, and must not touch budgets at all."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="admin"),
            Member(user_id="user-2", role="user", user_email="two@example.com"),
            Member(user_id="user-3", role="user"),
        ],
    )
    db = _bulk_setup(monkeypatch, team_row, memberships=[])

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "user-2"],
            update_fields=TeamMemberBulkUpdateFields(role="user"),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert db.membership_find_many_wheres == []
    assert db.writes.budget_update_many == []
    assert db.writes.budget_creates == []
    assert len(db.writes.team_updates) == 1
    update = db.writes.team_updates[0]
    assert update["where"] == {"team_id": "team-1234"}
    members = json.loads(update["data"]["members_with_roles"])
    assert [(member["user_id"], member["role"]) for member in members] == [
        ("user-1", "user"),
        ("user-2", "user"),
        ("user-3", "user"),
    ]
    assert members[1]["user_email"] == "two@example.com"


@pytest.mark.asyncio
async def test_bulk_update_reports_non_members_as_failed(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    db = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="bud-1")],
    )

    response = await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "ghost-user"],
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert response.total_requested == 2
    assert [member.user_id for member in response.successful_updates] == ["user-1"]
    assert response.failed_updates[0].user_id == "ghost-user"
    assert "not a member" in response.failed_updates[0].failed_reason
    assert db.writes.budget_update_many[0]["where"] == {"budget_id": {"in": ["bud-1"]}}
    assert db.writes.budget_creates == []


@pytest.mark.asyncio
async def test_bulk_update_explicit_null_duration_clears_reset_at(monkeypatch):
    """budget_duration: null must clear both the duration and budget_reset_at in
    the same update_many, otherwise stale reset timestamps keep firing."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    db = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="bud-1")],
    )

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1"],
            update_fields=TeamMemberBulkUpdateFields(budget_duration=None),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert db.writes.budget_update_many == [
        {
            "where": {"budget_id": {"in": ["bud-1"]}},
            "data": {"updated_by": "admin", "budget_duration": None, "budget_reset_at": None},
        }
    ]


@pytest.mark.asyncio
async def test_bulk_update_admin_role_requires_premium(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "premium_user", False)

    with pytest.raises(HTTPException) as exc_info:
        await bulk_update_team_members(
            team_id="team-1234",
            data=BulkTeamMemberUpdateRequest(
                user_ids=["user-1"],
                update_fields=TeamMemberBulkUpdateFields(role="admin"),
            ),
            http_request=_bulk_request(),
            user_api_key_dict=_ADMIN,
        )

    assert exc_info.value.status_code == 400
    assert "premium feature" in str(exc_info.value.detail)


def test_bulk_team_member_update_requires_exactly_one_member_selector():
    with pytest.raises(ValueError, match="either user_ids or all_members_in_team"):
        BulkTeamMemberUpdateRequest(
            user_ids=["user-1"],
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        )

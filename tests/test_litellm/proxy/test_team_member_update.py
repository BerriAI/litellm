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
from litellm.proxy.management_endpoints.team_member_budget_writes import (
    BudgetFieldSnapshot,
    MembershipBudgetSnapshot,
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


class _FakeTx:
    def __init__(self, team_row, recorder):
        self._team_row = team_row
        self._recorder = recorder
        self.litellm_teamtable = types.SimpleNamespace(update=self._team_update)
        self.litellm_budgettable = types.SimpleNamespace(
            create_many=self._create_many,
            update=self._budget_update,
        )
        self.litellm_teammembership = types.SimpleNamespace(
            update=self._membership_update,
            upsert=self._membership_upsert,
        )

    async def _team_update(self, **kwargs):
        self._recorder.team_updates.append(kwargs)
        return self._team_row

    async def _create_many(self, data):
        self._recorder.budget_creates.extend(data)
        return {"count": len(data)}

    async def _budget_update(self, **kwargs):
        self._recorder.budget_updates.append(kwargs)
        return types.SimpleNamespace(budget_id=kwargs["where"]["budget_id"])

    async def _membership_update(self, **kwargs):
        self._recorder.membership_updates.append(kwargs)
        return types.SimpleNamespace()

    async def _membership_upsert(self, **kwargs):
        self._recorder.membership_upserts.append(kwargs)
        return types.SimpleNamespace()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeBulkDb:
    def __init__(self, team_row):
        self.team_row = team_row
        self.team_updates: list = []
        self.budget_creates: list = []
        self.budget_updates: list = []
        self.membership_updates: list = []
        self.membership_upserts: list = []
        self.litellm_teamtable = types.SimpleNamespace(find_unique=AsyncMock(return_value=team_row))

    def tx(self):
        return _FakeTx(self.team_row, self)


def _bulk_setup(
    monkeypatch,
    team_row,
    *,
    membership_snapshots=(),
    budgets_by_id=None,
):
    db = _FakeBulkDb(team_row)
    cache = types.SimpleNamespace(async_delete_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", types.SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "premium_user", False)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    load_mock = AsyncMock(return_value=(tuple(membership_snapshots), budgets_by_id or {}))
    monkeypatch.setattr(team_endpoints, "_load_member_budget_snapshots", load_mock)
    refresh_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_refresh_cached_team", refresh_mock)
    return db, load_mock, refresh_mock, cache


def _bulk_request():
    return Request({"type": "http", "method": "PATCH", "path": "/v2/team/team-1234/members"})


_ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")


@pytest.mark.asyncio
async def test_bulk_update_plans_budget_writes_without_per_member_inline_prisma(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
            Member(user_id="user-3", role="user"),
        ],
    )
    db, load_mock, refresh_mock, cache = _bulk_setup(
        monkeypatch,
        team_row,
        membership_snapshots=(
            MembershipBudgetSnapshot(user_id="user-1", budget_id="bud-1"),
            MembershipBudgetSnapshot(user_id="user-2", budget_id="bud-2"),
        ),
        budgets_by_id={
            "bud-1": BudgetFieldSnapshot(budget_id="bud-1", fields={"tpm_limit": 1}),
            "bud-2": BudgetFieldSnapshot(budget_id="bud-2", fields={"tpm_limit": 2}),
        },
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

    load_mock.assert_awaited_once()
    assert load_mock.await_args.kwargs["user_ids"] == ["user-1", "user-2"]
    assert {update["where"]["budget_id"] for update in db.budget_updates} == {"bud-1", "bud-2"}
    assert db.team_updates == []
    refresh_mock.assert_not_awaited()
    assert cache.async_delete_cache.await_count == 4
    assert response.total_requested == 2
    assert [member.user_id for member in response.successful_updates] == ["user-1", "user-2"]


@pytest.mark.asyncio
async def test_bulk_update_clones_shared_default_instead_of_mutating_it(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
        metadata={"team_member_budget_id": "default-bud"},
    )
    ids = iter(["new-bud-1", "new-bud-2"])
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.team_member_budget_writes.uuid4",
        lambda: next(ids),
    )
    db, _load, _refresh, _cache = _bulk_setup(
        monkeypatch,
        team_row,
        membership_snapshots=(
            MembershipBudgetSnapshot(user_id="user-1", budget_id="default-bud"),
            MembershipBudgetSnapshot(user_id="user-2", budget_id=None),
        ),
        budgets_by_id={
            "default-bud": BudgetFieldSnapshot(budget_id="default-bud", fields={"max_budget": 100.0}),
        },
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

    assert db.budget_updates == []
    created_ids = {row["budget_id"] for row in db.budget_creates}
    assert created_ids == {"new-bud-1", "new-bud-2"}
    user1_create = next(row for row in db.budget_creates if row["budget_id"] == "new-bud-1")
    assert user1_create["max_budget"] == 100.0
    assert user1_create["tpm_limit"] == 42
    user2_create = next(row for row in db.budget_creates if row["budget_id"] == "new-bud-2")
    assert "max_budget" not in user2_create
    assert user2_create["tpm_limit"] == 42
    assert len(db.membership_upserts) == 2


@pytest.mark.asyncio
async def test_bulk_update_role_writes_team_row_once_and_refreshes_cache(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="admin"),
            Member(user_id="user-2", role="user", user_email="two@example.com"),
            Member(user_id="user-3", role="user"),
        ],
    )
    db, load_mock, refresh_mock, cache = _bulk_setup(monkeypatch, team_row)

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "user-2"],
            update_fields=TeamMemberBulkUpdateFields(role="user"),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    load_mock.assert_awaited_once()
    assert load_mock.await_args.kwargs["user_ids"] == []
    assert len(db.team_updates) == 1
    update = db.team_updates[0]
    assert update["where"] == {"team_id": "team-1234"}
    members = json.loads(update["data"]["members_with_roles"])
    assert [(member["user_id"], member["role"]) for member in members] == [
        ("user-1", "user"),
        ("user-2", "user"),
        ("user-3", "user"),
    ]
    assert members[1]["user_email"] == "two@example.com"
    refresh_mock.assert_awaited_once()
    assert refresh_mock.await_args.kwargs["team_row"] is team_row
    assert cache.async_delete_cache.await_count == 4


@pytest.mark.asyncio
async def test_bulk_update_all_members_in_team_dedups_members(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
    )
    db, load_mock, _refresh, _cache = _bulk_setup(
        monkeypatch,
        team_row,
        membership_snapshots=(
            MembershipBudgetSnapshot(user_id="user-1", budget_id="bud-1"),
            MembershipBudgetSnapshot(user_id="user-2", budget_id="bud-2"),
        ),
        budgets_by_id={
            "bud-1": BudgetFieldSnapshot(budget_id="bud-1", fields={"tpm_limit": 1}),
            "bud-2": BudgetFieldSnapshot(budget_id="bud-2", fields={"tpm_limit": 2}),
        },
    )

    response = await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert load_mock.await_args.kwargs["user_ids"] == ["user-1", "user-2"]
    assert {update["where"]["budget_id"] for update in db.budget_updates} == {"bud-1", "bud-2"}
    assert response.total_requested == 2


@pytest.mark.asyncio
async def test_bulk_update_reports_non_members_as_failed(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    db, load_mock, _refresh, _cache = _bulk_setup(
        monkeypatch,
        team_row,
        membership_snapshots=(MembershipBudgetSnapshot(user_id="user-1", budget_id="bud-1"),),
        budgets_by_id={"bud-1": BudgetFieldSnapshot(budget_id="bud-1", fields={"tpm_limit": 1})},
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
    assert load_mock.await_args.kwargs["user_ids"] == ["user-1"]
    assert {update["where"]["budget_id"] for update in db.budget_updates} == {"bud-1"}


@pytest.mark.asyncio
async def test_bulk_update_explicit_null_duration_disconnects_private_budget(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    db, _load, _refresh, _cache = _bulk_setup(
        monkeypatch,
        team_row,
        membership_snapshots=(MembershipBudgetSnapshot(user_id="user-1", budget_id="bud-1"),),
        budgets_by_id={
            "bud-1": BudgetFieldSnapshot(budget_id="bud-1", fields={"budget_duration": "1d"}),
        },
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

    assert db.budget_updates == []
    assert len(db.membership_updates) == 1
    assert db.membership_updates[0]["data"] == {"litellm_budget_table": {"disconnect": True}}


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

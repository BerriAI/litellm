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
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
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


class _FakeTx:
    def __init__(self, team_row, recorder):
        self._team_row = team_row
        self._recorder = recorder
        self.litellm_teamtable = types.SimpleNamespace(update=self._team_update)

    async def _team_update(self, **kwargs):
        self._recorder.team_updates.append(kwargs)
        return self._team_row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeBulkDb:
    def __init__(self, team_row, memberships):
        self.team_row = team_row
        self.memberships = memberships
        self.membership_find_many_wheres: list = []
        self.team_updates: list = []

        async def _team_find_unique(where):
            return team_row

        async def _membership_find_many(where):
            self.membership_find_many_wheres.append(where)
            return memberships

        self.litellm_teamtable = types.SimpleNamespace(find_unique=_team_find_unique)
        self.litellm_teammembership = types.SimpleNamespace(find_many=_membership_find_many)

    def tx(self):
        return _FakeTx(self.team_row, self)


def _bulk_setup(monkeypatch, team_row, memberships):
    db = _FakeBulkDb(team_row, memberships)
    user_api_key_cache = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "prisma_client", types.SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "premium_user", False)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    upsert_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_upsert_budget_and_membership", upsert_mock)
    refresh_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_refresh_cached_team", refresh_mock)
    return db, upsert_mock, refresh_mock, user_api_key_cache


def _bulk_request():
    return Request({"type": "http", "method": "PATCH", "path": "/v2/team/team-1234/members"})


_ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")


def _upsert_call_by_user(upsert_mock):
    return {call.kwargs["user_id"]: call.kwargs for call in upsert_mock.await_args_list}


@pytest.mark.asyncio
async def test_bulk_update_delegates_budget_upsert_per_valid_member(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
            Member(user_id="user-3", role="user"),
        ],
    )
    db, upsert_mock, refresh_mock, _cache = _bulk_setup(
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

    assert db.membership_find_many_wheres == [{"team_id": "team-1234", "user_id": {"in": ["user-1", "user-2"]}}]
    calls = _upsert_call_by_user(upsert_mock)
    assert set(calls) == {"user-1", "user-2"}
    assert calls["user-1"]["existing_budget_id"] == "bud-1"
    assert calls["user-2"]["existing_budget_id"] == "bud-2"
    for kwargs in calls.values():
        assert kwargs["budget_patch"] == {"tpm_limit": 42}
        assert kwargs["team_default_budget_id"] is None
    assert db.team_updates == []
    refresh_mock.assert_not_awaited()
    assert response.total_requested == 2
    assert [member.user_id for member in response.successful_updates] == ["user-1", "user-2"]


@pytest.mark.asyncio
async def test_bulk_update_invalidates_membership_cache_for_written_members_only(monkeypatch):
    """A budget/allowed_models write must evict the per-member
    `team_membership:{user_id}:{team_id}` cache and the legacy
    `{team_id}_{user_id}` cache for every member actually written, so the next
    request re-reads the new limits instead of enforcing stale ones. Members
    the patch didn't touch (no budget_patch, or filtered out as non-members)
    must be left alone."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
    )
    _db, _upsert_mock, _refresh_mock, cache = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[
            LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="bud-1"),
            LiteLLM_TeamMembership(user_id="user-2", team_id="team-1234", budget_id="bud-2"),
        ],
    )

    stale = "stale-cached-membership"
    for user_id in ("user-1", "user-2", "user-3"):
        await cache.async_set_cache(key="team_membership:{}:team-1234".format(user_id), value=stale)
        await cache.async_set_cache(key="team-1234_{}".format(user_id), value=stale)

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1", "user-2"],
            update_fields=TeamMemberBulkUpdateFields(allowed_models=["gpt-4"]),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    for user_id in ("user-1", "user-2"):
        assert await cache.async_get_cache(key="team_membership:{}:team-1234".format(user_id)) is None
        assert await cache.async_get_cache(key="team-1234_{}".format(user_id)) is None

    # user-3 was never part of this batch; its cache must be untouched.
    assert await cache.async_get_cache(key="team_membership:user-3:team-1234") == stale
    assert await cache.async_get_cache(key="team-1234_user-3") == stale


@pytest.mark.asyncio
async def test_bulk_update_role_only_change_does_not_touch_membership_cache(monkeypatch):
    """A role-only patch (no budget fields) never writes litellm_teammembership,
    so it must not evict the membership cache -- there's nothing stale to fix,
    and clearing it would just cause an avoidable DB read on the next request."""
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="admin")],
    )
    _db, _upsert_mock, _refresh_mock, cache = _bulk_setup(monkeypatch, team_row, memberships=[])

    stale = "stale-cached-membership"
    await cache.async_set_cache(key="team_membership:user-1:team-1234", value=stale)

    await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            user_ids=["user-1"],
            update_fields=TeamMemberBulkUpdateFields(role="user"),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert await cache.async_get_cache(key="team_membership:user-1:team-1234") == stale


@pytest.mark.asyncio
async def test_bulk_update_forwards_shared_default_only_for_members_still_on_it(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
        metadata={"team_member_budget_id": "default-bud"},
    )
    _db, upsert_mock, _refresh, _cache = _bulk_setup(
        monkeypatch,
        team_row,
        memberships=[LiteLLM_TeamMembership(user_id="user-1", team_id="team-1234", budget_id="default-bud")],
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

    calls = _upsert_call_by_user(upsert_mock)
    assert calls["user-1"]["existing_budget_id"] == "default-bud"
    assert calls["user-2"]["existing_budget_id"] is None
    for kwargs in calls.values():
        assert kwargs["team_default_budget_id"] == "default-bud"


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
    db, upsert_mock, refresh_mock, _cache = _bulk_setup(monkeypatch, team_row, memberships=[])

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
    upsert_mock.assert_not_awaited()
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
    db, upsert_mock, _refresh, _cache = _bulk_setup(monkeypatch, team_row, memberships=[])

    response = await bulk_update_team_members(
        team_id="team-1234",
        data=BulkTeamMemberUpdateRequest(
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        ),
        http_request=_bulk_request(),
        user_api_key_dict=_ADMIN,
    )

    assert db.membership_find_many_wheres == [{"team_id": "team-1234", "user_id": {"in": ["user-1", "user-2"]}}]
    assert list(_upsert_call_by_user(upsert_mock)) == ["user-1", "user-2"]
    assert response.total_requested == 2


@pytest.mark.asyncio
async def test_bulk_update_reports_non_members_as_failed(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    _db, upsert_mock, _refresh, _cache = _bulk_setup(
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
    assert list(_upsert_call_by_user(upsert_mock)) == ["user-1"]


@pytest.mark.asyncio
async def test_bulk_update_explicit_null_duration_forwarded_as_patch(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    _db, upsert_mock, _refresh, _cache = _bulk_setup(
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

    assert _upsert_call_by_user(upsert_mock)["user-1"]["budget_patch"] == {"budget_duration": None}


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


@pytest.mark.asyncio
async def test_bulk_update_without_db_connection_raises_500(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    with pytest.raises(HTTPException) as exc_info:
        await bulk_update_team_members(
            team_id="team-1234",
            data=BulkTeamMemberUpdateRequest(
                user_ids=["user-1"],
                update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
            ),
            http_request=_bulk_request(),
            user_api_key_dict=_ADMIN,
        )

    assert exc_info.value.status_code == 500
    assert "No db connected" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_update_missing_team_raises_400(monkeypatch):
    db, _upsert_mock, _refresh_mock, _cache = _bulk_setup(monkeypatch, team_row=None, memberships=[])

    async def _find_none(where):
        return None

    db.litellm_teamtable = types.SimpleNamespace(find_unique=_find_none)

    with pytest.raises(HTTPException) as exc_info:
        await bulk_update_team_members(
            team_id="team-1234",
            data=BulkTeamMemberUpdateRequest(
                user_ids=["user-1"],
                update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
            ),
            http_request=_bulk_request(),
            user_api_key_dict=_ADMIN,
        )

    assert exc_info.value.status_code == 400
    assert "does not exist" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_update_non_admin_caller_forbidden_raises_403(monkeypatch):
    team_row = LiteLLM_TeamTable(
        team_id="team-1234",
        members_with_roles=[Member(user_id="user-1", role="user")],
    )
    _bulk_setup(monkeypatch, team_row, memberships=[])
    monkeypatch.setattr(team_endpoints, "_is_user_org_admin_for_team", AsyncMock(return_value=False))

    outsider = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER.value, user_id="outsider")
    with pytest.raises(HTTPException) as exc_info:
        await bulk_update_team_members(
            team_id="team-1234",
            data=BulkTeamMemberUpdateRequest(
                user_ids=["user-1"],
                update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
            ),
            http_request=_bulk_request(),
            user_api_key_dict=outsider,
        )

    assert exc_info.value.status_code == 403
    assert "not proxy admin OR team admin" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_update_exceeding_max_batch_size_raises_400(monkeypatch):
    team_row = LiteLLM_TeamTable(team_id="team-1234", members_with_roles=[])
    _bulk_setup(monkeypatch, team_row, memberships=[])

    with pytest.raises(HTTPException) as exc_info:
        await bulk_update_team_members(
            team_id="team-1234",
            data=BulkTeamMemberUpdateRequest(
                user_ids=[f"user-{i}" for i in range(501)],
                update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
            ),
            http_request=_bulk_request(),
            user_api_key_dict=_ADMIN,
        )

    assert exc_info.value.status_code == 400
    assert "Maximum 500 team members" in str(exc_info.value.detail)


def test_bulk_team_member_update_requires_exactly_one_member_selector():
    with pytest.raises(ValueError, match="either user_ids or all_members_in_team"):
        BulkTeamMemberUpdateRequest(
            user_ids=["user-1"],
            all_members_in_team=True,
            update_fields=TeamMemberBulkUpdateFields(tpm_limit=42),
        )

import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.management_endpoints.team_endpoints as team_endpoints
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    TeamMemberUpdateRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_endpoints import team_member_update


def test_team_member_update_models_accept_spend():
    """The request/response models must carry an optional `spend` so an admin
    can reset a team member's accrued spend via /team/member_update."""
    from litellm.proxy._types import TeamMemberUpdateResponse

    req = TeamMemberUpdateRequest(team_id="t", user_id="u", spend=0.0)
    assert req.spend == 0.0

    resp = TeamMemberUpdateResponse(
        team_id="t", user_id="u", user_email=None, spend=0.0
    )
    assert resp.spend == 0.0


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
                "team_memberships": [
                    types.SimpleNamespace(user_id="user-1", budget_id="bud-1")
                ],
            }
        ),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(team_endpoints, "_upsert_budget_and_membership", upsert_mock)
    return upsert_mock


def _member_update_request(**overrides):
    data = TeamMemberUpdateRequest(
        team_id="team-1234", user_id="user-1", role="user", **overrides
    )
    request = Request({"type": "http", "method": "POST", "path": "/team/member_update"})
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")
    return data, request, auth


@pytest.mark.asyncio
async def test_team_member_update_sends_provided_fields_as_patch(happy_path_upsert):
    """Fields the request sets must reach _upsert_budget_and_membership as a
    budget patch, otherwise the member budget is never written/reset."""
    data, request, auth = _member_update_request(
        max_budget_in_team=10.0, budget_duration="30d"
    )

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

    assert happy_path_upsert.await_args.kwargs["budget_patch"] == {
        "budget_duration": None
    }


@pytest.mark.asyncio
async def test_team_member_update_omits_unset_fields_from_patch(happy_path_upsert):
    """A request that touches no budget fields must produce an empty patch so the
    member's existing budget is left untouched."""
    data, request, auth = _member_update_request()

    await team_member_update(data, request, auth)

    assert happy_path_upsert.await_args.kwargs["budget_patch"] == {}


@pytest.mark.asyncio
async def test_team_member_update_spend_writes_membership_and_invalidates(
    happy_path_upsert, monkeypatch
):
    """A `spend` value resets LiteLLM_TeamMembership.spend (upsert, since the
    row may not exist) and invalidates the cross-pod team-member counter."""
    prisma_client = proxy_server.prisma_client
    membership_upsert = AsyncMock()
    prisma_client.db.litellm_teammembership.upsert = membership_upsert
    invalidate = AsyncMock()
    monkeypatch.setattr(proxy_server, "_invalidate_spend_counter", invalidate)

    # Use a negative spend: this also implicitly validates that negative spend
    # is allowed, which is desirable. Admins may grant a team member extra
    # allowance for the current budget period only (a one-time spend grant)
    # without raising the recurring budget ceiling. Future changes should
    # continue allowing negative spend counters.
    data, request, auth = _member_update_request(spend=-25.0)

    response = await team_member_update(data, request, auth)

    membership_upsert.assert_awaited_once()
    kwargs = membership_upsert.await_args.kwargs
    assert kwargs["where"] == {
        "user_id_team_id": {"user_id": "user-1", "team_id": "team-1234"}
    }
    assert kwargs["data"]["update"] == {"spend": -25.0}
    assert kwargs["data"]["create"]["spend"] == -25.0
    invalidate.assert_awaited_once_with(
        counter_key="spend:team_member:user-1:team-1234"
    )
    assert response.spend == -25.0


@pytest.mark.asyncio
async def test_team_member_update_rejects_non_finite_spend(
    happy_path_upsert, monkeypatch
):
    """NaN/inf spend is rejected with 400 before any membership write or
    counter invalidation."""
    prisma_client = proxy_server.prisma_client
    membership_upsert = AsyncMock()
    prisma_client.db.litellm_teammembership.upsert = membership_upsert
    invalidate = AsyncMock()
    monkeypatch.setattr(proxy_server, "_invalidate_spend_counter", invalidate)

    data, request, auth = _member_update_request(spend=float("inf"))

    with pytest.raises(HTTPException) as exc_info:
        await team_member_update(data, request, auth)

    assert exc_info.value.status_code == 400
    membership_upsert.assert_not_called()
    invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_member_update_without_spend_skips_membership_write(
    happy_path_upsert, monkeypatch
):
    """A budget-only update must not touch the membership spend row or the
    counter."""
    prisma_client = proxy_server.prisma_client
    membership_upsert = AsyncMock()
    prisma_client.db.litellm_teammembership.upsert = membership_upsert
    invalidate = AsyncMock()
    monkeypatch.setattr(proxy_server, "_invalidate_spend_counter", invalidate)

    data, request, auth = _member_update_request(max_budget_in_team=10.0)

    await team_member_update(data, request, auth)

    membership_upsert.assert_not_called()
    invalidate.assert_not_awaited()


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
async def test_team_member_update_rejects_invalid_budget_duration(
    monkeypatch, bad_duration
):
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

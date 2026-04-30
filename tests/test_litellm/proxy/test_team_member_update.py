import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    TeamMemberUpdateRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_endpoints import team_member_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request():
    scope = {"type": "http", "method": "POST", "path": "/team/member_update"}
    return Request(scope)


def _admin_auth():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
    )


def _make_team_table(metadata=None):
    return LiteLLM_TeamTable(
        team_id="team-1",
        members_with_roles=[Member(user_id="user-A", role="user")],
        metadata=metadata or {},
    )


def _team_info_response(team_table, budget_id=None):
    membership = LiteLLM_TeamMembership(
        user_id="user-A",
        team_id="team-1",
        budget_id=budget_id,
        litellm_budget_table=None,
    )
    return {"team_info": team_table, "team_memberships": [membership]}


def _mock_prisma(team_table, team_budget_duration=None):
    """Build a minimal mock prisma client for team_member_update tests."""
    mock_existing_team = MagicMock()
    mock_existing_team.model_dump.return_value = team_table.model_dump()

    mock_budget_row = MagicMock()
    mock_budget_row.budget_duration = team_budget_duration

    # tx context manager
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)

    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)
    prisma.db.litellm_budgettable.find_unique = AsyncMock(return_value=mock_budget_row)
    prisma.db.tx = MagicMock(return_value=mock_tx)
    return prisma


# ---------------------------------------------------------------------------
# Tests for budget_duration resolution logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_team_member_update_explicit_budget_duration():
    """
    When budget_duration is explicitly provided in the request, it should be
    passed straight to _upsert_budget_and_membership, ignoring any team setting.
    """
    team_table = _make_team_table(metadata={"team_member_budget_id": "team-bud-1"})
    prisma = _mock_prisma(team_table, team_budget_duration="90d")

    data = TeamMemberUpdateRequest(
        team_id="team-1",
        user_id="user-A",
        max_budget_in_team=5.0,
        budget_duration="30d",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.premium_user", True), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints.team_info",
             new=AsyncMock(return_value=_team_info_response(team_table)),
         ), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
             new=AsyncMock(),
         ) as mock_upsert:
        await team_member_update(data, _make_request(), _admin_auth())

    mock_upsert.assert_awaited_once()
    assert mock_upsert.call_args.kwargs["budget_duration"] == "30d"
    assert mock_upsert.call_args.kwargs["budget_duration_explicit"] is True
    # Should NOT have fetched the team-level budget row
    prisma.db.litellm_budgettable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_member_update_explicit_null_budget_duration():
    """
    When budget_duration is explicitly set to null in the request, None should
    be used (lifetime cap), NOT the team's configured duration.
    """
    team_table = _make_team_table(metadata={"team_member_budget_id": "team-bud-1"})
    prisma = _mock_prisma(team_table, team_budget_duration="30d")

    # Simulate the client sending {"budget_duration": null} by including the
    # field in the model_fields_set while keeping the value None.
    data = TeamMemberUpdateRequest.model_validate(
        {"team_id": "team-1", "user_id": "user-A", "max_budget_in_team": 5.0, "budget_duration": None}
    )
    assert "budget_duration" in data.model_fields_set  # sanity check

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.premium_user", True), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints.team_info",
             new=AsyncMock(return_value=_team_info_response(team_table)),
         ), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
             new=AsyncMock(),
         ) as mock_upsert:
        await team_member_update(data, _make_request(), _admin_auth())

    mock_upsert.assert_awaited_once()
    assert mock_upsert.call_args.kwargs["budget_duration"] is None
    assert mock_upsert.call_args.kwargs["budget_duration_explicit"] is True
    # Should NOT have fetched the team-level budget row
    prisma.db.litellm_budgettable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_member_update_inherits_team_budget_duration():
    """
    When budget_duration is omitted from the request and the team has a
    team_member_budget_id, the team budget's duration should be inherited.
    """
    team_table = _make_team_table(metadata={"team_member_budget_id": "team-bud-1"})
    prisma = _mock_prisma(team_table, team_budget_duration="30d")

    # budget_duration NOT included in request at all
    data = TeamMemberUpdateRequest(
        team_id="team-1",
        user_id="user-A",
        max_budget_in_team=5.0,
    )
    assert "budget_duration" not in data.model_fields_set  # sanity check

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.premium_user", True), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints.team_info",
             new=AsyncMock(return_value=_team_info_response(team_table)),
         ), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
             new=AsyncMock(),
         ) as mock_upsert:
        await team_member_update(data, _make_request(), _admin_auth())

    mock_upsert.assert_awaited_once()
    assert mock_upsert.call_args.kwargs["budget_duration"] == "30d"
    assert mock_upsert.call_args.kwargs["budget_duration_explicit"] is False
    prisma.db.litellm_budgettable.find_unique.assert_awaited_once_with(
        where={"budget_id": "team-bud-1"}
    )


@pytest.mark.asyncio
async def test_team_member_update_no_team_budget_duration_defaults_to_none():
    """
    When budget_duration is omitted and the team has no team_member_budget_id,
    budget_duration should default to None.
    """
    team_table = _make_team_table(metadata={})  # no team_member_budget_id
    prisma = _mock_prisma(team_table)

    data = TeamMemberUpdateRequest(
        team_id="team-1",
        user_id="user-A",
        max_budget_in_team=5.0,
    )

    with patch("litellm.proxy.proxy_server.prisma_client", prisma), \
         patch("litellm.proxy.proxy_server.premium_user", True), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints.team_info",
             new=AsyncMock(return_value=_team_info_response(team_table)),
         ), \
         patch(
             "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
             new=AsyncMock(),
         ) as mock_upsert:
        await team_member_update(data, _make_request(), _admin_auth())

    mock_upsert.assert_awaited_once()
    assert mock_upsert.call_args.kwargs["budget_duration"] is None
    assert mock_upsert.call_args.kwargs["budget_duration_explicit"] is False
    prisma.db.litellm_budgettable.find_unique.assert_not_awaited()


# ---------------------------------------------------------------------------
# Role / premium-user guard tests
# ---------------------------------------------------------------------------

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

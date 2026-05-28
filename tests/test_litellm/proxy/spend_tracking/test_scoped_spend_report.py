"""Tests for caller-scoped /spend/report endpoints (LIT-2401)."""

import datetime
import os
import sys
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking import spend_management_endpoints as sme


def _admin():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-1",
        api_key="sk-admin-hashed",
        team_id="team-admin",
        org_id="org-admin",
    )


def _internal_user():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user-7",
        api_key="sk-user-hashed",
        team_id="team-blue",
        org_id="org-acme",
    )


def _org_admin():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.ORG_ADMIN,
        user_id="orgadmin-1",
        api_key="sk-orgadmin-hashed",
        team_id="team-red",
        org_id="org-acme",
    )


def test_resolve_api_key_defaults_to_caller():
    auth = _internal_user()
    assert (
        sme._resolve_api_key_scope(user_api_key_dict=auth, api_key=None) == auth.api_key
    )


def test_resolve_api_key_non_admin_override_forbidden():
    auth = _internal_user()
    with pytest.raises(HTTPException) as ei:
        sme._resolve_api_key_scope(
            user_api_key_dict=auth, api_key="someone-elses-hashed-token"
        )
    assert ei.value.status_code == 403


def test_resolve_api_key_admin_can_override():
    auth = _admin()
    assert (
        sme._resolve_api_key_scope(
            user_api_key_dict=auth, api_key="another-hashed-token"
        )
        == "another-hashed-token"
    )


def test_resolve_api_key_sk_prefix_gets_hashed():
    auth = _admin()
    out = sme._resolve_api_key_scope(user_api_key_dict=auth, api_key="sk-12345")
    assert not out.startswith("sk-")
    assert len(out) == 64


def test_resolve_api_key_no_caller_key_raises_400():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="u")
    with pytest.raises(HTTPException) as ei:
        sme._resolve_api_key_scope(user_api_key_dict=auth, api_key=None)
    assert ei.value.status_code == 400


def test_resolve_user_defaults_to_caller():
    auth = _internal_user()
    assert (
        sme._resolve_user_scope(user_api_key_dict=auth, internal_user_id=None)
        == auth.user_id
    )


def test_resolve_user_non_admin_override_forbidden():
    auth = _internal_user()
    with pytest.raises(HTTPException) as ei:
        sme._resolve_user_scope(user_api_key_dict=auth, internal_user_id="someone-else")
    assert ei.value.status_code == 403


def test_resolve_user_admin_can_override():
    auth = _admin()
    assert (
        sme._resolve_user_scope(user_api_key_dict=auth, internal_user_id="other-user")
        == "other-user"
    )


def test_resolve_user_no_user_id_raises_400():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-x")
    with pytest.raises(HTTPException) as ei:
        sme._resolve_user_scope(user_api_key_dict=auth, internal_user_id=None)
    assert ei.value.status_code == 400


def test_resolve_team_defaults_to_caller():
    auth = _internal_user()
    assert sme._resolve_team_scope(user_api_key_dict=auth, team_id=None) == auth.team_id


def test_resolve_team_non_admin_override_forbidden():
    auth = _internal_user()
    with pytest.raises(HTTPException) as ei:
        sme._resolve_team_scope(user_api_key_dict=auth, team_id="other-team")
    assert ei.value.status_code == 403


def test_resolve_team_admin_can_override():
    auth = _admin()
    assert (
        sme._resolve_team_scope(user_api_key_dict=auth, team_id="other-team")
        == "other-team"
    )


def test_resolve_team_no_team_id_raises_400():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="u")
    with pytest.raises(HTTPException) as ei:
        sme._resolve_team_scope(user_api_key_dict=auth, team_id=None)
    assert ei.value.status_code == 400


def _mock_prisma_with_teams(team_ids):
    pc = MagicMock()
    team_rows = [MagicMock(team_id=t) for t in team_ids]
    pc.db.litellm_teamtable.find_many = AsyncMock(return_value=team_rows)
    return pc


@pytest.mark.asyncio
async def test_resolve_org_internal_user_denied():
    auth = _internal_user()
    pc = _mock_prisma_with_teams(["team-blue"])
    with pytest.raises(HTTPException) as ei:
        await sme._resolve_org_scope(
            user_api_key_dict=auth, organization_id=None, prisma_client=pc
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_org_admin_defaults_to_own_org():
    auth = _org_admin()
    pc = _mock_prisma_with_teams(["team-red", "team-blue"])
    out = await sme._resolve_org_scope(
        user_api_key_dict=auth, organization_id=None, prisma_client=pc
    )
    assert out == ["team-red", "team-blue"]
    pc.db.litellm_teamtable.find_many.assert_awaited_once_with(
        where={"organization_id": "org-acme"}
    )


@pytest.mark.asyncio
async def test_resolve_org_admin_can_override():
    auth = _admin()
    pc = _mock_prisma_with_teams(["team-x"])
    out = await sme._resolve_org_scope(
        user_api_key_dict=auth, organization_id="other-org", prisma_client=pc
    )
    assert out == ["team-x"]
    pc.db.litellm_teamtable.find_many.assert_awaited_once_with(
        where={"organization_id": "other-org"}
    )


@pytest.mark.asyncio
async def test_resolve_org_org_admin_cannot_override_to_other_org():
    auth = _org_admin()
    pc = _mock_prisma_with_teams([])
    with pytest.raises(HTTPException) as ei:
        await sme._resolve_org_scope(
            user_api_key_dict=auth,
            organization_id="not-my-org",
            prisma_client=pc,
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_no_org_raises_400():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.ORG_ADMIN, user_id="oa")
    pc = _mock_prisma_with_teams([])
    with pytest.raises(HTTPException) as ei:
        await sme._resolve_org_scope(
            user_api_key_dict=auth, organization_id=None, prisma_client=pc
        )
    assert ei.value.status_code == 400


def test_build_spend_by_model_sql_inlines_column_and_index():
    s = sme._build_spend_by_model_sql(filter_column="team_id", filter_param_index=3)
    assert "sl.team_id = $3" in s
    assert "sl.api_key" in s
    assert "model_details" in s


def _mock_prisma_with_response(rows):
    pc = MagicMock()
    pc.db.query_raw = AsyncMock(return_value=rows)
    return pc


@pytest.mark.asyncio
async def test_key_spend_report_calls_query_with_resolved_hash(monkeypatch):
    pc = _mock_prisma_with_response([{"api_key": "h", "total_cost": 1.0}])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    auth = _internal_user()
    out = await sme.get_key_spend_report(
        start_date="2026-05-01",
        end_date="2026-05-28",
        api_key=None,
        user_api_key_dict=auth,
    )
    assert out == [{"api_key": "h", "total_cost": 1.0}]
    args, _kw = pc.db.query_raw.await_args
    sql, _sd, _ed, k = args[0], args[1], args[2], args[3]
    assert "sl.api_key = $3" in sql
    assert k == auth.api_key
    assert isinstance(_sd, datetime.datetime) and _sd.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_user_spend_report_non_admin_cannot_override(monkeypatch):
    pc = _mock_prisma_with_response([])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    with pytest.raises(HTTPException) as ei:
        await sme.get_user_spend_report(
            start_date="2026-05-01",
            end_date="2026-05-28",
            internal_user_id="someone-else",
            user_api_key_dict=_internal_user(),
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_team_spend_report_uses_team_id_filter(monkeypatch):
    pc = _mock_prisma_with_response([{"api_key": "h"}])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    await sme.get_team_spend_report(
        start_date="2026-05-01",
        end_date="2026-05-28",
        team_id=None,
        user_api_key_dict=_internal_user(),
    )
    args, _kw = pc.db.query_raw.await_args
    sql, _sd, _ed, t = args[0], args[1], args[2], args[3]
    assert "sl.team_id = $3" in sql
    assert t == "team-blue"


@pytest.mark.asyncio
async def test_org_spend_report_org_admin_runs_against_own_org_teams(monkeypatch):
    pc = MagicMock()
    team_rows = [MagicMock(team_id="team-red"), MagicMock(team_id="team-blue")]
    pc.db.litellm_teamtable.find_many = AsyncMock(return_value=team_rows)
    pc.db.query_raw = AsyncMock(return_value=[{"api_key": "h"}])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    await sme.get_org_spend_report(
        start_date="2026-05-01",
        end_date="2026-05-28",
        organization_id=None,
        user_api_key_dict=_org_admin(),
    )
    args, _kw = pc.db.query_raw.await_args
    sql, _sd, _ed, teams_param = args[0], args[1], args[2], args[3]
    assert "team_id = ANY($3::text[])" in sql
    assert teams_param == ["team-red", "team-blue"]


@pytest.mark.asyncio
async def test_org_spend_report_no_teams_short_circuits(monkeypatch):
    pc = MagicMock()
    pc.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    pc.db.query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    out = await sme.get_org_spend_report(
        start_date="2026-05-01",
        end_date="2026-05-28",
        organization_id=None,
        user_api_key_dict=_org_admin(),
    )
    assert out == []
    pc.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_key_spend_report_not_premium_returns_403(monkeypatch):
    pc = _mock_prisma_with_response([])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", False, raising=False)
    with pytest.raises(HTTPException) as ei:
        await sme.get_key_spend_report(
            start_date="2026-05-01",
            end_date="2026-05-28",
            api_key=None,
            user_api_key_dict=_admin(),
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_key_spend_report_missing_dates_400(monkeypatch):
    pc = _mock_prisma_with_response([])
    monkeypatch.setattr(ps, "prisma_client", pc, raising=False)
    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    with pytest.raises(HTTPException) as ei:
        await sme.get_key_spend_report(
            start_date=None,
            end_date=None,
            api_key=None,
            user_api_key_dict=_admin(),
        )
    assert ei.value.status_code == 400

"""LIT-3301 regression tests: team-scoped virtual keys can only see their own
team's spend logs even when the underlying user is a Proxy Admin."""

import datetime
import sys
from datetime import timezone

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy import proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy.spend_tracking import spend_management_endpoints as sme

sys.path.insert(0, "tests/test_litellm/proxy/spend_tracking")
from test_spend_management_endpoints import (  # noqa: E402
    _default_date_range,
    make_ui_spend_logs_mock_prisma,
)


@pytest.fixture
def three_team_logs():
    now = datetime.datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": "l1",
            "request_id": "req1",
            "api_key": "sk-team1",
            "user": "admin_user",
            "team_id": "team-1",
            "spend": 0.05,
            "startTime": now,
            "model": "gpt-4o",
        },
        {
            "id": "l2",
            "request_id": "req2",
            "api_key": "sk-team2",
            "user": "other",
            "team_id": "team-2",
            "spend": 0.10,
            "startTime": now,
            "model": "gpt-4o",
        },
        {
            "id": "l3",
            "request_id": "req3",
            "api_key": "sk-team3",
            "user": "third",
            "team_id": "team-3",
            "spend": 0.20,
            "startTime": now,
            "model": "gpt-4o",
        },
    ]


def _filter_by_team(rows):
    def fn(where):
        out = list(rows)
        if "team_id" in where and isinstance(where["team_id"], str):
            out = [r for r in out if r["team_id"] == where["team_id"]]
        return out

    return fn


def test_get_calling_key_team_id_returns_team_id_when_present():
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user", team_id="team-1"
    )
    assert sme._get_calling_key_team_id(auth) == "team-1"


def test_get_calling_key_team_id_returns_none_for_non_team_key():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    assert sme._get_calling_key_team_id(auth) is None


def test_get_calling_key_team_id_strips_whitespace():
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", team_id="  team-1  "
    )
    assert sme._get_calling_key_team_id(auth) == "team-1"


def test_get_calling_key_team_id_blank_string_is_none():
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", team_id="   "
    )
    assert sme._get_calling_key_team_id(auth) is None


def test_get_calling_key_team_id_swallows_exception():
    class Boom:
        @property
        def team_id(self):
            raise RuntimeError("boom")

    assert sme._get_calling_key_team_id(Boom()) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_admin_team_key_is_restricted_to_own_team_logs(
    three_team_logs, monkeypatch
):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(
            three_team_logs, _filter_by_team(three_team_logs)
        ),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        team_id="team-1",
        api_key="sk-team1",
    )
    try:
        client = TestClient(app)
        start, end = _default_date_range()
        r = client.get(
            "/spend/logs/ui",
            params={"start_date": start, "end_date": end},
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        team_ids = sorted({row["team_id"] for row in data["data"]})
        assert team_ids == ["team-1"]
        assert data["total"] == 1
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_admin_team_key_team_id_filter_must_match_calling_key(
    three_team_logs, monkeypatch
):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(
            three_team_logs, _filter_by_team(three_team_logs)
        ),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        team_id="team-1",
        api_key="sk-team1",
    )
    try:
        client = TestClient(app)
        start, end = _default_date_range()
        r = client.get(
            "/spend/logs/ui",
            params={"team_id": "team-2", "start_date": start, "end_date": end},
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 403, r.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_admin_team_key_redundant_team_id_filter_ok(three_team_logs, monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(
            three_team_logs, _filter_by_team(three_team_logs)
        ),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        team_id="team-1",
        api_key="sk-team1",
    )
    try:
        client = TestClient(app)
        start, end = _default_date_range()
        r = client.get(
            "/spend/logs/ui",
            params={"team_id": "team-1", "start_date": start, "end_date": end},
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        team_ids = sorted({row["team_id"] for row in data["data"]})
        assert team_ids == ["team-1"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_admin_key_without_team_id_still_sees_everything(
    three_team_logs, monkeypatch
):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(
            three_team_logs, _filter_by_team(three_team_logs)
        ),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="sk-admin",
    )
    try:
        client = TestClient(app)
        start, end = _default_date_range()
        r = client.get(
            "/spend/logs/ui",
            params={"start_date": start, "end_date": end},
            headers={"Authorization": "Bearer sk-admin"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        team_ids = sorted({row["team_id"] for row in data["data"]})
        assert team_ids == ["team-1", "team-2", "team-3"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_internal_user_team_key_also_restricted(three_team_logs, monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(
            three_team_logs, _filter_by_team(three_team_logs)
        ),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="regular_user",
        team_id="team-1",
        api_key="sk-team1",
    )
    try:
        client = TestClient(app)
        start, end = _default_date_range()
        r = client.get(
            "/spend/logs/ui",
            params={"team_id": "team-2", "start_date": start, "end_date": end},
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 403, r.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def _make_detail_mock_prisma(rows_by_id):
    class Row:
        def __init__(self, d):
            self.team_id = d.get("team_id")
            self.user = d.get("user")
            self.request_id = d["request_id"]

    class FindUnique:
        async def find_unique(self, where, include=None):
            d = rows_by_id.get(where["request_id"])
            return Row(d) if d else None

    class DB:
        def __init__(self):
            self.litellm_spendlogs = FindUnique()

        async def query_raw(self, sql, *params):
            return []

    class Prisma:
        def __init__(self):
            self.db = DB()

    return Prisma()


@pytest.mark.asyncio
async def test_detail_endpoint_admin_team_key_blocked_on_other_teams_row(monkeypatch):
    rows = {
        "req-team1": {"request_id": "req-team1", "team_id": "team-1", "user": "u"},
        "req-team2": {"request_id": "req-team2", "team_id": "team-2", "user": "u"},
    }
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        _make_detail_mock_prisma(rows),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin",
        team_id="team-1",
    )
    try:
        client = TestClient(app)
        r = client.get(
            "/spend/logs/ui/req-team2",
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 403, r.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_detail_endpoint_admin_team_key_allowed_on_own_teams_row(monkeypatch):
    rows = {"req-team1": {"request_id": "req-team1", "team_id": "team-1", "user": "u"}}
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        _make_detail_mock_prisma(rows),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin",
        team_id="team-1",
    )
    try:
        client = TestClient(app)
        r = client.get(
            "/spend/logs/ui/req-team1",
            headers={"Authorization": "Bearer sk-team1"},
        )
        assert r.status_code == 200, r.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_assert_calling_key_team_can_view_request_id_blocks_rowless_teams():
    rows = {"req-orphan": {"request_id": "req-orphan", "team_id": None, "user": "u"}}
    prisma = _make_detail_mock_prisma(rows)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await sme._assert_calling_key_team_can_view_request_id(
            prisma_client=prisma,
            calling_key_team_id="team-1",
            request_id="req-orphan",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_assert_calling_key_team_can_view_request_id_missing_row_is_noop():
    prisma = _make_detail_mock_prisma({})
    await sme._assert_calling_key_team_can_view_request_id(
        prisma_client=prisma,
        calling_key_team_id="team-1",
        request_id="no-such-id",
    )

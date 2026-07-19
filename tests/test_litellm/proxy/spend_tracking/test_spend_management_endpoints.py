import asyncio
import collections
import datetime
import json
import os
import re
import sys
from datetime import timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import litellm.proxy.proxy_server as ps


def _default_date_range():
    """Return (start_date, end_date) for the common 7-day range used in UI spend tests."""
    now = datetime.datetime.now(timezone.utc)
    return (
        (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _filter_logs_by_date_range(logs, where):
    """Filter logs by startTime gte/lte from where conditions."""
    if "startTime" not in where:
        return logs
    date_filters = where["startTime"]
    filtered = []
    for log in logs:
        log_date = datetime.datetime.fromisoformat(
            log["startTime"].replace("Z", "+00:00")
        )
        if "gte" in date_filters:
            fd = date_filters["gte"]
            filter_date = (
                datetime.datetime.fromisoformat(fd.replace("Z", "+00:00"))
                if "T" in fd
                else datetime.datetime.strptime(fd, "%Y-%m-%d %H:%M:%S")
            )
            if log_date < filter_date:
                continue
        if "lte" in date_filters:
            fd = date_filters["lte"]
            filter_date = (
                datetime.datetime.fromisoformat(fd.replace("Z", "+00:00"))
                if "T" in fd
                else datetime.datetime.strptime(fd, "%Y-%m-%d %H:%M:%S")
            )
            if log_date > filter_date:
                continue
        filtered.append(log)
    return filtered


def _reconstruct_ui_where_from_sql(sql_query, params):
    """
    Rebuild the Prisma-style ``where`` dict the filter_fns below expect from the
    raw SQL + params the endpoint emits.

    ``ui_view_spend_logs`` computes the total with a bounded
    ``SELECT COUNT(*) FROM (SELECT 1 ... LIMIT $cap+1)`` query and fetches the
    page with a separate ``ORDER BY ... LIMIT/OFFSET`` query. Both carry the
    same WHERE clause, so the terminator can be ``ORDER BY`` (page query) or
    ``LIMIT`` (bounded count query).
    """
    where: dict = {}
    clause = re.search(r"WHERE (.*?)\s+(?:ORDER BY|LIMIT)", sql_query, re.DOTALL)
    if clause is None:
        return where

    def _iso(value):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    eq_cols = {
        "team_id": "team_id",
        '"user"': "user",
        "api_key": "api_key",
        "request_id": "request_id",
        "model": "model",
        "model_id": "model_id",
        "model_group": "model_group",
        "end_user": "end_user",
    }
    date_bounds: dict = {}
    metadata_conds: list = []
    for cond in (c.strip() for c in clause.group(1).split(" AND ")):
        gte = re.search(r'"startTime" >= \(\$(\d+)', cond)
        lte = re.search(r'"startTime" <= \(\$(\d+)', cond)
        alias = re.search(r"user_api_key_alias' LIKE \$(\d+)", cond)
        code = re.search(r"error_code' = \$(\d+)", cond)
        msg = re.search(r"error_message' LIKE \$(\d+)", cond)
        sess = re.fullmatch(r"session_id LIKE \$(\d+)", cond)
        status = re.fullmatch(r"status = \$(\d+)", cond)
        if gte:
            date_bounds["gte"] = _iso(params[int(gte.group(1)) - 1])
        elif lte:
            date_bounds["lte"] = _iso(params[int(lte.group(1)) - 1])
        elif "OR team_id = ANY" in cond:
            where["OR"] = where.get("OR", []) + [{"multi_team": True}]
        elif "status = 'success'" in cond:
            where["OR"] = where.get("OR", []) + [{"status": "success"}]
        elif sess:
            where["session_id"] = {"contains": str(params[int(sess.group(1)) - 1]).strip("%")}
        elif status:
            where["status"] = {"equals": params[int(status.group(1)) - 1]}
        elif alias:
            metadata_conds.append(
                {
                    "path": ["user_api_key_alias"],
                    "string_contains": str(params[int(alias.group(1)) - 1]).strip("%"),
                }
            )
        elif code:
            metadata_conds.append(
                {
                    "path": ["error_information", "error_code"],
                    "equals": params[int(code.group(1)) - 1],
                }
            )
        elif msg:
            metadata_conds.append(
                {
                    "path": ["error_information", "error_message"],
                    "string_contains": str(params[int(msg.group(1)) - 1]).strip("%"),
                }
            )
        else:
            for sql_col, key in eq_cols.items():
                eq = re.fullmatch(rf"{re.escape(sql_col)} = \$(\d+)", cond)
                if eq:
                    where[key] = params[int(eq.group(1)) - 1]
                    break

    if date_bounds:
        where["startTime"] = date_bounds
    if len(metadata_conds) == 1:
        where["metadata"] = metadata_conds[0]
    elif len(metadata_conds) > 1:
        where["AND"] = [{"metadata": cond} for cond in metadata_conds]
    return where


def make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_fn, team_lookup_fn=None):
    """
    Create a MockPrismaClient for /spend/logs/ui endpoint tests.

    Args:
        mock_spend_logs: List of mock spend log dicts.
        filter_fn: Callable[[dict], list] - receives the reconstructed
                   where_conditions, returns the filtered list of logs.
        team_lookup_fn: Optional async callable for team RBAC (find_unique).
                        If provided, adds litellm_teamtable to db.
    """

    class MockDB:
        async def count(self, *args, **kwargs):
            return len(filter_fn(kwargs.get("where", {})))

        async def group_by(self, by, where, count):
            col = by[0]
            allowed = where.get(col, {}).get("in")
            tallied = collections.Counter(
                log[col]
                for log in mock_spend_logs
                if log.get(col) is not None and (allowed is None or log[col] in allowed)
            )
            return [{col: value, "_count": {col: n}} for value, n in tallied.items()]

        async def query_raw(self, sql_query, *params):
            if "mcp_tool_call_count" in sql_query:
                return []
            filtered = filter_fn(_reconstruct_ui_where_from_sql(sql_query, params))
            total = len(filtered)
            if "COUNT(*)" in sql_query:
                cap_plus_one = params[-1]
                return [{"total_count": min(total, cap_plus_one)}]
            page_size = params[-2] if len(params) >= 2 else 50
            skip = params[-1] if len(params) >= 1 else 0
            return [row for row in filtered[skip : skip + page_size]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db
            if team_lookup_fn is not None:
                self.db.litellm_teamtable = self
                self.find_unique = team_lookup_fn

    return MockPrismaClient()


from litellm.proxy._types import (
    LitellmUserRoles,
    Member,
    SpendLogsPayload,
    UserAPIKeyAuth,
)
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.management_endpoints import common_utils
from litellm.proxy.proxy_server import app
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.router import Router
from litellm.types.utils import BudgetConfig


@pytest.mark.asyncio
async def test_is_admin_view_safe_true():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user")
    assert spend_management_endpoints._is_admin_view_safe(auth) is True
    auth_view = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, user_id="admin_view"
    )
    assert spend_management_endpoints._is_admin_view_safe(auth_view) is True


@pytest.mark.asyncio
async def test_is_admin_view_safe_false():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    assert spend_management_endpoints._is_admin_view_safe(auth) is False


@pytest.mark.asyncio
async def test_is_admin_view_safe_exception():
    # Ensure exceptions are swallowed and return False
    class ExplodingAuth:
        @property
        def user_role(self):
            raise RuntimeError("boom")

    assert spend_management_endpoints._is_admin_view_safe(ExplodingAuth()) is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_can_team_member_view_log_none_team_id():
    # team_id=None should immediately return False
    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return None

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, None
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_team_not_found(monkeypatch):
    # Non-existent team should return False
    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return None

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    # Even if admin check would return True, no team means False
    monkeypatch.setattr(
        common_utils,
        "_is_user_team_admin",
        lambda user_api_key_dict, team_obj: True,
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_not_admin(monkeypatch):
    # Existing team but caller is not a team admin and no /spend/logs permission -> False
    class MockTeam:
        team_id = "team_x"
        members_with_roles = [Member(user_id="user_1", role="user")]
        team_member_permissions = None

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "user_1", "role": "user"}],
                "team_member_permissions": self.team_member_permissions,
            }

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    monkeypatch.setattr(
        common_utils,
        "_is_user_team_admin",
        lambda user_api_key_dict, team_obj: False,
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_admin(monkeypatch):
    # Existing team and caller is team admin -> True
    class MockTeam:
        team_id = "team_x"
        members_with_roles = [Member(user_id="user_1", role="admin")]
        team_member_permissions = None

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "user_1", "role": "admin"}],
                "team_member_permissions": self.team_member_permissions,
            }

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is True


def test_can_user_view_spend_log_true_for_internal_user():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="u1")
    assert spend_management_endpoints._can_user_view_spend_log(auth) is True


def test_can_user_view_spend_log_true_for_internal_view_only():
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, user_id="u1"
    )
    assert spend_management_endpoints._can_user_view_spend_log(auth) is True


def test_can_user_view_spend_log_false_without_user_id():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id=None)
    assert spend_management_endpoints._can_user_view_spend_log(auth) is False


def test_can_user_view_spend_log_false_for_other_roles():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    assert spend_management_endpoints._can_user_view_spend_log(auth) is False


@pytest.mark.asyncio
async def test_assert_user_can_view_request_id_rejects_both_users_none():
    """
    API keys with user_id=None must not be treated as owning a log whose user
    field is None (avoid None == None bypass).
    """

    class MockRow:
        user = None
        team_id = None

    class MockSpendLogs:
        async def find_unique(self, where, include=None):
            return MockRow()

    class MockDB:
        def __init__(self):
            self.litellm_spendlogs = MockSpendLogs()

    class MockPrisma:
        def __init__(self):
            self.db = MockDB()

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id=None)
    with pytest.raises(HTTPException) as exc_info:
        await spend_management_endpoints._assert_user_can_view_request_id(
            MockPrisma(), auth, "req-none-user"
        )
    assert exc_info.value.status_code == 403


def test_ui_view_request_response_forbids_non_admin_without_db(client, monkeypatch):
    """
    Without prisma, non-admins cannot be authorized to read request/response
    payloads (including from custom loggers); do not skip RBAC silently.
    """
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user_1",
    )
    try:
        response = client.get(
            "/spend/logs/ui/req-no-db",
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 403
        body = response.json()
        assert "database" in str(body).lower()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


ignored_keys = [
    "request_id",
    "metadata.litellm_call_id",
    "session_id",
    "startTime",
    "endTime",
    "completionStartTime",
    "endTime",
    "request_duration_ms",
    "organization_id",
    "metadata.model_map_information",
    "metadata.usage_object",
    "metadata.cold_storage_object_key",
    "metadata.additional_usage_values.prompt_tokens_details.cache_creation_tokens",
    "metadata.additional_usage_values.completion_tokens_details",
    "metadata.additional_usage_values.prompt_tokens_details",
    "metadata.additional_usage_values.cache_creation_input_tokens",
    "metadata.additional_usage_values.cache_read_input_tokens",
    "metadata.additional_usage_values.inference_geo",
    "metadata.additional_usage_values.speed",
    "metadata.additional_usage_values.service_tier",
    "metadata.additional_usage_values.iterations",
    "metadata.litellm_overhead_time_ms",
    "metadata.cost_breakdown",
    "metadata.user_api_key",
    "metadata.user_api_key_alias",
    "metadata.user_api_key_team_id",
    "metadata.user_api_key_project_id",
    "metadata.user_api_key_project_alias",
    "metadata.user_api_key_org_id",
    "metadata.user_api_key_user_id",
    "metadata.user_api_key_team_alias",
    "metadata.spend_logs_metadata",
    "metadata.requester_ip_address",
    "metadata.status",
    "metadata.proxy_server_request",
    "metadata.error_information",
    "metadata.attempted_retries",
    "metadata.max_retries",
    "metadata.eval_information",
]

MODEL_LIST = [
    {
        "model_name": "azure-gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o-mini",
            "mock_response": "Hello, world!",
            "tags": ["default"],
            "base_model": "gpt-4o-mini",
        },
    },
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def add_anthropic_api_key_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-1234567890")


@pytest.fixture
def disable_budget_sync(monkeypatch):
    """Disable periodic sync during tests"""

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.fixture(autouse=True)
def reset_router_callbacks():
    """Ensure router budget callbacks from previous tests do not leak state."""
    litellm.logging_callback_manager._reset_all_callbacks()
    yield
    litellm.logging_callback_manager._reset_all_callbacks()


@pytest.fixture(autouse=True)
def reset_proxy_auth_globals(monkeypatch):
    """
    Pin proxy auth-related globals to a known baseline so tests don't inherit
    leaked state (master_key, prisma_client, custom auth, cached tokens) from
    earlier tests. Individual tests can still override via their own
    monkeypatch calls — those run after this fixture and revert first.
    """
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "master_key", None)
    monkeypatch.setattr(ps, "user_custom_auth", None)
    monkeypatch.setattr(ps, "general_settings", {})
    try:
        ps.user_api_key_cache.in_memory_cache.cache_dict.clear()
    except AttributeError:
        pass


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_user_id(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_user(where):
        if "user" in where and where["user"] == "test_user_1":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_user),
    )

    start_date, end_date = _default_date_range()

    # Make the request with user_id filter
    response = client.get(
        "/spend/logs/ui",
        params={
            "user_id": "test_user_1",
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    # Assert response
    assert response.status_code == 200
    data = response.json()

    # Verify the response structure
    assert "data" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data

    # Verify the filtered data
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["user"] == "test_user_1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_id_query,expected_request_ids",
    [
        ("session-filter-demo-1", {"req1", "req2"}),
        ("session-filter-demo-2", {"req3"}),
        ("session-filter", {"req1", "req2", "req3"}),
        ("demo", {"req1", "req2", "req3"}),
        ("no-such-session", set()),
    ],
)
async def test_ui_view_spend_logs_with_session_id(
    client, monkeypatch, session_id_query, expected_request_ids
):
    def make_log(request_id, session_id):
        return {
            "id": f"log-{request_id}",
            "request_id": request_id,
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "session_id": session_id,
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        }

    mock_spend_logs = [
        make_log("req1", "session-filter-demo-1"),
        make_log("req2", "session-filter-demo-1"),
        make_log("req3", "session-filter-demo-2"),
        make_log("req4", "unrelated-abc"),
    ]

    def filter_by_session(where):
        session_filter = where.get("session_id")
        if session_filter is None:
            return mock_spend_logs
        return [
            log
            for log in mock_spend_logs
            if session_filter["contains"] in log["session_id"]
        ]

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_session),
    )

    start_date, end_date = _default_date_range()

    response = client.get(
        "/spend/logs/ui",
        params={
            "session_id": session_id_query,
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(expected_request_ids)
    assert {log["request_id"] for log in data["data"]} == expected_request_ids
    assert all(session_id_query in log["session_id"] for log in data["data"])


# Mock spend logs with distinct values for sorting tests.
# req_a: spend=0.10, tokens=500, start/end earliest
# req_b: spend=0.05, tokens=200, start/end 2nd
# req_c: spend=0.20, tokens=50, start/end latest
# req_d: spend=0.01, tokens=100, start/end 3rd
_SORT_TEST_LOGS = [
    {
        "request_id": "req_a",
        "api_key": "sk-test-key",
        "user": "user1",
        "spend": 0.10,
        "total_tokens": 500,
        "startTime": "2025-01-01T00:00:00+00:00",
        "endTime": "2025-01-01T00:01:00+00:00",
        "model": "gpt-3.5-turbo",
    },
    {
        "request_id": "req_b",
        "api_key": "sk-test-key",
        "user": "user1",
        "spend": 0.05,
        "total_tokens": 200,
        "startTime": "2025-01-01T00:00:01+00:00",
        "endTime": "2025-01-01T00:01:01+00:00",
        "model": "gpt-3.5-turbo",
    },
    {
        "request_id": "req_c",
        "api_key": "sk-test-key",
        "user": "user1",
        "spend": 0.20,
        "total_tokens": 50,
        "startTime": "2025-01-01T00:00:03+00:00",
        "endTime": "2025-01-01T00:01:03+00:00",
        "model": "gpt-3.5-turbo",
    },
    {
        "request_id": "req_d",
        "api_key": "sk-test-key",
        "user": "user1",
        "spend": 0.01,
        "total_tokens": 100,
        "startTime": "2025-01-01T00:00:02+00:00",
        "endTime": "2025-01-01T00:01:02+00:00",
        "model": "gpt-3.5-turbo",
    },
]


def _sort_logs(logs, order_clause):
    """Sort logs by the given Prisma-style order clause, e.g. {'spend': 'asc'}."""
    if not order_clause:
        return list(logs)
    key, direction = next(iter(order_clause.items()))
    reverse = direction.lower() == "desc"
    return sorted(logs, key=lambda x: x.get(key, 0), reverse=reverse)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sort_by,sort_order,expected_request_ids",
    [
        # spend: 0.01(d) < 0.05(b) < 0.10(a) < 0.20(c)
        ("spend", "asc", ["req_d", "req_b", "req_a", "req_c"]),
        ("spend", "desc", ["req_c", "req_a", "req_b", "req_d"]),
        # total_tokens: 50(c) < 100(d) < 200(b) < 500(a)
        ("total_tokens", "asc", ["req_c", "req_d", "req_b", "req_a"]),
        ("total_tokens", "desc", ["req_a", "req_b", "req_d", "req_c"]),
        # startTime: 00:00:00(a) < 00:00:01(b) < 00:00:02(d) < 00:00:03(c)
        ("startTime", "asc", ["req_a", "req_b", "req_d", "req_c"]),
        ("startTime", "desc", ["req_c", "req_d", "req_b", "req_a"]),
        # endTime: same ordering as startTime
        ("endTime", "asc", ["req_a", "req_b", "req_d", "req_c"]),
        ("endTime", "desc", ["req_c", "req_d", "req_b", "req_a"]),
        # default when sort_by not provided: startTime desc
        (None, "desc", ["req_c", "req_d", "req_b", "req_a"]),
    ],
)
async def test_ui_view_spend_logs_sort_by_and_sort_order(
    client, monkeypatch, sort_by, sort_order, expected_request_ids
):
    """Test that spend logs are returned in the correct order for each sort_by/sort_order."""
    base_logs = list(_SORT_TEST_LOGS)

    async def mock_count(*args, **kwargs):
        return len(base_logs)

    async def mock_query_raw(sql_query, *params):
        if "COUNT(*)" in sql_query:
            return [{"total_count": len(base_logs)}]
        # Endpoint uses raw SQL with ORDER BY startTime DESC; mock returns sorted data
        order = (
            {"startTime": "desc"}
            if sort_by is None
            else {sort_by: sort_order or "desc"}
        )
        sorted_logs = _sort_logs(base_logs, order)
        page_size = params[-2] if len(params) >= 2 else 50
        skip = params[-1] if len(params) >= 1 else 0
        return [row for row in sorted_logs[skip : skip + page_size]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        start_date = "2024-12-25 00:00:00"
        end_date = "2025-01-02 23:59:59"

        params = {
            "start_date": start_date,
            "end_date": end_date,
        }
        if sort_by is not None:
            params["sort_by"] = sort_by
        if sort_order is not None:
            params["sort_order"] = sort_order

        response = client.get(
            "/spend/logs/ui",
            params=params,
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert "data" in data

        actual_ids = [log["request_id"] for log in data["data"]]
        assert actual_ids == expected_request_ids, (
            f"Expected order {expected_request_ids}, got {actual_ids} "
            f"(sort_by={sort_by}, sort_order={sort_order})"
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sort_by,sort_order",
    [
        ("invalid", "asc"),
        ("spend", "invalid"),
    ],
)
async def test_ui_view_spend_logs_sort_validation_errors(
    client, monkeypatch, sort_by, sort_order
):
    """Test that invalid sort_by and sort_order return 400."""

    async def mock_count(*args, **kwargs):
        return 0

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        start_date = "2024-12-25 00:00:00"
        end_date = "2025-01-02 23:59:59"

        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 400
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_sort_by_request_duration_ms(client, monkeypatch):
    """Test that request_duration_ms is accepted as a valid sort_by field."""
    base_logs = [
        {
            "request_id": "req_fast",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "request_duration_ms": 100,
            "startTime": "2025-01-01T00:00:00+00:00",
            "endTime": "2025-01-01T00:00:00.100000+00:00",
            "model": "gpt-4",
        },
        {
            "request_id": "req_slow",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.05,
            "total_tokens": 50,
            "request_duration_ms": 5000,
            "startTime": "2025-01-01T00:00:01+00:00",
            "endTime": "2025-01-01T00:00:06+00:00",
            "model": "gpt-4",
        },
    ]

    async def mock_count(*args, **kwargs):
        return len(base_logs)

    async def mock_query_raw(sql_query, *params):
        if "COUNT(*)" in sql_query:
            return [{"total_count": len(base_logs)}]
        reverse = "DESC" in sql_query
        sorted_logs = sorted(
            base_logs, key=lambda x: x.get("request_duration_ms", 0), reverse=reverse
        )
        page_size = params[-2] if len(params) >= 2 else 50
        skip = params[-1] if len(params) >= 1 else 0
        return [row for row in sorted_logs[skip : skip + page_size]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
                "sort_by": "request_duration_ms",
                "sort_order": "asc",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        actual_ids = [log["request_id"] for log in data["data"]]
        assert actual_ids == ["req_fast", "req_slow"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sort_order,expected_request_ids",
    [
        ("asc", ["req_anthropic", "req_gpt35", "req_gpt4"]),
        ("desc", ["req_gpt4", "req_gpt35", "req_anthropic"]),
    ],
)
async def test_ui_view_spend_logs_sort_by_model(
    client, monkeypatch, sort_order, expected_request_ids
):
    """Test that model is accepted as a valid sort_by field and orders alphabetically."""
    base_logs = [
        {
            "request_id": "req_gpt4",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:00+00:00",
            "endTime": "2025-01-01T00:00:01+00:00",
            "model": "gpt-4",
        },
        {
            "request_id": "req_anthropic",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:01+00:00",
            "endTime": "2025-01-01T00:00:02+00:00",
            "model": "claude-3-opus",
        },
        {
            "request_id": "req_gpt35",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:02+00:00",
            "endTime": "2025-01-01T00:00:03+00:00",
            "model": "gpt-3.5-turbo",
        },
    ]

    async def mock_count(*args, **kwargs):
        return len(base_logs)

    async def mock_query_raw(sql_query, *params):
        if "COUNT(*)" in sql_query:
            return [{"total_count": len(base_logs)}]
        assert "model" in sql_query
        # model is non-nullable in the schema, so NULLS LAST should NOT be
        # appended — only ttft_ms gets that clause. This guards against
        # accidentally widening the change to all sort columns.
        assert "NULLS LAST" not in sql_query
        reverse = "DESC" in sql_query
        sorted_logs = sorted(
            base_logs, key=lambda x: x.get("model", ""), reverse=reverse
        )
        page_size = params[-2] if len(params) >= 2 else 50
        skip = params[-1] if len(params) >= 1 else 0
        return [row for row in sorted_logs[skip : skip + page_size]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
                "sort_by": "model",
                "sort_order": sort_order,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        actual_ids = [log["request_id"] for log in data["data"]]
        assert actual_ids == expected_request_ids
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_sort_by_ttft_ms(client, monkeypatch):
    """sort_by=ttft_ms orders streaming rows by TTFT and pushes non-streaming rows last (NULLS LAST)."""
    # req_fast_stream: TTFT = 100ms (streaming)
    # req_slow_stream: TTFT = 2000ms (streaming)
    # req_no_stream:   completionStartTime == endTime (non-streaming, treated as NULL)
    # req_null_stream: completionStartTime is null (non-streaming, NULL)
    base_logs = [
        {
            "request_id": "req_fast_stream",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:00+00:00",
            "completionStartTime": "2025-01-01T00:00:00.100000+00:00",
            "endTime": "2025-01-01T00:00:01+00:00",
            "model": "gpt-4",
            "_ttft_ms": 100,
        },
        {
            "request_id": "req_slow_stream",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:02+00:00",
            "completionStartTime": "2025-01-01T00:00:04+00:00",
            "endTime": "2025-01-01T00:00:05+00:00",
            "model": "gpt-4",
            "_ttft_ms": 2000,
        },
        {
            "request_id": "req_no_stream",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:06+00:00",
            "completionStartTime": "2025-01-01T00:00:07+00:00",
            "endTime": "2025-01-01T00:00:07+00:00",
            "model": "gpt-4",
            "_ttft_ms": None,
        },
        {
            "request_id": "req_null_stream",
            "api_key": "sk-test-key",
            "user": "user1",
            "spend": 0.10,
            "total_tokens": 100,
            "startTime": "2025-01-01T00:00:08+00:00",
            "completionStartTime": None,
            "endTime": "2025-01-01T00:00:09+00:00",
            "model": "gpt-4",
            "_ttft_ms": None,
        },
    ]

    async def mock_count(*args, **kwargs):
        return len(base_logs)

    async def mock_query_raw(sql_query, *params):
        if "COUNT(*)" in sql_query:
            return [{"total_count": len(base_logs)}]
        # Endpoint must compute TTFT inline and use NULLS LAST.
        assert "completionStartTime" in sql_query
        assert "NULLS LAST" in sql_query
        reverse = "DESC" in sql_query
        non_null = [r for r in base_logs if r["_ttft_ms"] is not None]
        nulls = [r for r in base_logs if r["_ttft_ms"] is None]
        non_null.sort(key=lambda x: x["_ttft_ms"], reverse=reverse)
        sorted_logs = non_null + nulls
        page_size = params[-2] if len(params) >= 2 else 50
        skip = params[-1] if len(params) >= 1 else 0
        return [
            {k: v for k, v in row.items() if k != "_ttft_ms"}
            for row in sorted_logs[skip : skip + page_size]
        ]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        # asc: fast stream, slow stream, then NULLs (non-streaming) last
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
                "sort_by": "ttft_ms",
                "sort_order": "asc",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200, response.text
        actual_ids = [log["request_id"] for log in response.json()["data"]]
        assert actual_ids[:2] == ["req_fast_stream", "req_slow_stream"]
        assert set(actual_ids[2:]) == {"req_no_stream", "req_null_stream"}

        # desc: slow stream, fast stream, then NULLs still last
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
                "sort_by": "ttft_ms",
                "sort_order": "desc",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200, response.text
        actual_ids = [log["request_id"] for log in response.json()["data"]]
        assert actual_ids[:2] == ["req_slow_stream", "req_fast_stream"]
        assert set(actual_ids[2:]) == {"req_no_stream", "req_null_stream"}
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_team_id(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team2",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_team(where):
        if "team_id" in where and where["team_id"] == "team1":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_team),
    )
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        start_date, end_date = _default_date_range()

        # Make the request with team_id filter
        response = client.get(
            "/spend/logs/ui",
            params={
                "team_id": "team1",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        # Assert response
        assert response.status_code == 200
        data = response.json()

        # Verify the filtered data
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["team_id"] == "team1"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_internal_user_scoped_without_user_id(
    client, monkeypatch
):
    """
    Internal users should only be able to view their own spend even if user_id is not provided.
    """
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "internal_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "internal_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_user(where):
        if "user" in where and where["user"] == "internal_user_1":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_user),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="internal_user_1"
    )

    try:
        start_date, end_date = _default_date_range()

        # No user_id provided; should auto-scope to authenticated internal user's own id
        response = client.get(
            "/spend/logs/ui",
            params={"start_date": start_date, "end_date": end_date},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["user"] == "internal_user_1"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_team_admin_can_view_team_spend(client, monkeypatch):
    """
    Team admins should be able to view team-wide spend when team_id is provided.
    """
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "member1",
            "team_id": "team_admin_team",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "member2",
            "team_id": "team_other",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_team(where):
        if "team_id" in where and where["team_id"] == "team_admin_team":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    class TeamTable:
        team_id = "team_admin_team"
        members_with_roles = [Member(user_id="admin_user", role="admin")]
        team_member_permissions = None

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "admin_user", "role": "admin"}],
                "team_member_permissions": self.team_member_permissions,
            }

    async def team_lookup(where):
        return TeamTable() if where == {"team_id": "team_admin_team"} else None

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_team, team_lookup),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="admin_user"
    )

    try:
        start_date, end_date = _default_date_range()

        response = client.get(
            "/spend/logs/ui",
            params={
                "team_id": "team_admin_team",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["team_id"] == "team_admin_team"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_pagination(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": f"log{i}",
            "request_id": f"req{i}",
            "api_key": "sk-test-key",
            "user": f"test_user_{i % 3}",
            "team_id": f"team{i % 2 + 1}",
            "spend": 0.05 * i,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo" if i % 2 == 0 else "gpt-4",
        }
        for i in range(1, 26)  # 25 records
    ]

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, lambda where: mock_spend_logs),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        start_date, end_date = _default_date_range()

        # Test first page
        response = client.get(
            "/spend/logs/ui",
            params={
                "page": 1,
                "page_size": 10,
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 25
        assert len(data["data"]) == 10
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert data["total_pages"] == 3

        # Test second page
        response = client.get(
            "/spend/logs/ui",
            params={
                "page": 2,
                "page_size": 10,
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 25
        assert len(data["data"]) == 10
        assert data["page"] == 2
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_session_spend_logs_pagination(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "session_id": "session-123",
            "startTime": "2024-01-01T00:00:00Z",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "session_id": "session-123",
            "startTime": "2024-01-02T00:00:00Z",
        },
    ]

    class MockDB:
        async def count(self, *args, **kwargs):
            assert kwargs.get("where") == {"session_id": "session-123"}
            return len(mock_spend_logs)

        async def query_raw(self, sql_query, session_id, page_size, skip):
            # Endpoint uses raw SQL for pagination - verify params
            assert session_id == "session-123"
            assert page_size == 1
            assert skip == 1  # page=2, page_size=1
            assert 'ORDER BY "startTime" DESC' in sql_query
            return [mock_spend_logs[0]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    response = client.get(
        "/spend/logs/session/ui",
        params={"session_id": "session-123", "page": 2, "page_size": 1},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["page"] == 2
    assert data["page_size"] == 1
    assert data["total_pages"] == 2
    assert len(data["data"]) == 1
    assert data["data"][0]["request_id"] == "req1"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_date_range_filter(client, monkeypatch):
    today = datetime.datetime.now(timezone.utc)
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": (today - datetime.timedelta(days=10)).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": (today - datetime.timedelta(days=2)).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_date(where):
        return _filter_logs_by_date_range(mock_spend_logs, where)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_date),
    )

    # Date range that should only include the second log (log1 is 10 days ago, log2 is 2 days ago)
    start_date = (today - datetime.timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = today.strftime("%Y-%m-%d %H:%M:%S")

    response = client.get(
        "/spend/logs/ui",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "log2"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_unauthorized(client):
    # Test without authorization header
    response = client.get("/spend/logs/ui")
    assert response.status_code in (401, 403), response.text

    # Test with invalid authorization
    response = client.get(
        "/spend/logs/ui",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code in (401, 403), response.text


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_status(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "status": "success",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "status": "failure",
        },
    ]

    def filter_by_status(where):
        if "OR" in where:
            return [mock_spend_logs[0]]  # success
        if "status" in where and where["status"].get("equals") == "failure":
            return [mock_spend_logs[1]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_status),
    )

    start_date, end_date = _default_date_range()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        # Test success status
        response = client.get(
            "/spend/logs/ui",
            params={
                "status_filter": "success",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["status"] == "success"

        # Test failure status
        response = client.get(
            "/spend/logs/ui",
            params={
                "status_filter": "failure",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["status"] == "failure"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_model(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "status": "success",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "status": "success",
        },
    ]

    def filter_by_model(where):
        if "model" in where and where["model"] == "gpt-3.5-turbo":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_model),
    )

    start_date, end_date = _default_date_range()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        # Make the request with model filter
        response = client.get(
            "/spend/logs/ui",
            params={
                "model": "gpt-3.5-turbo",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        # Assert response
        assert response.status_code == 200
        data = response.json()

        # Verify the filtered data
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["model"] == "gpt-3.5-turbo"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_model_id(client, monkeypatch):
    """Test that the model_id query param filters spend logs by litellm model deployment id."""
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "model_id": "deployment-id-1",
            "status": "success",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "model_id": "deployment-id-2",
            "status": "success",
        },
    ]

    def filter_by_model_id(where):
        if "model_id" in where and where["model_id"] == "deployment-id-1":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_model_id),
    )

    start_date, end_date = _default_date_range()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "model_id": "deployment-id-1",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["model_id"] == "deployment-id-1"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_model_group(client, monkeypatch):
    """Test that the model_group query param filters spend logs by model group."""
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "model_group": "gpt-3.5-turbo",
            "status": "success",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4-0613",
            "model_group": "gpt-4",
            "status": "success",
        },
    ]

    def filter_by_model_group(where):
        if "model_group" in where and where["model_group"] == "gpt-4":
            return [mock_spend_logs[1]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_model_group),
    )

    start_date, end_date = _default_date_range()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "model_group": "gpt-4",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["model_group"] == "gpt-4"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_key_hash(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key-1",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key-2",
            "user": "test_user_2",
            "team_id": "team2",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_api_key(where):
        if "api_key" in where and where["api_key"] == "sk-test-key-1":
            return [mock_spend_logs[0]]
        return mock_spend_logs

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_api_key),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        start_date, end_date = _default_date_range()

        # Make the request with key_hash filter
        response = client.get(
            "/spend/logs/ui",
            params={
                "api_key": "sk-test-key-1",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        # Assert response
        assert response.status_code == 200
        data = response.json()

        # Verify the filtered data
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["api_key"] == "sk-test-key-1"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


async def _wait_for_mock_call(mock, timeout=10, interval=0.1):
    """Poll until mock has been called at least once, or timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock.call_count > 0:
            return
        await asyncio.sleep(interval)
    mock.assert_called_once()  # will raise with a clear message


class TestSpendLogsPayload:
    def setup_method(self):
        self._original_callbacks = litellm.callbacks[:]
        self._original_cache = litellm.cache
        litellm.cache = None

    def teardown_method(self):
        litellm.callbacks = self._original_callbacks
        litellm.cache = self._original_cache

    @pytest.mark.asyncio
    async def test_spend_logs_payload_e2e(self):
        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        with (
            patch.object(
                litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
                "_insert_spend_log_to_db",
            ) as mock_client,
            patch.object(litellm.proxy.proxy_server, "prisma_client"),
        ):
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, world!"}],
                mock_response="Hello, world!",
                metadata={"user_api_key_end_user_id": "test_user_1"},
            )

            assert response.choices[0].message.content == "Hello, world!"

            await _wait_for_mock_call(mock_client)

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "session_id": "1234567890",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "gpt-4o",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "compression_savings": null, "autorouter_savings": null, "usage_object": {"completion_tokens": 20, "prompt_tokens": 10, "total_tokens": 30, "completion_tokens_details": null, "prompt_tokens_details": null}, "model_map_information": {"model_map_key": "gpt-4o", "model_map_value": {"key": "gpt-4o", "max_tokens": 16384, "max_input_tokens": 128000, "max_output_tokens": 16384, "input_cost_per_token": 2.5e-06, "cache_creation_input_token_cost": null, "cache_read_input_token_cost": 1.25e-06, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": 1.25e-06, "output_cost_per_token_batches": 5e-06, "output_cost_per_token": 1e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_reasoning_token": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "openai", "mode": "chat", "supports_system_messages": true, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": false, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": false, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": true, "supports_reasoning": false, "search_context_cost_per_query": {"search_context_size_low": 0.03, "search_context_size_medium": 0.035, "search_context_size_high": 0.05}, "tpm": null, "rpm": null, "supported_openai_params": ["frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "max_tokens", "max_completion_tokens", "modalities", "prediction", "n", "presence_penalty", "seed", "stop", "stream", "stream_options", "temperature", "top_p", "tools", "tool_choice", "function_call", "functions", "max_retries", "extra_headers", "parallel_tool_calls", "audio", "response_format", "user"]}}, "additional_usage_values": {"completion_tokens_details": null, "prompt_tokens_details": null}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.00022500000000000002,
                    "total_tokens": 30,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "",
                    "model_group": "",
                    "model_id": "",
                    "requester_ip_address": None,
                    "custom_llm_provider": "openai",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"

    def mock_anthropic_response(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-4-sonnet-20250514",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 2095, "output_tokens": 503},
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_spend_logs_payload_success_log_with_api_base(self, monkeypatch):
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        # Clear any env overrides that would change the recorded api_base
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)

        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        client = AsyncHTTPHandler()

        with (
            patch.object(
                litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
                "_insert_spend_log_to_db",
            ) as mock_client,
            patch.object(litellm.proxy.proxy_server, "prisma_client"),
            patch.object(client, "post", side_effect=self.mock_anthropic_response),
        ):
            response = await litellm.acompletion(
                model="claude-4-sonnet-20250514",
                messages=[{"role": "user", "content": "Hello, world!"}],
                metadata={"user_api_key_end_user_id": "test_user_1"},
                client=client,
            )

            assert response.choices[0].message.content == "Hi! My name is Claude."

            await _wait_for_mock_call(mock_client)

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "claude-4-sonnet-20250514",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "compression_savings": null, "autorouter_savings": null, "usage_object": {"completion_tokens": 503, "prompt_tokens": 2095, "total_tokens": 2598, "completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "model_map_information": {"model_map_key": "claude-4-sonnet-20250514", "model_map_value": {"key": "claude-4-sonnet-20250514", "max_tokens": 128000, "max_input_tokens": 200000, "max_output_tokens": 128000, "input_cost_per_token": 3e-06, "cache_creation_input_token_cost": 3.75e-06, "cache_read_input_token_cost": 3e-07, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": null, "output_cost_per_token_batches": null, "output_cost_per_token": 1.5e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "anthropic", "mode": "chat", "supports_system_messages": null, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": true, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": true, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": false, "supports_reasoning": true, "search_context_cost_per_query": null, "tpm": null, "rpm": null, "supported_openai_params": ["stream", "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "extra_headers", "parallel_tool_calls", "response_format", "user", "reasoning_effort", "thinking"]}}, "additional_usage_values": {"completion_tokens_details": {"accepted_prediction_tokens": null, "audio_tokens": null, "reasoning_tokens": null, "rejected_prediction_tokens": null, "text_tokens": 503, "image_tokens": null}, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0, "text_tokens": null, "image_tokens": null}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.01383,
                    "total_tokens": 2598,
                    "prompt_tokens": 2095,
                    "completion_tokens": 503,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "https://api.anthropic.com/v1/messages",
                    "model_group": "",
                    "model_id": "",
                    "requester_ip_address": None,
                    "custom_llm_provider": "anthropic",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"

    @pytest.mark.asyncio
    async def test_spend_logs_payload_success_log_with_router(self, monkeypatch):
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        # Clear any env overrides that would change the recorded api_base
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)

        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        client = AsyncHTTPHandler()

        router = Router(
            model_list=[
                {
                    "model_name": "my-anthropic-model-group",
                    "litellm_params": {
                        "model": "claude-4-sonnet-20250514",
                    },
                    "model_info": {
                        "id": "my-unique-model-id",
                    },
                }
            ]
        )

        with (
            patch.object(
                litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
                "_insert_spend_log_to_db",
            ) as mock_client,
            patch.object(litellm.proxy.proxy_server, "prisma_client"),
            patch.object(client, "post", side_effect=self.mock_anthropic_response),
        ):
            response = await router.acompletion(
                model="my-anthropic-model-group",
                messages=[{"role": "user", "content": "Hello, world!"}],
                metadata={"user_api_key_end_user_id": "test_user_1"},
                client=client,
            )

            assert response.choices[0].message.content == "Hi! My name is Claude."

            await _wait_for_mock_call(mock_client)

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "claude-4-sonnet-20250514",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "compression_savings": null, "autorouter_savings": null, "usage_object": {"completion_tokens": 503, "prompt_tokens": 2095, "total_tokens": 2598, "completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "model_map_information": {"model_map_key": "claude-4-sonnet-20250514", "model_map_value": {"key": "claude-4-sonnet-20250514", "max_tokens": 128000, "max_input_tokens": 200000, "max_output_tokens": 128000, "input_cost_per_token": 3e-06, "cache_creation_input_token_cost": 3.75e-06, "cache_read_input_token_cost": 3e-07, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": null, "output_cost_per_token_batches": null, "output_cost_per_token": 1.5e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "anthropic", "mode": "chat", "supports_system_messages": null, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": true, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": true, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": false, "supports_reasoning": true, "search_context_cost_per_query": null, "tpm": null, "rpm": null, "supported_openai_params": ["stream", "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "extra_headers", "parallel_tool_calls", "response_format", "user", "reasoning_effort", "thinking"]}}, "additional_usage_values": {"completion_tokens_details": {"accepted_prediction_tokens": null, "audio_tokens": null, "reasoning_tokens": null, "rejected_prediction_tokens": null, "text_tokens": 503, "image_tokens": null}, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0, "text_tokens": null, "image_tokens": null}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.01383,
                    "total_tokens": 2598,
                    "prompt_tokens": 2095,
                    "completion_tokens": 503,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "https://api.anthropic.com/v1/messages",
                    "model_group": "my-anthropic-model-group",
                    "model_id": "my-unique-model-id",
                    "requester_ip_address": None,
                    "custom_llm_provider": "anthropic",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"


def _compare_nested_dicts(
    actual: dict, expected: dict, path: str = "", ignore_keys: list[str] = []
) -> list[str]:
    """Compare nested dictionaries and return a list of differences in a human-friendly format."""
    differences = []

    # Check if current path should be ignored
    if path in ignore_keys:
        return differences

    # Check for keys in actual but not in expected
    for key in actual.keys():
        current_path = f"{path}.{key}" if path else key
        if current_path not in ignore_keys and key not in expected:
            differences.append(f"Extra key in actual: {current_path}")

    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if current_path in ignore_keys:
            continue
        if key not in actual:
            differences.append(f"Missing key: {current_path}")
            continue

        actual_value = actual[key]

        # Try to parse JSON strings
        if isinstance(expected_value, str):
            try:
                expected_value = json.loads(expected_value)
            except json.JSONDecodeError:
                pass
        if isinstance(actual_value, str):
            try:
                actual_value = json.loads(actual_value)
            except json.JSONDecodeError:
                pass

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differences.extend(
                _compare_nested_dicts(
                    actual_value, expected_value, current_path, ignore_keys
                )
            )
        elif isinstance(expected_value, dict) or isinstance(actual_value, dict):
            differences.append(
                f"Type mismatch at {current_path}: expected dict, got {type(actual_value).__name__}"
            )
        else:
            # For non-dict values, only report if they're different
            if actual_value != expected_value:
                # Format the values to be more readable
                actual_str = str(actual_value)
                expected_str = str(expected_value)
                if len(actual_str) > 50 or len(expected_str) > 50:
                    actual_str = f"{actual_str[:50]}..."
                    expected_str = f"{expected_str[:50]}..."
                differences.append(
                    f"Value mismatch at {current_path}:\n  expected: {expected_str}\n  got:      {actual_str}"
                )
    return differences


@pytest.mark.asyncio
async def test_global_spend_keys_endpoint_limit_validation(client, monkeypatch):
    """
    Test to ensure that the global_spend_keys endpoint is protected against SQL injection attacks.
    Verifies that the limit parameter is properly parameterized and not directly interpolated.
    """
    # Create a simple mock for prisma client with empty response
    mock_prisma_client = MagicMock()
    mock_db = MagicMock()
    mock_query_raw = AsyncMock(return_value=[])
    mock_db.query_raw = mock_query_raw
    mock_prisma_client.db = mock_db
    # Apply the mock to the prisma_client module
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Override auth to bypass API key validation
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        # Call the endpoint without specifying a limit
        no_limit_response = client.get("/global/spend/keys")
        assert no_limit_response.status_code == 200
        mock_query_raw.assert_called_once_with('SELECT * FROM "Last30dKeysBySpend";')
        # Reset the mock for the next test
        mock_query_raw.reset_mock()
        # Test with valid input
        normal_limit = "10"
        good_input_response = client.get(f"/global/spend/keys?limit={normal_limit}")
        assert good_input_response.status_code == 200
        # Verify the mock was called with the correct parameters
        mock_query_raw.assert_called_once_with(
            'SELECT * FROM "Last30dKeysBySpend" LIMIT $1 ;', 10
        )
        # Reset the mock for the next test
        mock_query_raw.reset_mock()
        # Test with SQL injection payload
        sql_injection_limit = "10; DROP TABLE spend_logs; --"
        response = client.get(f"/global/spend/keys?limit={sql_injection_limit}")
        # Verify the response is a validation error (422)
        assert response.status_code == 422
        # Verify the mock was not called with the SQL injection payload
        # This confirms that the validation happens before the database query
        mock_query_raw.assert_not_called()
        # Reset the mock for the next test
        mock_query_raw.reset_mock()
        # Test with non-numeric input
        non_numeric_limit = "abc"
        response = client.get(f"/global/spend/keys?limit={non_numeric_limit}")
        assert response.status_code == 422
        mock_query_raw.assert_not_called()
        mock_query_raw.reset_mock()
        # Test with negative number
        negative_limit = "-5"
        response = client.get(f"/global/spend/keys?limit={negative_limit}")
        assert response.status_code == 422
        mock_query_raw.assert_not_called()
        mock_query_raw.reset_mock()
        # Test with zero
        zero_limit = "0"
        response = client.get(f"/global/spend/keys?limit={zero_limit}")
        assert response.status_code == 422
        mock_query_raw.assert_not_called()
        mock_query_raw.reset_mock()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_logs_summarize_parameter(client, monkeypatch):
    """Test the new summarize parameter in the /spend/logs endpoint"""
    import datetime
    from datetime import timedelta, timezone

    # Mock spend logs data
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": (
                datetime.datetime.now(timezone.utc) - timedelta(days=1)
            ).isoformat(),
            "model": "gpt-3.5-turbo",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": (
                datetime.datetime.now(timezone.utc) - timedelta(days=1)
            ).isoformat(),
            "model": "gpt-4",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
    ]

    # Mock for unsummarized data (summarize=false)
    class MockDB:
        def __init__(self):
            self.litellm_spendlogs = self

        async def find_many(self, *args, **kwargs):
            # Return individual log entries when summarize=false
            return mock_spend_logs

        async def group_by(self, *args, **kwargs):
            # Return grouped data when summarize=true
            # Simplified mock response for grouped data
            yesterday = datetime.datetime.now(timezone.utc) - timedelta(days=1)
            return [
                {
                    "api_key": "sk-test-key",
                    "user": "test_user_1",
                    "model": "gpt-3.5-turbo",
                    "startTime": yesterday.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "_sum": {"spend": 0.05},
                },
                {
                    "api_key": "sk-test-key",
                    "user": "test_user_1",
                    "model": "gpt-4",
                    "startTime": yesterday.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "_sum": {"spend": 0.10},
                },
            ]

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()

    # Apply the monkeypatch
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set up test dates
    start_date = (datetime.datetime.now(timezone.utc) - timedelta(days=2)).strftime(
        "%Y-%m-%d"
    )
    end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d")

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        # Test 1: summarize=false should return individual log entries
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "summarize": "false",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should return the raw log entries
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "log1"
        assert data[1]["id"] == "log2"
        assert data[0]["request_id"] == "req1"
        assert data[1]["request_id"] == "req2"

        # Test 2: summarize=true should return grouped data
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "summarize": "true",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should return grouped/summarized data
        assert isinstance(data, list)
        # The structure should be different - grouped by date with aggregated spend
        assert "startTime" in data[0]
        assert "spend" in data[0]
        assert "users" in data[0]
        assert "models" in data[0]

        # Test 3: default behavior (no summarize parameter) should maintain backward compatibility
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should return grouped/summarized data (same as summarize=true)
        assert isinstance(data, list)
        assert "startTime" in data[0]
        assert "spend" in data[0]
        assert "users" in data[0]
        assert "models" in data[0]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_tags(client, monkeypatch):
    """Test the /spend/tags endpoint"""

    # Mock the prisma client and get_spend_by_tags function
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock response data
    mock_response = [
        {"individual_request_tag": "tag1", "log_count": 10, "total_spend": 0.15},
        {"individual_request_tag": "tag2", "log_count": 5, "total_spend": 0.08},
    ]

    # Mock the get_spend_by_tags function
    async def mock_get_spend_by_tags(prisma_client, start_date=None, end_date=None):
        return mock_response

    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints.get_spend_by_tags",
        mock_get_spend_by_tags,
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        # Test without date filters
        response = client.get(
            "/spend/tags",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["individual_request_tag"] == "tag1"
        assert data[0]["log_count"] == 10
        assert data[0]["total_spend"] == 0.15

        # Test with date filters
        start_date = "2024-01-01"
        end_date = "2024-01-31"

        response = client.get(
            "/spend/tags",
            params={
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_tags_no_database(client, monkeypatch):
    """Test /spend/tags endpoint when database is not connected"""

    # Mock prisma_client as None
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get(
            "/spend/tags",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 500
        data = response.json()
        # Check the actual error message structure
        assert "error" in data
        assert "Database not connected" in data["error"]["message"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_provider_budget_under(disable_budget_sync):
    """Test that router allows completion when under budget"""
    provider_budget_config = {
        "azure": BudgetConfig(max_budget=0.01, budget_duration="10d")
    }

    router = Router(
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    response = await router.acompletion(
        model="azure-gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    assert response is not None


@pytest.mark.asyncio
async def test_provider_budget_over(disable_budget_sync):
    """Test that router allows completion when over budget"""
    provider_budget_config = {
        "azure": BudgetConfig(max_budget=-0.01, budget_duration="10d")
    }

    router = Router(
        num_retries=0,
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    with pytest.raises(Exception) as e:
        await router.acompletion(
            model="azure-gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
    assert "Exceeded budget for provider" in str(e.value)


@pytest.mark.asyncio
async def test_provider_budget_provider_budgets(disable_budget_sync):
    """Test that provider_budgets() returns correct values"""
    provider = "azure"
    max_budget = -0.01
    budget_duration = "10d"
    provider_budget_config = {
        provider: BudgetConfig(max_budget=max_budget, budget_duration=budget_duration)
    }

    router = Router(
        num_retries=0,
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    with patch("litellm.proxy.proxy_server.llm_router", router):
        response = await spend_management_endpoints.provider_budgets()
        provider_budget_response = response.providers[provider]
        assert provider_budget_response.budget_limit == max_budget
        assert provider_budget_response.time_period == budget_duration


@pytest.mark.asyncio
async def test_view_spend_logs_with_date_range_summarized(client, monkeypatch):
    """
    Tests the /spend/logs endpoint with both start_date and end_date,
    ensuring it returns summarized data and not an empty list.
    This test specifically validates the fix for dates being passed as ISO strings.
    """
    from datetime import datetime, timedelta, timezone

    # This simulates the summarized data that Prisma's `group_by` would return.
    mock_summarized_response = [
        {
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "model": "gpt-4",
            "startTime": (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "_sum": {"spend": 0.15},
        }
    ]

    # This mock class will replace the real Prisma client.
    class MockDB:
        def __init__(self):
            self.litellm_spendlogs = self

        async def group_by(self, *args, **kwargs):
            # We assert that the `gte` and `lte` values are strings in ISO format.
            # If they were datetime objects, this test would fail.
            where_clause = kwargs.get("where", {})
            start_time_filter = where_clause.get("startTime", {})

            assert "gte" in start_time_filter
            assert "lte" in start_time_filter
            assert isinstance(start_time_filter["gte"], str)
            assert isinstance(start_time_filter["lte"], str)
            assert "T" in start_time_filter["gte"]  # Check for ISO format 'T' separator

            # If the assertions pass, return the mock response.
            return mock_summarized_response

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()

    # Apply the monkeypatch to replace the real prisma_client with our mock.
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())

    # Define a date range for the test.
    start_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        # Call the endpoint with both start and end dates.
        # We don't need `summarize=true` as it's the default.
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        # ASSERTIONS
        assert response.status_code == 200
        data = response.json()

        # Check that the response is not empty and has the summarized structure.
        assert isinstance(data, list)
        assert len(data) > 0
        assert "startTime" in data[0]
        assert "spend" in data[0]
        assert "users" in data[0]
        assert "models" in data[0]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_error_code(client):
    """Test filtering spend logs by error code"""
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "metadata": '{"error_information": {"error_code": "404"}}',
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "metadata": '{"error_information": {"error_code": "500"}}',
        },
    ]

    def filter_by_error_code(where):
        if "metadata" in where:
            mf = where["metadata"]
            if mf.get("path") == ["error_information", "error_code"]:
                code = str(mf.get("equals", "")).strip('"')
                if code == "404":
                    return [mock_spend_logs[0]]
                if code == "500":
                    return [mock_spend_logs[1]]
        return mock_spend_logs

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        with patch.object(
            ps,
            "prisma_client",
            make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_error_code),
        ):
            start_date, end_date = _default_date_range()

            response = client.get(
                "/spend/logs/ui",
                params={
                    "error_code": "404",
                    "start_date": start_date,
                    "end_date": end_date,
                },
                headers={"Authorization": "Bearer sk-test"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "log1"
            metadata = data["data"][0]["metadata"]
            assert isinstance(metadata, dict)
            assert "error_information" in metadata
            assert metadata["error_information"]["error_code"] == "404"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_error_message(client):
    """Test filtering spend logs by error message"""
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "metadata": '{"error_information": {"error_message": "Rate limit exceeded"}}',
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "metadata": '{"error_information": {"error_message": "Invalid API key"}}',
        },
    ]

    def filter_by_error_message(where):
        if "metadata" in where:
            mf = where["metadata"]
            if mf.get("path") == ["error_information", "error_message"]:
                msg = mf.get("string_contains")
                if msg == "Rate limit":
                    return [mock_spend_logs[0]]
                if msg == "Invalid API":
                    return [mock_spend_logs[1]]
        return mock_spend_logs

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        with patch.object(
            ps,
            "prisma_client",
            make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_error_message),
        ):
            start_date, end_date = _default_date_range()

            response = client.get(
                "/spend/logs/ui",
                params={
                    "error_message": "Rate limit",
                    "start_date": start_date,
                    "end_date": end_date,
                },
                headers={"Authorization": "Bearer sk-test"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "log1"
            metadata = data["data"][0]["metadata"]
            assert isinstance(metadata, dict)
            assert "error_information" in metadata
            assert (
                "Rate limit exceeded" in metadata["error_information"]["error_message"]
            )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_error_code_and_key_alias(client):
    """Test merging error_code and key_alias filters with AND logic"""
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "metadata": '{"user_api_key_alias": "test-key-1", "error_information": {"error_code": "404"}}',
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "metadata": '{"user_api_key_alias": "test-key-2", "error_information": {"error_code": "500"}}',
        },
        {
            "id": "log3",
            "request_id": "req3",
            "api_key": "sk-test-key",
            "user": "test_user_3",
            "team_id": "team1",
            "spend": 0.15,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
            "metadata": '{"user_api_key_alias": "test-key-1", "error_information": {"error_code": "500"}}',
        },
    ]

    def filter_by_error_code_and_key_alias(where):
        if "AND" in where:
            key_alias = error_code = None
            for cond in where["AND"]:
                if "metadata" in cond:
                    mf = cond["metadata"]
                    if mf.get("path") == ["user_api_key_alias"]:
                        key_alias = mf.get("string_contains")
                    elif mf.get("path") == ["error_information", "error_code"]:
                        error_code = str(mf.get("equals", "")).strip('"')
            if key_alias == "test-key-1" and error_code == "500":
                return [mock_spend_logs[2]]
        return mock_spend_logs

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        with patch.object(
            ps,
            "prisma_client",
            make_ui_spend_logs_mock_prisma(
                mock_spend_logs, filter_by_error_code_and_key_alias
            ),
        ):
            start_date, end_date = _default_date_range()

            response = client.get(
                "/spend/logs/ui",
                params={
                    "error_code": "500",
                    "key_alias": "test-key-1",
                    "start_date": start_date,
                    "end_date": end_date,
                },
                headers={"Authorization": "Bearer sk-test"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "log3"
            metadata = data["data"][0]["metadata"]
            assert isinstance(metadata, dict)
            assert "user_api_key_alias" in metadata
            assert metadata["user_api_key_alias"] == "test-key-1"
            assert "error_information" in metadata
            assert metadata["error_information"]["error_code"] == "500"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_build_ui_spend_logs_response_dict_rows_session_counts():
    """
    Regression test: _build_ui_spend_logs_response must enrich session_total_count
    even when rows are plain dicts (as returned by query_raw) rather than Prisma
    model instances.  Previously getattr(dict, "session_id", None) silently
    returned None, so every row got session_total_count=1 and the UI never
    grouped session rows.
    """
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _build_ui_spend_logs_response,
    )

    session_id = "sess-abc-123"
    api_key = "hashed-key-xyz"
    dict_rows = [
        {"request_id": "req-1", "session_id": session_id, "call_type": "completion", "api_key": api_key},
        {"request_id": "req-2", "session_id": session_id, "call_type": "mcp_tool_call", "api_key": api_key},
        {"request_id": "req-3", "session_id": None, "call_type": "completion", "api_key": api_key},
    ]

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_spendlogs.group_by = AsyncMock(
        return_value=[
            {"session_id": session_id, "_count": {"session_id": 2}},
        ]
    )
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            {
                "session_id": session_id,
                "session_total_spend": 15.0,
                "mcp_tool_call_count": 1,
                "mcp_tool_call_spend": 10.0,
            }
        ]
    )

    result = await _build_ui_spend_logs_response(
        prisma_client=mock_prisma,
        data=dict_rows,
        total_records=3,
        page=1,
        page_size=50,
        total_pages=1,
        enrich_session_counts=True,
    )

    rows = result["data"]
    assert len(rows) == 3

    # Rows with the shared session_id should have session_total_count=2
    assert rows[0]["session_total_count"] == 2
    assert rows[1]["session_total_count"] == 2
    assert rows[0]["mcp_tool_call_count"] == 1
    assert rows[0]["mcp_tool_call_spend"] == 10.0
    assert rows[1]["mcp_tool_call_count"] == 1
    assert rows[1]["mcp_tool_call_spend"] == 10.0

    # Every row in the session carries the full session spend, not just its own
    assert rows[0]["session_total_spend"] == 15.0
    assert rows[1]["session_total_spend"] == 15.0

    # Row without a session_id defaults to 1
    assert rows[2]["session_total_count"] == 1

    # group_by should have been called with the session_id
    mock_prisma.db.litellm_spendlogs.group_by.assert_called_once_with(
        by=["session_id"],
        where={"session_id": {"in": [session_id]}},
        count={"session_id": True},
    )


@pytest.mark.asyncio
async def test_build_ui_spend_logs_response_sums_multi_round_session_spend():
    """
    Regression test for LIT-4342: for a multi-round session the UI must show the
    summed cost of every round, not just the first call.  _build_ui_spend_logs_response
    enriches each row of a session with session_total_spend aggregated across the
    whole session, scoped to the authorized api_keys of the page.
    """
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _build_ui_spend_logs_response,
    )

    session_id = "sess-multi-round"
    api_key = "hashed-key-xyz"
    # Three rounds of the same chat session with different per-call spend.
    dict_rows = [
        {"request_id": "req-1", "session_id": session_id, "call_type": "completion", "api_key": api_key, "spend": 0.01},
        {"request_id": "req-2", "session_id": session_id, "call_type": "completion", "api_key": api_key, "spend": 0.02},
        {"request_id": "req-3", "session_id": session_id, "call_type": "completion", "api_key": api_key, "spend": 0.03},
    ]

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_spendlogs.group_by = AsyncMock(
        return_value=[{"session_id": session_id, "_count": {"session_id": 3}}]
    )
    # The raw aggregate query returns the full session spend (0.01 + 0.02 + 0.03).
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            {
                "session_id": session_id,
                "session_total_spend": 0.06,
                "mcp_tool_call_count": 0,
                "mcp_tool_call_spend": 0.0,
            }
        ]
    )

    result = await _build_ui_spend_logs_response(
        prisma_client=mock_prisma,
        data=dict_rows,
        total_records=3,
        page=1,
        page_size=50,
        total_pages=1,
        enrich_session_counts=True,
    )

    rows = result["data"]
    assert [row["session_total_spend"] for row in rows] == [0.06, 0.06, 0.06]
    # No MCP calls in this session, so MCP fields must not be attached.
    assert all("mcp_tool_call_count" not in row for row in rows)

    # The aggregate must be scoped to the authorized api_keys of the page.
    _, call_args, _ = mock_prisma.db.query_raw.mock_calls[0]
    assert call_args[1] == [session_id]
    assert call_args[2] == [api_key]


# ---------------------------------------------------------------------------
# Tests for /spend/logs team-member permission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_team_member_view_log_with_spend_logs_permission(monkeypatch):
    """
    Non-admin team member WITH /spend/logs permission should be allowed.
    """

    class MockTeam:
        team_id = "team_abc"
        members_with_roles = [Member(user_id="member_1", role="user")]
        team_member_permissions = ["/spend/logs"]

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "member_1", "role": "user"}],
                "team_member_permissions": self.team_member_permissions,
            }

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="member_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_abc"
    )
    assert allowed is True


@pytest.mark.asyncio
async def test_can_team_member_view_log_without_spend_logs_permission(monkeypatch):
    """
    Non-admin team member WITHOUT /spend/logs permission should be denied.
    """

    class MockTeam:
        team_id = "team_abc"
        members_with_roles = [Member(user_id="member_1", role="user")]
        team_member_permissions = ["/key/info"]

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "member_1", "role": "user"}],
                "team_member_permissions": self.team_member_permissions,
            }

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="member_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_abc"
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_ui_view_spend_logs_team_member_with_spend_logs_permission(
    client, monkeypatch
):
    """
    A non-admin team member with /spend/logs permission should see team-wide
    spend logs when filtering by that team_id.
    """
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-key-1",
            "user": "member_1",
            "team_id": "team_perm",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-key-2",
            "user": "member_2",
            "team_id": "team_perm",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_by_team(where):
        if "team_id" in where and where["team_id"] == "team_perm":
            return mock_spend_logs
        return []

    class TeamTable:
        team_id = "team_perm"
        members_with_roles = [Member(user_id="member_1", role="user")]
        team_member_permissions = ["/spend/logs"]

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "member_1", "role": "user"}],
                "team_member_permissions": self.team_member_permissions,
            }

    async def team_lookup(where):
        return TeamTable() if where == {"team_id": "team_perm"} else None

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_by_team, team_lookup),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="member_1"
    )

    try:
        start_date, end_date = _default_date_range()
        response = client.get(
            "/spend/logs/ui",
            params={
                "team_id": "team_perm",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["data"]) == 2
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_team_member_no_permission_blocked(
    client, monkeypatch
):
    """
    A non-admin team member WITHOUT /spend/logs permission should be
    rejected when filtering by team_id.
    """
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-key-1",
            "user": "member_1",
            "team_id": "team_noperm",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    def filter_fn(where):
        return mock_spend_logs

    class TeamTable:
        team_id = "team_noperm"
        members_with_roles = [Member(user_id="member_1", role="user")]
        team_member_permissions = ["/key/info"]

        def model_dump(self):
            return {
                "team_id": self.team_id,
                "members_with_roles": [{"user_id": "member_1", "role": "user"}],
                "team_member_permissions": self.team_member_permissions,
            }

    async def team_lookup(where):
        return TeamTable() if where == {"team_id": "team_noperm"} else None

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        make_ui_spend_logs_mock_prisma(mock_spend_logs, filter_fn, team_lookup),
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="member_1"
    )

    try:
        start_date, end_date = _default_date_range()
        response = client.get(
            "/spend/logs/ui",
            params={
                "team_id": "team_noperm",
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


class _CaptureFilterDB:
    """Mock DB that records the `where` filter passed to find_many."""

    def __init__(self):
        self.litellm_spendlogs = self
        self.captured_where = None

    async def find_many(self, *args, **kwargs):
        self.captured_where = kwargs.get("where")
        return []

    async def group_by(self, *args, **kwargs):
        self.captured_where = kwargs.get("where")
        return []


class _CapturePrismaClient:
    def __init__(self):
        self.db = _CaptureFilterDB()

    def hash_token(self, token):
        return "hashed::" + token


@pytest.mark.asyncio
async def test_view_spend_logs_internal_user_combines_user_with_api_key(
    client, monkeypatch
):
    """Internal users must have their user filter applied alongside api_key."""
    mock_client = _CapturePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)

    start_date = "2024-01-01"
    end_date = "2024-12-31"
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal-user-1",
    )
    try:
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "summarize": "false",
                "api_key": "sk-some-raw-token",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        where = mock_client.db.captured_where
        assert where is not None
        assert where["user"] == "internal-user-1"
        assert where["api_key"] == "hashed::sk-some-raw-token"
        assert "startTime" in where
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_logs_internal_user_combines_user_with_request_id(
    client, monkeypatch
):
    """Internal users must have their user filter applied alongside request_id."""
    mock_client = _CapturePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)

    start_date = "2024-01-01"
    end_date = "2024-12-31"
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal-user-2",
    )
    try:
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "summarize": "false",
                "request_id": "req-abc",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        where = mock_client.db.captured_where
        assert where is not None
        assert where["user"] == "internal-user-2"
        assert where["request_id"] == "req-abc"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_logs_non_date_range_combines_user_with_request_id(
    client, monkeypatch
):
    """Non-date-range path must also combine user + request_id filters."""
    mock_client = _CapturePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal-user-3",
    )
    try:
        response = client.get(
            "/spend/logs",
            params={"request_id": "req-xyz"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        where = mock_client.db.captured_where
        assert where is not None
        assert where["user"] == "internal-user-3"
        assert where["request_id"] == "req-xyz"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_logs_non_date_range_hashes_sk_api_key(client, monkeypatch):
    """Non-date-range path must hash sk- prefixed api_keys before filtering."""
    mock_client = _CapturePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    try:
        response = client.get(
            "/spend/logs",
            params={"api_key": "sk-raw-admin-token"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        where = mock_client.db.captured_where
        assert where is not None
        assert where["api_key"] == "hashed::sk-raw-admin-token"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_view_spend_logs_date_range_hashes_sk_api_key(client, monkeypatch):
    """Date-range path must hash sk- prefixed api_keys before filtering."""
    mock_client = _CapturePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)

    start_date = "2024-01-01"
    end_date = "2024-12-31"
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    try:
        response = client.get(
            "/spend/logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "summarize": "false",
                "api_key": "sk-raw-admin-token",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        where = mock_client.db.captured_where
        assert where is not None
        assert where["api_key"] == "hashed::sk-raw-admin-token"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


class _SpendScopeMockPrismaClient:

    def __init__(self, get_data_returns=None, find_many_returns=None):
        self._get_data_returns = (
            get_data_returns if get_data_returns is not None else []
        )
        self._find_many_returns = (
            find_many_returns if find_many_returns is not None else []
        )
        self.get_data_calls = []
        self.find_many_calls = []

        client = self

        class _VerificationTokenTable:
            async def find_many(self, where=None, order=None, include=None):
                client.find_many_calls.append(
                    {"where": where, "order": order, "include": include}
                )
                return client._find_many_returns

        class _DB:
            def __init__(self):
                self.litellm_verificationtoken = _VerificationTokenTable()

        self.db = _DB()

    async def get_data(self, table_name=None, query_type=None, **kwargs):
        self.get_data_calls.append(
            {"table_name": table_name, "query_type": query_type, **kwargs}
        )
        if query_type == "find_unique":
            return self._get_data_returns[0] if self._get_data_returns else None
        return self._get_data_returns


@pytest.mark.asyncio
async def test_spend_key_fn_proxy_admin_returns_all_keys(client, monkeypatch):
    """Admins keep their existing full-table view of /spend/keys."""
    mock_keys = [
        {"token": "hashed-a", "user_id": "alice", "spend": 10.0},
        {"token": "hashed-b", "user_id": "bob", "spend": 5.0},
    ]
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=mock_keys)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
    )
    try:
        response = client.get(
            "/spend/keys", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        # Admin path: goes through get_data (full table), never the scoped find_many
        assert len(mock_prisma.get_data_calls) == 1
        assert mock_prisma.get_data_calls[0]["table_name"] == "key"
        assert mock_prisma.get_data_calls[0]["query_type"] == "find_all"
        assert mock_prisma.find_many_calls == []
        assert response.json() == mock_keys
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_key_fn_proxy_admin_view_only_returns_all_keys(client, monkeypatch):
    """View-only admins are still admins for this endpoint."""
    mock_keys = [{"token": "hashed-a", "user_id": "alice"}]
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=mock_keys)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, user_id="admin_viewer"
    )
    try:
        response = client.get(
            "/spend/keys", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        assert mock_prisma.find_many_calls == []
        assert len(mock_prisma.get_data_calls) == 1
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY],
)
async def test_spend_key_fn_internal_user_scoped_to_own_keys(client, monkeypatch, role):
    """Both internal-user roles must only see keys they own."""
    caller_owned_keys = [
        {"token": "hashed-mine-1", "user_id": "alice", "spend": 2.0},
        {"token": "hashed-mine-2", "user_id": "alice", "spend": 1.0},
    ]
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=caller_owned_keys)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=role, user_id="alice"
    )
    try:
        response = client.get(
            "/spend/keys", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        # Non-admin path goes through the same get_data helper as admin,
        # but with a user_id scope so only the caller's rows come back.
        assert mock_prisma.find_many_calls == []
        assert len(mock_prisma.get_data_calls) == 1
        call = mock_prisma.get_data_calls[0]
        assert call["table_name"] == "key"
        assert call["query_type"] == "find_all"
        assert call["user_id"] == "alice"
        assert response.json() == caller_owned_keys
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_key_fn_internal_user_without_user_id_returns_empty(
    client, monkeypatch
):
    """
    A non-admin key with no user_id has no tenant scope. Returning the full
    table would re-introduce the leak; return an empty list instead.
    """
    mock_prisma = _SpendScopeMockPrismaClient(
        get_data_returns=[{"token": "do-not-leak"}],
        find_many_returns=[{"token": "do-not-leak"}],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id=None
    )
    try:
        response = client.get(
            "/spend/keys", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        assert response.json() == []
        assert mock_prisma.get_data_calls == []
        assert mock_prisma.find_many_calls == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_proxy_admin_returns_all_users_without_user_id(
    client, monkeypatch
):
    """Admins keep their existing full-table view of /spend/users."""
    mock_users = [
        {"user_id": "alice", "user_email": "alice@example.com", "spend": 1.0},
        {"user_id": "bob", "user_email": "bob@example.com", "spend": 2.0},
    ]
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=mock_users)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
    )
    try:
        response = client.get(
            "/spend/users", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        assert len(mock_prisma.get_data_calls) == 1
        assert mock_prisma.get_data_calls[0]["table_name"] == "user"
        assert mock_prisma.get_data_calls[0]["query_type"] == "find_all"
        assert response.json() == mock_users
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_proxy_admin_can_query_specific_user_id(
    client, monkeypatch
):
    """Admins can still target a specific user_id."""
    mock_user = {
        "user_id": "carol",
        "user_email": "carol@example.com",
        "spend": 7.0,
    }
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=[mock_user])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
    )
    try:
        response = client.get(
            "/spend/users",
            params={"user_id": "carol"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        assert len(mock_prisma.get_data_calls) == 1
        assert mock_prisma.get_data_calls[0]["query_type"] == "find_unique"
        assert mock_prisma.get_data_calls[0]["user_id"] == "carol"
        assert response.json() == [mock_user]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY],
)
async def test_spend_user_fn_internal_user_scoped_without_user_id(
    client, monkeypatch, role
):
    """No user_id supplied -> must query the caller's own row, not the table."""
    own_row = {"user_id": "alice", "user_email": "alice@example.com", "spend": 3.0}
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=[own_row])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=role, user_id="alice"
    )
    try:
        response = client.get(
            "/spend/users", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        assert len(mock_prisma.get_data_calls) == 1
        assert mock_prisma.get_data_calls[0]["query_type"] == "find_unique"
        assert mock_prisma.get_data_calls[0]["user_id"] == "alice"
        assert response.json() == [own_row]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_internal_user_supplying_other_user_id_returns_403(
    client, monkeypatch
):
    """
    An internal user passing user_id=victim must be rejected outright, not
    silently rewritten. A 403 makes the attempt observable in logs.
    """
    leaked_victim_row = {
        "user_id": "victim",
        "user_email": "victim@example.com",
        "spend": 999.0,
    }
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=[leaked_victim_row])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice"
    )
    try:
        response = client.get(
            "/spend/users",
            params={"user_id": "victim"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 403
        assert mock_prisma.get_data_calls == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_internal_user_supplying_own_user_id_is_allowed(
    client, monkeypatch
):
    """
    Passing your own user_id explicitly is fine — the 403 only fires when
    the supplied id differs from the caller's.
    """
    own_row = {"user_id": "alice", "user_email": "alice@example.com", "spend": 3.0}
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=[own_row])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice"
    )
    try:
        response = client.get(
            "/spend/users",
            params={"user_id": "alice"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        assert len(mock_prisma.get_data_calls) == 1
        assert mock_prisma.get_data_calls[0]["query_type"] == "find_unique"
        assert mock_prisma.get_data_calls[0]["user_id"] == "alice"
        assert response.json() == [own_row]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_internal_user_without_user_id_returns_empty(
    client, monkeypatch
):
    """
    A non-admin key with no user_id has no tenant scope -> return empty,
    never the full table. Same defensive contract as /spend/keys.
    """
    mock_prisma = _SpendScopeMockPrismaClient(
        get_data_returns=[{"user_id": "do-not-leak"}]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, user_id=None
    )
    try:
        response = client.get(
            "/spend/users", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        assert response.json() == []
        assert mock_prisma.get_data_calls == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_spend_user_fn_strips_password_field(client, monkeypatch):
    """
    Existing password-redaction behavior must be preserved on the scoped
    path so we don't regress a separate disclosure when adding the fix.
    """
    own_row = {
        "user_id": "alice",
        "user_email": "alice@example.com",
        "password": "hashed-password-must-not-leak",
        "spend": 1.0,
    }
    mock_prisma = _SpendScopeMockPrismaClient(get_data_returns=[own_row])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice"
    )
    try:
        response = client.get(
            "/spend/users", headers={"Authorization": "Bearer sk-test"}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert "password" not in body[0]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_rehydrates_metadata_jsonb_text(client, monkeypatch):
    """
    Regression for #29674: query_raw returns the JSONB `metadata` column as a
    string, so failure rows (status="failure", error_information.error_code=...)
    looked like successes at the UI layer because metadata.status was the
    string ".status" attribute lookup on a str. The endpoint must re-hydrate
    `metadata` to a dict before returning.
    """
    failure_metadata = {
        "status": "failure",
        "error_information": {
            "error_code": "403",
            "error_message": "Forbidden by upstream",
        },
        "user_api_key_alias": "alias-1",
    }

    raw_row = {
        "request_id": "req-failure-1",
        "call_type": "completion",
        "api_key": "hashed-key",
        "spend": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "startTime": "2025-01-01T00:00:00Z",
        "endTime": "2025-01-01T00:00:01Z",
        "completionStartTime": None,
        "model": "gpt-4o",
        "model_id": None,
        "model_group": None,
        "custom_llm_provider": "openai",
        "api_base": None,
        "user": "u",
        "metadata": json.dumps(failure_metadata),  # JSONB column comes back as str
        "cache_hit": None,
        "cache_key": None,
        "request_tags": None,
        "team_id": None,
        "organization_id": None,
        "end_user": None,
        "requester_ip_address": None,
        "session_id": None,
        "status": "failure",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "request_duration_ms": 1000,
    }

    async def mock_count(*args, **kwargs):
        return 1

    async def mock_query_raw(sql_query, *params):
        return [{**raw_row, "total_count": 1}]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"], "expected one row in data"
        row = body["data"][0]
        md = row["metadata"]
        # The bug had metadata returned as a JSON string; the fix re-hydrates
        # it so the dashboard's metadata.status / metadata.error_information
        # accessors work.
        assert isinstance(md, dict), f"metadata should be dict, got {type(md)}"
        assert md["status"] == "failure"
        assert md["error_information"]["error_code"] == "403"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_metadata_invalid_json_falls_back_to_empty_dict(
    client, monkeypatch
):
    """
    Defensive: if `metadata` is somehow not valid JSON, fall back to {} rather
    than 500-ing the whole UI page.
    """
    raw_row = {
        "request_id": "req-bad-json",
        "call_type": "completion",
        "api_key": "hashed-key",
        "spend": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "startTime": "2025-01-01T00:00:00Z",
        "endTime": "2025-01-01T00:00:01Z",
        "completionStartTime": None,
        "model": "gpt-4o",
        "model_id": None,
        "model_group": None,
        "custom_llm_provider": "openai",
        "api_base": None,
        "user": "u",
        "metadata": "{not-json",
        "cache_hit": None,
        "cache_key": None,
        "request_tags": None,
        "team_id": None,
        "organization_id": None,
        "end_user": None,
        "requester_ip_address": None,
        "session_id": None,
        "status": "success",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "request_duration_ms": 500,
    }

    async def mock_count(*args, **kwargs):
        return 1

    async def mock_query_raw(sql_query, *params):
        return [{**raw_row, "total_count": 1}]

    class MockPrismaClient:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_spendlogs = MagicMock()
            self.db.litellm_spendlogs.count = AsyncMock(side_effect=mock_count)
            self.db.query_raw = AsyncMock(side_effect=mock_query_raw)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get(
            "/spend/logs/ui",
            params={
                "start_date": "2024-12-25 00:00:00",
                "end_date": "2025-01-02 23:59:59",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"]
        assert body["data"][0]["metadata"] == {}
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


class _FakeColdStorageLogger:
    """Injectable cold storage logger that records the object key it was asked for."""

    def __init__(self, payload):
        self._payload = payload
        self.requested_object_keys = []

    async def get_proxy_server_request_from_cold_storage_with_object_key(
        self, object_key
    ):
        self.requested_object_keys.append(object_key)
        return self._payload


def _cold_storage_handler(payload):
    from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler

    logger = _FakeColdStorageLogger(payload)
    return ColdStorageHandler(cold_storage_logger=logger), logger


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("{}", False),
        ("[]", False),
        ("null", False),
        ('{"a": 1}', True),
        ({}, False),
        ({"a": 1}, True),
        ([], False),
        ([1], True),
        (5, True),
    ],
)
def test_spend_log_field_has_content(value, expected):
    assert spend_management_endpoints._spend_log_field_has_content(value) is expected


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, None),
        ("{}", None),
        ("not-json", None),
        ({"cold_storage_object_key": ""}, None),
        ({"cold_storage_object_key": "k/req-1.json"}, "k/req-1.json"),
        ('{"cold_storage_object_key": "k/req-2.json"}', "k/req-2.json"),
    ],
)
def test_cold_storage_object_key_from_metadata(metadata, expected):
    assert (
        spend_management_endpoints._cold_storage_object_key_from_metadata(metadata)
        == expected
    )


@pytest.mark.asyncio
async def test_resolve_payload_prefers_pg_and_skips_cold_storage():
    handler, logger = _cold_storage_handler({"messages": "X", "response": "Y"})
    row = {
        "messages": "{}",
        "response": '{"choices": [{"message": {"content": "hi"}}]}',
        "proxy_server_request": "{}",
        "metadata": {"cold_storage_object_key": "k/req.json"},
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert resolved.response == '{"choices": [{"message": {"content": "hi"}}]}'
    assert logger.requested_object_keys == []


@pytest.mark.asyncio
async def test_resolve_payload_fetches_from_cold_storage_when_pg_empty():
    cold_payload = {
        "messages": [{"role": "user", "content": "what is 2+2"}],
        "response": {"choices": [{"message": {"content": "4"}}]},
        "proxy_server_request": {"body": {"model": "gpt-4o-mini"}},
    }
    handler, logger = _cold_storage_handler(cold_payload)
    row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": {"cold_storage_object_key": "llm-gateway/prod/req-42.json"},
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert logger.requested_object_keys == ["llm-gateway/prod/req-42.json"]
    assert resolved.messages == cold_payload["messages"]
    assert resolved.response == cold_payload["response"]
    assert resolved.proxy_server_request == cold_payload["proxy_server_request"]


@pytest.mark.asyncio
async def test_resolve_payload_metadata_as_json_string():
    cold_payload = {"messages": "in", "response": "out", "proxy_server_request": None}
    handler, logger = _cold_storage_handler(cold_payload)
    row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": json.dumps({"cold_storage_object_key": "k/str-meta.json"}),
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert logger.requested_object_keys == ["k/str-meta.json"]
    assert resolved.response == "out"


@pytest.mark.asyncio
async def test_resolve_payload_no_object_key_returns_empty_without_fetch():
    handler, logger = _cold_storage_handler({"messages": "should-not-be-used"})
    row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": {},
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert logger.requested_object_keys == []
    assert resolved == spend_management_endpoints.RequestResponsePayload(
        "{}", "{}", "{}"
    )


@pytest.mark.asyncio
async def test_resolve_payload_cold_storage_miss_falls_back_to_pg_values():
    handler, logger = _cold_storage_handler(None)
    row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": {"cold_storage_object_key": "k/missing.json"},
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert logger.requested_object_keys == ["k/missing.json"]
    assert resolved == spend_management_endpoints.RequestResponsePayload(
        "{}", "{}", "{}"
    )


@pytest.mark.asyncio
async def test_resolve_payload_cold_storage_exception_falls_back_to_pg_values():
    """A backend error during fetch degrades to PG values instead of bubbling a 500."""

    class _RaisingLogger:
        async def get_proxy_server_request_from_cold_storage_with_object_key(
            self, object_key
        ):
            raise RuntimeError("cold storage backend unavailable")

    from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler

    handler = ColdStorageHandler(cold_storage_logger=_RaisingLogger())
    row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": {"cold_storage_object_key": "k/boom.json"},
    }

    resolved = await spend_management_endpoints._resolve_request_response_payload(
        row, cold_storage_handler=handler
    )

    assert resolved == spend_management_endpoints.RequestResponsePayload(
        "{}", "{}", "{}"
    )


@pytest.mark.asyncio
async def test_cold_storage_handler_uses_injected_logger():
    from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler

    logger = _FakeColdStorageLogger({"messages": "in", "response": "out"})
    handler = ColdStorageHandler(cold_storage_logger=logger)

    result = await handler.get_proxy_server_request_from_cold_storage_with_object_key(
        object_key="k/req.json"
    )

    assert result == {"messages": "in", "response": "out"}
    assert logger.requested_object_keys == ["k/req.json"]


@pytest.mark.asyncio
async def test_cold_storage_handler_returns_none_when_no_logger_configured(monkeypatch):
    from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler

    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None, raising=False)
    handler = ColdStorageHandler()

    result = await handler.get_proxy_server_request_from_cold_storage_with_object_key(
        object_key="k/req.json"
    )

    assert result is None


@pytest.mark.asyncio
async def test_cold_storage_handler_resolves_configured_logger_from_registry(
    monkeypatch,
):
    from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler

    logger = _FakeColdStorageLogger({"messages": "from-registry"})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2", raising=False)
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "get_active_custom_logger_for_callback_name",
        lambda name: logger if name == "s3_v2" else None,
    )
    handler = ColdStorageHandler()

    result = await handler.get_proxy_server_request_from_cold_storage_with_object_key(
        object_key="k/req.json"
    )

    assert result == {"messages": "from-registry"}
    assert logger.requested_object_keys == ["k/req.json"]


def test_ui_view_request_response_reads_from_cold_storage(client, monkeypatch):
    """End-to-end: a placeholder row with a cold_storage_object_key is served from
    cold storage through the detail endpoint."""
    from types import SimpleNamespace

    placeholder_row = {
        "messages": "{}",
        "response": "{}",
        "proxy_server_request": "{}",
        "metadata": {"cold_storage_object_key": "k/cold.json"},
    }

    async def _query_raw(_sql, *_args):
        return [placeholder_row]

    fake_prisma = SimpleNamespace(db=SimpleNamespace(query_raw=_query_raw))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", fake_prisma)

    cold_logger = _FakeColdStorageLogger(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "response": {"choices": [{"message": {"content": "hello"}}]},
            "proxy_server_request": None,
        }
    )
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2", raising=False)
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "get_active_additional_logging_utils_from_custom_logger",
        lambda: [],
    )
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "get_active_custom_logger_for_callback_name",
        lambda name: cold_logger if name == "s3_v2" else None,
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_1"
    )
    try:
        response = client.get(
            "/spend/logs/ui/req-cold",
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["response"] == {"choices": [{"message": {"content": "hello"}}]}
        assert cold_logger.requested_object_keys == ["k/cold.json"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

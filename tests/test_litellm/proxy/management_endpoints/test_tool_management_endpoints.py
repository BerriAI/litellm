"""
Unit tests for tool management endpoints (/v1/tool/*).
Uses FastAPI TestClient with mocked DB functions.

Patches target the source modules (litellm.proxy.db.tool_registry_writer.*
and litellm.proxy.proxy_server.prisma_client) because the endpoint code
imports these inside function bodies to avoid circular imports.
"""

import inspect
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.actions import LiteLLM_TeamTableActions


from litellm.proxy.management_endpoints.tool_management_endpoints import router
from litellm.types.tool_management import LiteLLM_ToolTableRow

# --- helpers ---


def _make_tool_row(
    tool_name: str = "my_tool",
    input_policy: str = "untrusted",
    origin: Optional[str] = None,
) -> LiteLLM_ToolTableRow:
    now = datetime.now(timezone.utc)
    return LiteLLM_ToolTableRow(
        tool_id="uuid-1",
        tool_name=tool_name,
        origin=origin,
        input_policy=input_policy,  # type: ignore[arg-type]
        assignments={},
        created_at=now,
        updated_at=now,
    )


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the tool management router."""
    app = FastAPI()
    app.include_router(router)
    return app


# Stub the auth dependency so we don't need a real proxy running.
def _override_auth():
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="sk-test", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


# A real (non-None) prisma stub for truthiness checks.
_MOCK_PRISMA = MagicMock()


def _rollup_row(date: str, tool_name: str, spend: float, request_count: int, total_tokens: int) -> MagicMock:
    row = MagicMock()
    row.date = date
    row.tool_name = tool_name
    row.spend = spend
    row.request_count = request_count
    row.total_tokens = total_tokens
    return row


def _group_row(tool_name: str, spend: float, request_count: int, total_tokens: int) -> dict:
    return {"tool_name": tool_name, "_sum": {"spend": spend, "total_tokens": total_tokens, "request_count": request_count}}


class FakeTeamTable:
    """Stand-in for ``prisma_client.db.litellm_teamtable``.

    ``AsyncMock`` accepts any keyword argument, so a plain mock cannot catch a call
    the generated prisma client would reject at runtime. This double binds every
    call against the real action signature, so an unsupported kwarg (e.g. ``select``)
    raises the same ``TypeError`` the proxy surfaces as an HTTP 500.
    """

    def __init__(self, rows: Sequence[MagicMock], updated_count: int = 1):
        self._rows = tuple(rows)
        self._updated_count = updated_count
        self.find_unique_calls: list[dict[str, object]] = []
        self.update_many_calls: list[dict[str, object]] = []

    async def find_unique(self, **kwargs: object) -> Optional[MagicMock]:
        inspect.signature(LiteLLM_TeamTableActions.find_unique).bind(self, **kwargs)
        index = len(self.find_unique_calls)
        self.find_unique_calls.append(kwargs)
        return self._rows[index] if index < len(self._rows) else None

    async def update_many(self, **kwargs: object) -> int:
        inspect.signature(LiteLLM_TeamTableActions.update_many).bind(self, **kwargs)
        self.update_many_calls.append(kwargs)
        return self._updated_count


def _team_row(object_permission_id: Optional[str]) -> MagicMock:
    row = MagicMock()
    row.object_permission_id = object_permission_id
    return row


def _team_policy_prisma(team_table: FakeTeamTable) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_teamtable = team_table
    prisma.db.litellm_objectpermissiontable.create = AsyncMock()
    prisma.db.litellm_objectpermissiontable.delete = AsyncMock()
    return prisma


def _rollup_prisma(group_rows: list, daily_rows: list | None = None) -> MagicMock:
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    prisma.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_spendlogtoolindex.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_dailytoolspend.group_by = AsyncMock(return_value=group_rows)
    prisma.db.litellm_dailytoolspend.find_many = AsyncMock(return_value=daily_rows or [])
    return prisma


# --- test class ---


class TestToolManagementEndpoints:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch(
        "litellm.proxy.db.tool_registry_writer.list_tools",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_list_tools_returns_200(self, mock_db_list):
        mock_db_list.return_value = [_make_tool_row()]

        resp = self.client.get("/v1/tool/list")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["tools"][0]["tool_name"] == "my_tool"

    @patch(
        "litellm.proxy.db.tool_registry_writer.list_tools",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_list_tools_with_policy_filter(self, mock_db_list):
        mock_db_list.return_value = [_make_tool_row(input_policy="blocked")]

        resp = self.client.get("/v1/tool/list?input_policy=blocked")
        assert resp.status_code == 200
        assert resp.json()["tools"][0]["input_policy"] == "blocked"

    @patch(
        "litellm.proxy.db.tool_registry_writer.get_tool",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_get_tool_found(self, mock_db_get):
        mock_db_get.return_value = _make_tool_row(tool_name="tool_a")

        resp = self.client.get("/v1/tool/tool_a")
        assert resp.status_code == 200
        assert resp.json()["tool_name"] == "tool_a"

    @patch(
        "litellm.proxy.db.tool_registry_writer.get_tool",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_get_tool_not_found_returns_404(self, mock_db_get):
        mock_db_get.return_value = None

        resp = self.client.get("/v1/tool/nonexistent", follow_redirects=True)
        assert resp.status_code == 404

    @patch(
        "litellm.proxy.db.tool_registry_writer.update_tool_policy",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_update_tool_policy_blocked(self, mock_db_update):
        mock_db_update.return_value = _make_tool_row(input_policy="blocked")

        resp = self.client.post(
            "/v1/tool/policy",
            json={"tool_name": "my_tool", "input_policy": "blocked"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["input_policy"] == "blocked"
        assert body["updated"] is True

    @patch(
        "litellm.proxy.db.tool_registry_writer.add_tool_to_object_permission_blocked",
        new_callable=AsyncMock,
    )
    def test_update_tool_policy_for_team_does_not_500_on_unsupported_prisma_kwarg(self, mock_block):
        """
        Regression: POST /v1/tool/policy with a team_id returned 500 with
        "LiteLLM_TeamTableActions.find_unique() got an unexpected keyword argument
        'select'". The team lookup passed select={"object_permission_id": True}, and
        the generated prisma client has no select kwarg, so the call raised TypeError
        that the handler turned into a 500.
        """
        mock_block.return_value = True
        team_table = FakeTeamTable([_team_row("existing-op-id")])

        with patch("litellm.proxy.proxy_server.prisma_client", _team_policy_prisma(team_table)), patch(
            "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry"
        ) as mock_registry:
            mock_registry.return_value.is_initialized.return_value = False
            resp = self.client.post(
                "/v1/tool/policy",
                json={"tool_name": "my_tool", "input_policy": "blocked", "team_id": "team-123"},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["team_id"] == "team-123"
        assert team_table.find_unique_calls == [{"where": {"team_id": "team-123"}}]
        mock_block.assert_awaited_once_with(
            prisma_client=mock_block.await_args.kwargs["prisma_client"],
            object_permission_id="existing-op-id",
            tool_name="my_tool",
        )

    @patch(
        "litellm.proxy.db.tool_registry_writer.add_tool_to_object_permission_blocked",
        new_callable=AsyncMock,
    )
    def test_update_tool_policy_for_team_reresolves_when_permission_race_is_lost(self, mock_block):
        """
        Same regression on the second team lookup: when the team has no object
        permission yet and a concurrent writer wins the update_many (0 rows updated),
        the handler re-reads the team to pick up the winner's id. That fallback read
        passed the same unsupported select kwarg.
        """
        mock_block.return_value = True
        team_table = FakeTeamTable(
            [_team_row(None), _team_row("winner-op-id")],
            updated_count=0,
        )

        with patch("litellm.proxy.proxy_server.prisma_client", _team_policy_prisma(team_table)), patch(
            "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry"
        ) as mock_registry:
            mock_registry.return_value.is_initialized.return_value = False
            resp = self.client.post(
                "/v1/tool/policy",
                json={"tool_name": "my_tool", "input_policy": "blocked", "team_id": "team-123"},
            )

        assert resp.status_code == 200, resp.text
        assert team_table.find_unique_calls == [
            {"where": {"team_id": "team-123"}},
            {"where": {"team_id": "team-123"}},
        ]
        assert mock_block.await_args.kwargs["object_permission_id"] == "winner-op-id"

    @patch("litellm.proxy.proxy_server.prisma_client", None)
    def test_list_tools_no_db_returns_500(self):
        resp = self.client.get("/v1/tool/list")
        assert resp.status_code == 500

    def test_update_tool_policy_invalid_policy_returns_422(self):
        resp = self.client.post(
            "/v1/tool/policy",
            json={"tool_name": "my_tool", "input_policy": "invalid_value"},
        )
        assert resp.status_code == 422

    def test_tool_spend_route_not_shadowed_by_get_tool(self):
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend")
        assert resp.status_code == 200
        assert resp.json()["by_tool"] == []

    def test_tool_spend_serves_sql_aggregates_and_daily_series(self):
        group_rows = [
            _group_row("search", spend=5.0, request_count=3, total_tokens=150),
            _group_row("read_file", spend=2.0, request_count=3, total_tokens=300),
        ]
        daily_rows = [
            _rollup_row("2026-07-01", "search", spend=1.0, request_count=2, total_tokens=100),
            _rollup_row("2026-07-01", "read_file", spend=2.0, request_count=3, total_tokens=300),
            _rollup_row("2026-07-02", "search", spend=4.0, request_count=1, total_tokens=50),
        ]
        prisma = _rollup_prisma(group_rows, daily_rows)
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        body = resp.json()
        assert [t["tool_name"] for t in body["by_tool"]] == ["search", "read_file"]
        search = body["by_tool"][0]
        assert search["spend"] == 5.0
        assert search["call_count"] == 3
        assert search["total_tokens"] == 150
        assert len(body["daily"]) == 3
        assert body["daily"][0]["call_count"] == 2
        assert body["start_date"] == "2026-07-01"
        assert body["end_date"] == "2026-07-02"

    def test_tool_spend_coerces_bigint_string_sums(self):
        # prisma group_by returns BigInt sums as strings ("808"); the response
        # must coerce them to ints rather than 500 on validation.
        group_rows = [{"tool_name": "search", "_sum": {"spend": 0.5, "total_tokens": "808", "request_count": "3"}}]
        prisma = _rollup_prisma(group_rows)
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        assert resp.json()["by_tool"][0]["total_tokens"] == 808
        assert resp.json()["by_tool"][0]["call_count"] == 3

    def test_tool_spend_daily_restricted_to_top_tools_and_capped(self):
        from litellm.constants import TOOL_SPEND_TOP_TOOLS

        group_rows = [_group_row("search", spend=5.0, request_count=1, total_tokens=10)]
        prisma = _rollup_prisma(group_rows)
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        group_kwargs = prisma.db.litellm_dailytoolspend.group_by.await_args.kwargs
        assert group_kwargs["take"] == TOOL_SPEND_TOP_TOOLS
        assert group_kwargs["order"] == {"_sum": {"spend": "desc"}}
        daily_where = prisma.db.litellm_dailytoolspend.find_many.await_args.kwargs["where"]
        assert daily_where["tool_name"] == {"in": ["search"]}

    def test_tool_spend_skips_daily_query_when_no_tools(self):
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        prisma.db.litellm_dailytoolspend.find_many.assert_not_awaited()

    @patch("litellm.proxy.proxy_server.prisma_client", None)
    def test_tool_spend_no_db_returns_500(self):
        resp = self.client.get("/v1/tool/spend")
        assert resp.status_code == 500

    def test_tool_spend_reads_rollup_only_never_spendlogs(self):
        # Regression for the GA blocker: the dashboard aggregate must be served
        # entirely from LiteLLM_DailyToolSpend; any query_raw or SpendLogs table
        # access on this path reintroduces the per-request scan.
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        prisma.db.query_raw.assert_not_awaited()
        prisma.db.litellm_spendlogs.find_many.assert_not_awaited()
        prisma.db.litellm_spendlogtoolindex.find_many.assert_not_awaited()
        prisma.db.litellm_dailytoolspend.group_by.assert_awaited_once()

    def test_tool_spend_windows_rollup_by_inclusive_date_strings(self):
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        where = prisma.db.litellm_dailytoolspend.group_by.await_args.kwargs["where"]
        assert where == {"date": {"gte": "2026-07-01", "lte": "2026-07-02"}}
        assert resp.json()["end_date"] == "2026-07-02"

    def test_tool_spend_wide_range_served_fully(self):
        # Regression: the 30-day clamp is gone; a 182-day request is served as
        # requested because the rollup read is O(tools x dates).
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-01-01&end_date=2026-07-01")
        assert resp.status_code == 200
        where = prisma.db.litellm_dailytoolspend.group_by.await_args.kwargs["where"]
        assert where == {"date": {"gte": "2026-01-01", "lte": "2026-07-01"}}
        assert resp.json()["start_date"] == "2026-01-01"
        assert resp.json()["end_date"] == "2026-07-01"

    def test_tool_spend_defaults_to_trailing_30_days(self):
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend")
        assert resp.status_code == 200
        today = datetime.now(timezone.utc)
        assert resp.json()["end_date"] == today.strftime("%Y-%m-%d")
        assert resp.json()["start_date"] == (today - timedelta(days=30)).strftime("%Y-%m-%d")

    @pytest.mark.parametrize(
        "query",
        [
            "start_date=not-a-date",
            "start_date=2026-02-30",
            "start_date=07/01/2026",
            "end_date=2026-13-01",
            "end_date=20260701",
        ],
    )
    def test_tool_spend_malformed_date_returns_400(self, query: str):
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get(f"/v1/tool/spend?{query}")
        assert resp.status_code == 400
        assert "Invalid date format" in resp.json()["detail"]
        prisma.db.litellm_dailytoolspend.group_by.assert_not_awaited()

    def test_tool_spend_non_admin_returns_403(self):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="sk-user", user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER
        )
        client = TestClient(app, raise_server_exceptions=True)
        prisma = _rollup_prisma([])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = client.get("/v1/tool/spend")
        assert resp.status_code == 403
        prisma.db.litellm_dailytoolspend.group_by.assert_not_awaited()

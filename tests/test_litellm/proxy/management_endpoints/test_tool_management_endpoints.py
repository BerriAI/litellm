"""
Unit tests for tool management endpoints (/v1/tool/*).
Uses FastAPI TestClient with mocked DB functions.

Patches target the source modules (litellm.proxy.db.tool_registry_writer.*
and litellm.proxy.proxy_server.prisma_client) because the endpoint code
imports these inside function bodies to avoid circular imports.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.management_endpoints.tool_management_endpoints import (
    _build_tool_spend_response,
    _ToolSpendRow,
    router,
)
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
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(return_value=[])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend")
        assert resp.status_code == 200
        assert resp.json()["by_tool"] == []

    def test_tool_spend_aggregates_and_sorts(self):
        rows = [
            {"date": "2026-07-01", "tool_name": "search", "call_count": 2, "spend": 1.0, "total_tokens": 100},
            {"date": "2026-07-02", "tool_name": "search", "call_count": 1, "spend": 4.0, "total_tokens": 50},
            {"date": "2026-07-01", "tool_name": "read_file", "call_count": 3, "spend": 2.0, "total_tokens": 300},
        ]
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(side_effect=[rows, [{"total_spend": 5.5}]])
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
        assert body["start_date"] == "2026-07-01"
        assert body["end_date"] == "2026-07-02"
        assert body["total_spend"] == 5.5

    @patch("litellm.proxy.proxy_server.prisma_client", None)
    def test_tool_spend_no_db_returns_500(self):
        resp = self.client.get("/v1/tool/spend")
        assert resp.status_code == 500

    def test_tool_spend_end_date_is_inclusive_via_exclusive_next_day_bound(self):
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(return_value=[])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get("/v1/tool/spend?start_date=2026-07-01&end_date=2026-07-02")
        assert resp.status_code == 200
        expected_binds = (
            datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat(),
            datetime(2026, 7, 3, tzinfo=timezone.utc).isoformat(),
        )
        assert prisma.db.query_raw.await_count == 2
        for call in prisma.db.query_raw.await_args_list:
            assert tuple(call.args[1:]) == expected_binds
        assert resp.json()["end_date"] == "2026-07-02"

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
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(return_value=[])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = self.client.get(f"/v1/tool/spend?{query}")
        assert resp.status_code == 400
        assert "Invalid date format" in resp.json()["detail"]
        prisma.db.query_raw.assert_not_awaited()

    def test_tool_spend_non_admin_returns_403(self):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="sk-user", user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER
        )
        client = TestClient(app, raise_server_exceptions=True)
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(return_value=[])
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            resp = client.get("/v1/tool/spend")
        assert resp.status_code == 403
        prisma.db.query_raw.assert_not_awaited()


def _spend_row(date: str, tool_name: str, spend: float, call_count: int = 1, total_tokens: int = 10) -> _ToolSpendRow:
    return _ToolSpendRow(date=date, tool_name=tool_name, call_count=call_count, spend=spend, total_tokens=total_tokens)


class TestBuildToolSpendResponse:
    def test_multi_tool_attribution_double_counts_per_tool_but_not_total(self):
        rows = [
            _spend_row("2026-07-01", "a", spend=3.0),
            _spend_row("2026-07-01", "b", spend=3.0),
        ]
        resp = _build_tool_spend_response(rows, total_spend=3.0, start_date="2026-07-01", end_date="2026-07-01")
        by_tool = {t.tool_name: t.spend for t in resp.by_tool}
        assert by_tool == {"a": 3.0, "b": 3.0}
        assert resp.total_spend == 3.0

    def test_groups_across_days_and_sorts_by_spend(self):
        rows = [
            _spend_row("2026-07-01", "b", spend=1.0, call_count=2, total_tokens=100),
            _spend_row("2026-07-02", "b", spend=4.0, call_count=1, total_tokens=50),
            _spend_row("2026-07-01", "a", spend=2.0, call_count=3, total_tokens=300),
        ]
        resp = _build_tool_spend_response(rows, total_spend=7.0, start_date="2026-07-01", end_date="2026-07-02")
        assert [(t.tool_name, t.spend, t.call_count, t.total_tokens) for t in resp.by_tool] == [
            ("b", 5.0, 3, 150),
            ("a", 2.0, 3, 300),
        ]
        assert len(resp.daily) == 3

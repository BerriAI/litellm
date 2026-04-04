"""
Tests for MCP usage/operational visibility endpoints.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.mcp_usage_endpoints import (
    MCPUsageLogsResponse,
    MCPUsageOverviewResponse,
    MCPToolUsersResponse,
    MCPAlertRulesListResponse,
    _build_index_where,
    _snippet,
)

MOCK_ADMIN_USER = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


class TestBuildIndexWhere:
    """Test _build_index_where helper."""

    def test_should_return_empty_dict_for_no_filters(self):
        result = _build_index_where(None, None, None, None)
        assert result == {}

    def test_should_filter_by_server_name(self):
        result = _build_index_where("test-server", None, None, None)
        assert result["mcp_server_name"] == "test-server"

    def test_should_filter_by_tool_name(self):
        result = _build_index_where(None, "delete_item", None, None)
        assert result["tool_name"] == "delete_item"

    def test_should_filter_by_date_range(self):
        result = _build_index_where(None, None, "2024-01-01", "2024-01-31")
        assert "start_time" in result
        assert "gte" in result["start_time"]
        assert "lte" in result["start_time"]


class TestSnippet:
    """Test _snippet helper."""

    def test_should_return_none_for_none_input(self):
        assert _snippet(None) is None

    def test_should_truncate_long_strings(self):
        long_text = "x" * 300
        result = _snippet(long_text)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_should_return_none_for_empty_dict_string(self):
        assert _snippet("{}") is None

    def test_should_handle_list_of_messages(self):
        msgs = [{"content": "hello"}, {"content": "world"}]
        result = _snippet(msgs)
        assert result == "hello world"


class TestMCPUsageLogs:
    """Test mcp_usage_logs endpoint."""

    @pytest.mark.asyncio
    async def test_should_return_empty_when_no_prisma(self):
        from litellm.proxy.management_endpoints.mcp_usage_endpoints import (
            mcp_usage_logs,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await mcp_usage_logs(
                mcp_server_name="test",
                tool_name=None,
                page=1,
                page_size=50,
                start_date=None,
                end_date=None,
                user_api_key_dict=MOCK_ADMIN_USER,
            )
            assert result.total == 0
            assert result.logs == []


class TestMCPUsageOverview:
    """Test mcp_usage_overview endpoint."""

    @pytest.mark.asyncio
    async def test_should_return_empty_when_no_prisma(self):
        from litellm.proxy.management_endpoints.mcp_usage_endpoints import (
            mcp_usage_overview,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await mcp_usage_overview(
                start_date=None,
                end_date=None,
                user_api_key_dict=MOCK_ADMIN_USER,
            )
            assert result.total_requests == 0
            assert result.servers == []


class TestMCPUsageTools:
    """Test mcp_usage_tools endpoint."""

    @pytest.mark.asyncio
    async def test_should_return_empty_when_no_prisma(self):
        from litellm.proxy.management_endpoints.mcp_usage_endpoints import (
            mcp_usage_tools,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await mcp_usage_tools(
                mcp_server_name="test",
                start_date=None,
                end_date=None,
                user_api_key_dict=MOCK_ADMIN_USER,
            )
            assert result.total == 0
            assert result.entries == []


class TestMCPAlertRules:
    """Test alert rule CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_should_return_empty_rules_when_no_prisma(self):
        from litellm.proxy.management_endpoints.mcp_usage_endpoints import (
            list_mcp_alert_rules,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await list_mcp_alert_rules(
                mcp_server_name=None,
                user_api_key_dict=MOCK_ADMIN_USER,
            )
            assert result.total == 0
            assert result.rules == []

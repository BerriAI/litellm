"""
Tests for MCP alert rules (check_and_fire_mcp_alerts).
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.mcp_alert_rules import check_and_fire_mcp_alerts


def _make_rule(
    alert_name="test-alert",
    tool_name_pattern="*delete*",
    mcp_server_name=None,
    webhook_url="https://hooks.example.com/test",
    enabled=True,
    description=None,
):
    rule = MagicMock()
    rule.id = "rule-1"
    rule.alert_name = alert_name
    rule.tool_name_pattern = tool_name_pattern
    rule.mcp_server_name = mcp_server_name
    rule.webhook_url = webhook_url
    rule.enabled = enabled
    rule.description = description
    return rule


class TestCheckAndFireMCPAlerts:
    """Test check_and_fire_mcp_alerts."""

    @pytest.mark.asyncio
    async def test_should_skip_when_no_tool_name(self):
        mock_prisma = MagicMock()
        await check_and_fire_mcp_alerts(
            prisma_client=mock_prisma,
            mcp_server_name="test-server",
            tool_name=None,
            request_id="req-1",
            user_id=None,
            api_key_hash=None,
            team_id=None,
        )

    @pytest.mark.asyncio
    async def test_should_fire_webhook_when_pattern_matches(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_mcpalertrule = MagicMock()
        rule = _make_rule(tool_name_pattern="*delete*")
        mock_prisma.db.litellm_mcpalertrule.find_many = AsyncMock(
            return_value=[rule]
        )

        with patch("litellm.proxy.db.mcp_alert_rules.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=None)

            await check_and_fire_mcp_alerts(
                prisma_client=mock_prisma,
                mcp_server_name="test-server",
                tool_name="delete_item",
                request_id="req-1",
                user_id="user-1",
                api_key_hash="sk-hash",
                team_id="team-1",
            )

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            assert call_kwargs[0][0] == "https://hooks.example.com/test"
            payload = call_kwargs[1]["json"]
            assert payload["tool_name"] == "delete_item"
            assert payload["mcp_server_name"] == "test-server"

    @pytest.mark.asyncio
    async def test_should_not_fire_when_pattern_does_not_match(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_mcpalertrule = MagicMock()
        rule = _make_rule(tool_name_pattern="*delete*")
        mock_prisma.db.litellm_mcpalertrule.find_many = AsyncMock(
            return_value=[rule]
        )

        with patch("litellm.proxy.db.mcp_alert_rules.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=None)

            await check_and_fire_mcp_alerts(
                prisma_client=mock_prisma,
                mcp_server_name="test-server",
                tool_name="get_page_content",
                request_id="req-1",
                user_id="user-1",
                api_key_hash="sk-hash",
                team_id="team-1",
            )

            mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_filter_by_server_name(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_mcpalertrule = MagicMock()
        rule = _make_rule(
            tool_name_pattern="*delete*", mcp_server_name="specific-server"
        )
        mock_prisma.db.litellm_mcpalertrule.find_many = AsyncMock(
            return_value=[rule]
        )

        with patch("litellm.proxy.db.mcp_alert_rules.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=None)

            await check_and_fire_mcp_alerts(
                prisma_client=mock_prisma,
                mcp_server_name="other-server",
                tool_name="delete_item",
                request_id="req-1",
                user_id=None,
                api_key_hash=None,
                team_id=None,
            )

            mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_handle_db_failure_gracefully(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_mcpalertrule = MagicMock()
        mock_prisma.db.litellm_mcpalertrule.find_many = AsyncMock(
            side_effect=Exception("DB error")
        )

        # Should not raise
        await check_and_fire_mcp_alerts(
            prisma_client=mock_prisma,
            mcp_server_name="test-server",
            tool_name="delete_item",
            request_id="req-1",
            user_id=None,
            api_key_hash=None,
            team_id=None,
        )

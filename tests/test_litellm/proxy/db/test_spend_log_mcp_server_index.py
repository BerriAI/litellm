"""
Tests for MCP server usage tracking (SpendLogMCPServerIndex).
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.spend_log_mcp_server_index import (
    _parse_mcp_info_from_payload,
    process_spend_logs_mcp_server_usage,
)


class TestParseMCPInfoFromPayload:
    """Test _parse_mcp_info_from_payload helper."""

    def test_should_return_none_for_empty_payload(self):
        result = _parse_mcp_info_from_payload({})
        assert result is None

    def test_should_return_none_for_no_mcp_metadata(self):
        payload = {
            "metadata": {"some_key": "some_value"}
        }
        result = _parse_mcp_info_from_payload(payload)
        assert result is None

    def test_should_extract_mcp_server_name_from_metadata(self):
        payload = {
            "metadata": {
                "mcp_tool_call_metadata": {
                    "name": "get_page_content",
                    "mcp_server_name": "deepwiki-mcp",
                    "namespaced_tool_name": "deepwiki-mcp/get_page_content",
                }
            }
        }
        result = _parse_mcp_info_from_payload(payload)
        assert result is not None
        assert result["mcp_server_name"] == "deepwiki-mcp"
        assert result["tool_name"] == "get_page_content"

    def test_should_extract_server_name_from_namespaced_tool_name(self):
        payload = {
            "mcp_namespaced_tool_name": "github-mcp/create_issue",
            "metadata": {
                "mcp_tool_call_metadata": {
                    "name": "create_issue",
                    "namespaced_tool_name": "github-mcp/create_issue",
                }
            },
        }
        result = _parse_mcp_info_from_payload(payload)
        assert result is not None
        assert result["mcp_server_name"] == "github-mcp"
        assert result["tool_name"] == "create_issue"

    def test_should_handle_string_metadata(self):
        import json

        payload = {
            "metadata": json.dumps(
                {
                    "mcp_tool_call_metadata": {
                        "name": "search",
                        "mcp_server_name": "search-server",
                    }
                }
            )
        }
        result = _parse_mcp_info_from_payload(payload)
        assert result is not None
        assert result["mcp_server_name"] == "search-server"

    def test_should_return_none_for_invalid_json_metadata(self):
        payload = {"metadata": "not valid json"}
        result = _parse_mcp_info_from_payload(payload)
        assert result is None

    def test_should_return_none_when_no_server_name_derivable(self):
        payload = {
            "metadata": {
                "mcp_tool_call_metadata": {
                    "name": "some_tool",
                }
            }
        }
        result = _parse_mcp_info_from_payload(payload)
        assert result is None


class TestProcessSpendLogsMCPServerUsage:
    """Test process_spend_logs_mcp_server_usage."""

    @pytest.mark.asyncio
    async def test_should_skip_empty_logs(self):
        mock_prisma = MagicMock()
        await process_spend_logs_mcp_server_usage(mock_prisma, [])

    @pytest.mark.asyncio
    async def test_should_insert_index_rows(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex.create_many = AsyncMock(
            return_value=None
        )

        logs = [
            {
                "request_id": "req-123",
                "startTime": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "metadata": {
                    "mcp_tool_call_metadata": {
                        "name": "get_page_content",
                        "mcp_server_name": "deepwiki-mcp",
                    },
                    "user_api_key_hash": "sk-hash-123",
                    "user_api_key_user_id": "user-1",
                    "user_api_key_team_id": "team-1",
                },
                "api_key": "sk-hash-123",
                "user": "user-1",
                "team_id": "team-1",
            }
        ]

        with patch(
            "litellm.proxy.db.mcp_alert_rules.check_and_fire_mcp_alerts",
            new_callable=AsyncMock,
        ):
            await process_spend_logs_mcp_server_usage(mock_prisma, logs)

        mock_prisma.db.litellm_spendlogmcpserverindex.create_many.assert_called_once()
        call_args = (
            mock_prisma.db.litellm_spendlogmcpserverindex.create_many.call_args
        )
        data = call_args.kwargs.get("data") or call_args[1].get("data")
        assert len(data) == 1
        assert data[0]["request_id"] == "req-123"
        assert data[0]["mcp_server_name"] == "deepwiki-mcp"
        assert data[0]["tool_name"] == "get_page_content"
        assert data[0]["api_key_hash"] == "sk-hash-123"
        assert data[0]["user_id"] == "user-1"
        assert data[0]["team_id"] == "team-1"

    @pytest.mark.asyncio
    async def test_should_skip_non_mcp_logs(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex.create_many = AsyncMock()

        logs = [
            {
                "request_id": "req-456",
                "startTime": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "metadata": {"some_key": "some_value"},
            }
        ]

        await process_spend_logs_mcp_server_usage(mock_prisma, logs)
        mock_prisma.db.litellm_spendlogmcpserverindex.create_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_handle_create_many_failure_gracefully(self):
        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex = MagicMock()
        mock_prisma.db.litellm_spendlogmcpserverindex.create_many = AsyncMock(
            side_effect=Exception("DB error")
        )

        logs = [
            {
                "request_id": "req-789",
                "startTime": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "metadata": {
                    "mcp_tool_call_metadata": {
                        "name": "some_tool",
                        "mcp_server_name": "test-server",
                    }
                },
            }
        ]

        # Should not raise
        await process_spend_logs_mcp_server_usage(mock_prisma, logs)

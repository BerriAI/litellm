import json
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import Request
from fastapi.testclient import TestClient


from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.cost_calculator import MCPCostCalculator
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestMCPCostCalculator:
    def test_calculate_mcp_tool_call_cost_none_logging_obj(self):
        """Test that when litellm_logging_obj is None, it returns 0.0"""
        result = MCPCostCalculator.calculate_mcp_tool_call_cost(None)
        assert result == 0.0

    def test_calculate_mcp_tool_call_cost_with_tool_specific_cost(self):
        """Test that when a specific tool has a defined cost, it returns that cost"""
        # Mock the litellm_logging_obj
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {
                "name": "search_web",
                "mcp_server_cost_info": {
                    "default_cost_per_query": 0.01,
                    "tool_name_to_cost_per_query": {
                        "search_web": 0.05,
                        "generate_code": 0.03,
                    },
                },
            }
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.05

    def test_calculate_mcp_tool_call_cost_with_default_cost(self):
        """Test that when no tool-specific cost is found, it falls back to default cost"""
        # Mock the litellm_logging_obj
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {
                "name": "unknown_tool",
                "mcp_server_cost_info": {
                    "default_cost_per_query": 0.02,
                    "tool_name_to_cost_per_query": {"search_web": 0.05},
                },
            }
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.02

    def test_calculate_mcp_tool_call_cost_no_cost_configuration(self):
        """Test that when no cost configuration is provided, it returns 0.0"""
        # Mock the litellm_logging_obj with minimal metadata
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {"name": "some_tool", "mcp_server_cost_info": {}}
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.0

    def test_calculate_mcp_tool_call_cost_empty_metadata(self):
        """Test that when metadata is empty or missing, it returns 0.0"""
        # Mock the litellm_logging_obj with empty model_call_details
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.0


def _server(cost_info: object = None) -> MCPServer:
    mcp_info = None if cost_info is None else {"mcp_server_cost_info": cost_info}
    return MCPServer(server_id="s1", name="s1", transport=MCPTransport.http, mcp_info=mcp_info)


class _PricingHook(CustomLogger):
    async def async_post_mcp_tool_call_hook(self, kwargs, response_obj, start_time, end_time):
        response_obj.hidden_params.response_cost = 0.5
        return response_obj


@pytest.mark.parametrize(
    ("servers", "callbacks"),
    [
        pytest.param([], [], id="nothing configured"),
        pytest.param([_server()], [], id="server without cost info"),
        pytest.param(
            [_server({"default_cost_per_query": 0.0, "tool_name_to_cost_per_query": {"search": 0.0}})],
            [],
            id="prices all set to zero",
        ),
        pytest.param([], [CustomLogger()], id="callback without the post-call hook"),
        pytest.param([], ["langfuse"], id="callback registered by name"),
    ],
)
def test_tool_calls_cannot_cost_until_something_prices_them(servers, callbacks):
    assert MCPCostCalculator.tool_calls_may_cost(servers=servers, callbacks=callbacks) is False


@pytest.mark.parametrize(
    ("servers", "callbacks"),
    [
        pytest.param([_server({"default_cost_per_query": 0.01})], [], id="server default price"),
        pytest.param([_server({"tool_name_to_cost_per_query": {"search": 0.05}})], [], id="one tool priced"),
        pytest.param([_server(), _server({"default_cost_per_query": 0.01})], [], id="any one server priced"),
        pytest.param([_server({"default_cost_per_query": "0.01"})], [], id="unparsed price counts as priced"),
        pytest.param([_server({"tool_name_to_cost_per_query": ["search"]})], [], id="malformed tool prices"),
        pytest.param([_server("0.01")], [], id="malformed cost block"),
        pytest.param([], [_PricingHook()], id="callback that can set the cost"),
    ],
)
def test_tool_calls_may_cost_once_anything_can_price_them(servers, callbacks):
    assert MCPCostCalculator.tool_calls_may_cost(servers=servers, callbacks=callbacks) is True

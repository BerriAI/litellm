import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.cost_calculator import MCPCostCalculator


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
                        "generate_code": 0.03
                    }
                }
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
                    "tool_name_to_cost_per_query": {
                        "search_web": 0.05
                    }
                }
            }
        }
        
        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.02

    def test_calculate_mcp_tool_call_cost_no_cost_configuration(self):
        """Test that when no cost configuration is provided, it returns 0.0"""
        # Mock the litellm_logging_obj with minimal metadata
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {
                "name": "some_tool",
                "mcp_server_cost_info": {}
            }
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


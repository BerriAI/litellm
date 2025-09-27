"""
Cost calculator for MCP tools.
"""

from typing import TYPE_CHECKING, Any, Optional, cast

from litellm.types.mcp import MCPServerCostInfo
from litellm.types.utils import StandardLoggingMCPToolCall

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


class MCPCostCalculator:
    @staticmethod
    def calculate_mcp_tool_call_cost(
        litellm_logging_obj: Optional[LitellmLoggingObject],
    ) -> float:
        """
        Calculate the cost of an MCP tool call.

        Default is 0.0, unless user specifies a custom cost per request for MCP tools.
        """
        if litellm_logging_obj is None:
            return 0.0

        #########################################################
        # Get the response cost from logging object model_call_details
        # This is set when a user modifies the response in a post_mcp_tool_call_hook
        #########################################################
        response_cost = litellm_logging_obj.model_call_details.get(
            "response_cost", None
        )
        if response_cost is not None:
            return response_cost

        #########################################################
        # Unpack the mcp_tool_call_metadata
        #########################################################
        mcp_tool_call_metadata: StandardLoggingMCPToolCall = (
            cast(
                StandardLoggingMCPToolCall,
                litellm_logging_obj.model_call_details.get(
                    "mcp_tool_call_metadata", {}
                ),
            )
            or {}
        )
        mcp_server_cost_info: MCPServerCostInfo = (
            mcp_tool_call_metadata.get("mcp_server_cost_info") or MCPServerCostInfo()
        )
        #########################################################
        # User defined cost per query
        #########################################################
        default_cost_per_query = mcp_server_cost_info.get(
            "default_cost_per_query", None
        )
        tool_name_to_cost_per_query: dict = (
            mcp_server_cost_info.get("tool_name_to_cost_per_query", {}) or {}
        )
        tool_name = mcp_tool_call_metadata.get("name", "")

        #########################################################
        # 1. If tool_name is in tool_name_to_cost_per_query, use the cost per query
        # 2. If tool_name is not in tool_name_to_cost_per_query, use the default cost per query
        # 3. Default to 0.0 if no cost per query is found
        #########################################################
        cost_per_query: float = 0.0
        if tool_name in tool_name_to_cost_per_query:
            cost_per_query = tool_name_to_cost_per_query[tool_name]
        elif default_cost_per_query is not None:
            cost_per_query = default_cost_per_query
        return cost_per_query

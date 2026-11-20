"""
Cost calculator for MCP tools.
"""

from typing import TYPE_CHECKING, Any, Final, cast

from litellm.types.mcp import MCPServerCostInfo
from litellm.types.utils import StandardLoggingMCPToolCall

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


def mcp_tool_cost_per_query(cost_info: MCPServerCostInfo | None, tool_name: str) -> float:
    resolved_cost_info: Final = cost_info or MCPServerCostInfo()
    tool_name_to_cost_per_query: Final = resolved_cost_info.get("tool_name_to_cost_per_query") or {}
    if tool_name in tool_name_to_cost_per_query:
        return tool_name_to_cost_per_query[tool_name]
    return resolved_cost_info.get("default_cost_per_query") or 0.0


def mcp_server_charges_for_tool_calls(cost_info: MCPServerCostInfo | None) -> bool:
    resolved_cost_info: Final = cost_info or MCPServerCostInfo()
    return any(
        (cost or 0.0) > 0
        for cost in (
            resolved_cost_info.get("default_cost_per_query"),
            *(resolved_cost_info.get("tool_name_to_cost_per_query") or {}).values(),
        )
    )


class MCPCostCalculator:
    @staticmethod
    def calculate_mcp_tool_call_cost(
        litellm_logging_obj: LitellmLoggingObject | None,
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
        response_cost: Final = litellm_logging_obj.model_call_details.get("response_cost", None)
        if response_cost is not None:
            return response_cost

        #########################################################
        # Unpack the mcp_tool_call_metadata
        #########################################################
        mcp_tool_call_metadata: Final[StandardLoggingMCPToolCall] = (
            cast(
                StandardLoggingMCPToolCall,
                litellm_logging_obj.model_call_details.get("mcp_tool_call_metadata", {}),
            )
            or {}
        )
        return mcp_tool_cost_per_query(
            cost_info=mcp_tool_call_metadata.get("mcp_server_cost_info"),
            tool_name=mcp_tool_call_metadata.get("name", ""),
        )

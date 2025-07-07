"""
Cost calculator for MCP tools.
"""
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any

class MCPCostCalculator:
    @staticmethod
    def calculate_mcp_tool_call_cost(litellm_logging_obj: Optional[LitellmLoggingObject]) -> float:
        """
        Calculate the cost of an MCP tool call.

        Default is 0.0, unless user specifies a custom cost per request for MCP tools.
        """
        import json
        if litellm_logging_obj is None:
            return 0.0
        standard_logged_mcp_tool_call = litellm_logging_obj.model_call_details.get("mcp_tool_call_metadata")
        #print("json.dumps(standard_logged_mcp_tool_call, indent=4)", json.dumps(standard_logged_mcp_tool_call, indent=4))
        return 0.0
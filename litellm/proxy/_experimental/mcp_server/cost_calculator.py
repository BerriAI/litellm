"""
Cost calculator for MCP tools.
"""
class MCPCostCalculator:
    @staticmethod
    def calculate_mcp_tool_call_cost(model: str) -> float:
        """
        Calculate the cost of an MCP tool call.

        Default is 0.0, unless user specifies a custom cost per request for MCP tools.
        """
        return 0.0
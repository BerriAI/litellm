"""
Cost calculator for MCP tools.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.mcp import MCPServerCostInfo
from litellm.types.mcp_server.mcp_server_manager import MCPServer
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
        mcp_server_cost_info: Final[MCPServerCostInfo] = (
            mcp_tool_call_metadata.get("mcp_server_cost_info") or MCPServerCostInfo()
        )
        #########################################################
        # User defined cost per query
        #########################################################
        default_cost_per_query: Final = mcp_server_cost_info.get("default_cost_per_query", None)
        tool_name_to_cost_per_query: Final[dict] = mcp_server_cost_info.get("tool_name_to_cost_per_query", {}) or {}
        tool_name: Final = mcp_tool_call_metadata.get("name", "")

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

    @staticmethod
    def tool_calls_may_cost(servers: Iterable[MCPServer], callbacks: Iterable[object]) -> bool:
        """Whether any MCP tool call could be charged: a server priced above zero, or a
        callback overriding the post-call hook, which can set the cost of any call."""
        return any(_may_price_mcp_calls(callback) for callback in callbacks) or any(
            _is_priced(server) for server in servers
        )


_SERVER_COST_INFO: Final = TypeAdapter(MCPServerCostInfo)
_POST_CALL_HOOK: Final = "async_post_mcp_tool_call_hook"


def _is_priced(server: MCPServer) -> bool:
    raw_cost_info: Final = None if server.mcp_info is None else server.mcp_info.get("mcp_server_cost_info")
    if raw_cost_info is None:
        return False
    try:
        cost_info: Final = _SERVER_COST_INFO.validate_python(raw_cost_info, strict=True)
    except ValidationError:
        return True
    tool_costs: Final = cost_info.get("tool_name_to_cost_per_query")
    tool_prices: Final = () if tool_costs is None else tuple(tool_costs.values())
    return any(price is not None and price > 0 for price in (cost_info.get("default_cost_per_query"), *tool_prices))


def _may_price_mcp_calls(callback: object) -> bool:
    if not isinstance(callback, CustomLogger):
        return False
    mro: Final = type(callback).__mro__
    return any(_POST_CALL_HOOK in vars(cls) for cls in mro[: mro.index(CustomLogger)])

"""
MCP Server Utilities
"""
from typing import Tuple

import importlib

# Constants
LITELLM_MCP_SERVER_NAME = "litellm-mcp-server"
LITELLM_MCP_SERVER_VERSION = "1.0.0"
LITELLM_MCP_SERVER_DESCRIPTION = "MCP Server for LiteLLM"
MCP_TOOL_PREFIX_SEPARATOR = "/"
MCP_TOOL_PREFIX_FORMAT = "{server_name}{separator}{tool_name}"

def is_mcp_available() -> bool:
    """
    Returns True if the MCP module is available, False otherwise
    """
    try:
        importlib.import_module("mcp")
        return True
    except ImportError:
        return False

def normalize_server_name(server_name: str) -> str:
    """
    Normalize server name by replacing spaces with underscores
    """
    return server_name.replace(" ", "_")

def add_server_prefix_to_tool_name(tool_name: str, server_name: str) -> str:
    """
    Add server name prefix to tool name

    Args:
        tool_name: Original tool name
        server_name: MCP server name

    Returns:
        Prefixed tool name in format: server_name::tool_name
    """
    formatted_server_name = normalize_server_name(server_name)

    return MCP_TOOL_PREFIX_FORMAT.format(
        server_name=formatted_server_name,
        separator=MCP_TOOL_PREFIX_SEPARATOR,
        tool_name=tool_name
    )

def get_server_name_prefix_tool_mcp(prefixed_tool_name: str) -> Tuple[str, str]:
    """
    Remove server name prefix from tool name

    Args:
        prefixed_tool_name: Tool name with server prefix

    Returns:
        Tuple of (original_tool_name, server_name)
    """
    if MCP_TOOL_PREFIX_SEPARATOR in prefixed_tool_name:
        parts = prefixed_tool_name.split(MCP_TOOL_PREFIX_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[1], parts[0]  # tool_name, server_name
    return prefixed_tool_name, ""  # No prefix found, return original name

def is_tool_name_prefixed(tool_name: str) -> bool:
    """
    Check if tool name has server prefix

    Args:
        tool_name: Tool name to check

    Returns:
        True if tool name is prefixed, False otherwise
    """
    return MCP_TOOL_PREFIX_SEPARATOR in tool_name

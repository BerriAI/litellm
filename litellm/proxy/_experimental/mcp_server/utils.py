"""
MCP Server Utilities
"""
from typing import Tuple, Any

import os
import importlib

# Constants
LITELLM_MCP_SERVER_NAME = "litellm-mcp-server"
LITELLM_MCP_SERVER_VERSION = "1.0.0"
LITELLM_MCP_SERVER_DESCRIPTION = "MCP Server for LiteLLM"
MCP_TOOL_PREFIX_SEPARATOR = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR", "-")
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


def validate_and_normalize_mcp_server_payload(payload: Any) -> None:
    """
    Validate and normalize MCP server payload fields (server_name and alias).

    This function:
    1. Validates that server_name and alias don't contain the MCP_TOOL_PREFIX_SEPARATOR
    2. Normalizes alias by replacing spaces with underscores
    3. Sets default alias if not provided (using server_name as base)

    Args:
        payload: The payload object containing server_name and alias fields

    Raises:
        HTTPException: If validation fails
    """
    # Server name validation: disallow '-'
    if hasattr(payload, "server_name") and payload.server_name:
        validate_mcp_server_name(payload.server_name, raise_http_exception=True)

    # Alias validation: disallow '-'
    if hasattr(payload, "alias") and payload.alias:
        validate_mcp_server_name(payload.alias, raise_http_exception=True)

    # Alias normalization and defaulting
    alias = getattr(payload, "alias", None)
    server_name = getattr(payload, "server_name", None)

    if not alias and server_name:
        alias = normalize_server_name(server_name)
    elif alias:
        alias = normalize_server_name(alias)

    # Update the payload with normalized alias
    if hasattr(payload, "alias"):
        payload.alias = alias


def add_server_prefix_to_name(name: str, server_name: str) -> str:
    """Add server name prefix to any MCP resource name."""
    formatted_server_name = normalize_server_name(server_name)

    return MCP_TOOL_PREFIX_FORMAT.format(
        server_name=formatted_server_name,
        separator=MCP_TOOL_PREFIX_SEPARATOR,
        tool_name=name,
    )


def get_server_prefix(server: Any) -> str:
    """Return the prefix for a server: alias if present, else server_name, else server_id"""
    if hasattr(server, "alias") and server.alias:
        return server.alias
    if hasattr(server, "server_name") and server.server_name:
        return server.server_name
    if hasattr(server, "server_id"):
        return server.server_id
    return ""


def split_server_prefix_from_name(prefixed_name: str) -> Tuple[str, str]:
    """Return the unprefixed name plus the server name used as prefix."""
    if MCP_TOOL_PREFIX_SEPARATOR in prefixed_name:
        parts = prefixed_name.split(MCP_TOOL_PREFIX_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[1], parts[0]
    return prefixed_name, ""


def is_tool_name_prefixed(tool_name: str) -> bool:
    """
    Check if tool name has server prefix

    Args:
        tool_name: Tool name to check

    Returns:
        True if tool name is prefixed, False otherwise
    """
    return MCP_TOOL_PREFIX_SEPARATOR in tool_name


def validate_mcp_server_name(
    server_name: str, raise_http_exception: bool = False
) -> None:
    """
    Validate that MCP server name does not contain 'MCP_TOOL_PREFIX_SEPARATOR'.

    Args:
        server_name: The server name to validate
        raise_http_exception: If True, raises HTTPException instead of generic Exception

    Raises:
        Exception or HTTPException: If server name contains 'MCP_TOOL_PREFIX_SEPARATOR'
    """
    if server_name and MCP_TOOL_PREFIX_SEPARATOR in server_name:
        error_message = f"Server name cannot contain '{MCP_TOOL_PREFIX_SEPARATOR}'. Use an alternative character instead Found: {server_name}"
        if raise_http_exception:
            from fastapi import HTTPException
            from starlette import status

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail={"error": error_message}
            )
        else:
            raise Exception(error_message)

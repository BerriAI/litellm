"""Import-safe pre-check for the LiteLLM MCP gateway.

``litellm_proxy_mcp_handler`` sits on the proxy layer and needs the optional ``mcp`` package. The SDK entry
points must therefore not import it for tools that cannot name the gateway, or a base install without proxy
extras fails on any call that passes ordinary tools. This module imports only the standard library for that
reason.
"""

from collections.abc import Iterable


def _is_mcp_tool(tool: object) -> bool:
    match tool:
        case {"type": "mcp"}:
            return True
        case _:
            return False


def may_name_litellm_mcp_gateway(tools: Iterable[object] | None) -> bool:
    """False only when no tool can name the gateway, so the handler import can be skipped.

    ``LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway`` reads ``server_url`` from dict tools of type
    ``mcp`` and from nothing else, so this is a necessary condition for it: it never returns False for a tool
    list the handler would accept.
    """
    return any(_is_mcp_tool(tool) for tool in tools or ())

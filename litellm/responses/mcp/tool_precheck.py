"""Decide whether request tools name the LiteLLM MCP gateway, using the standard library alone.

``litellm_proxy_mcp_handler`` sits on the proxy layer and needs the optional ``mcp`` package. The SDK entry
points ask this module first, so a base install without proxy extras never imports the handler for tools
that do not name the gateway: function tools, provider-hosted tools, or an MCP server the provider calls itself.
"""

import re
from collections.abc import Iterable
from typing import Final

LITELLM_PROXY_MCP_SERVER_URL: Final = "litellm_proxy"
LITELLM_PROXY_MCP_SERVER_URL_PREFIX: Final = f"{LITELLM_PROXY_MCP_SERVER_URL}/mcp/"

_PROXY_MCP_PATH_RE: Final = re.compile(r"^https?://.+/mcp/([^/]+)$")


def _mcp_server_url(tool: object) -> str | None:
    match tool:
        case {"type": "mcp", "server_url": str() as server_url} if isinstance(tool, dict):
            return server_url
        case _:
            return None


def names_gateway_explicitly(tool: object) -> bool:
    return (_mcp_server_url(tool) or "").startswith(LITELLM_PROXY_MCP_SERVER_URL)


def proxy_path_mcp_name(tool: object) -> str | None:
    server_url: Final = _mcp_server_url(tool)
    match: Final = None if server_url is None else _PROXY_MCP_PATH_RE.match(server_url)
    return None if match is None else match.group(1)


def may_name_litellm_mcp_gateway(tools: Iterable[object] | None) -> bool:
    """True when a tool may name this gateway: server_url "litellm_proxy..." or an http(s) URL ending in
    /mcp/<name>. The handler's ``split_mcp_tools`` then settles which of the latter the gateway actually serves."""
    return any(names_gateway_explicitly(tool) or proxy_path_mcp_name(tool) is not None for tool in tools or ())

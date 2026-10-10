import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.litellm_proxy_mcp_handler import (
        LITELLM_PROXY_MCP_SERVER_URL_PREFIX,
        LiteLLM_Proxy_MCP_Handler,
        MCPToolResult,
    )

__all__ = ("LITELLM_PROXY_MCP_SERVER_URL_PREFIX", "LiteLLM_Proxy_MCP_Handler", "MCPToolResult")

_RELOCATED_MODULE: Final = "litellm.proxy._experimental.mcp_server.litellm_proxy_mcp_handler"


def __getattr__(name: str) -> object:
    return getattr(importlib.import_module(_RELOCATED_MODULE), name)

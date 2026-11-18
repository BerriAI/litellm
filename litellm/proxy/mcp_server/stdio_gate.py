import os
from typing import Final

from litellm._logging import verbose_logger
from litellm.types.mcp import MCPTransport

MCP_STDIO_ENABLED_ENV_VAR: Final = "LITELLM_ENABLE_MCP_STDIO"
MCP_STDIO_DISABLED_MESSAGE: Final = (
    f"stdio MCP servers are disabled on this proxy. "
    f"Set {MCP_STDIO_ENABLED_ENV_VAR}=true on the proxy and restart to enable them"
)


def is_mcp_stdio_enabled() -> bool:
    return os.getenv(MCP_STDIO_ENABLED_ENV_VAR, "").strip().lower() == "true"


def is_mcp_stdio_flag_key(env_var_name: str) -> bool:
    return env_var_name.upper() == MCP_STDIO_ENABLED_ENV_VAR


def is_mcp_stdio_blocked(transport: str | None) -> bool:
    return transport == MCPTransport.stdio and not is_mcp_stdio_enabled()


def warn_if_mcp_stdio_blocked(server_name: str | None, transport: str | None) -> None:
    if is_mcp_stdio_blocked(transport):
        verbose_logger.warning("MCP server '%s' will not start: %s", server_name, MCP_STDIO_DISABLED_MESSAGE)

from typing import Final

_MCP_TOOLS_EXECUTED_ATTR: Final = "_litellm_mcp_tools_executed"


def mark_mcp_tools_executed(exception: BaseException) -> None:
    """Tag an error raised after an MCP gateway loop already executed tool calls.

    Retrying or falling back re-runs the whole loop, which executes those tool calls
    again, so the router surfaces a tagged error instead of replaying the request. The
    exception is tagged rather than wrapped so the client sees the same error.
    """
    setattr(exception, _MCP_TOOLS_EXECUTED_ATTR, True)


def mcp_tools_executed(exception: BaseException | None) -> bool:
    """Whether ``exception`` was tagged by ``mark_mcp_tools_executed``."""
    return getattr(exception, _MCP_TOOLS_EXECUTED_ATTR, False) is True

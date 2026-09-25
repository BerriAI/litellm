from types import TracebackType
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
    """Whether ``exception`` or an exception in its chain was tagged by ``mark_mcp_tools_executed``.

    The chain matters because a caller can re-raise a tagged error as a new exception, as
    ``exception_type`` does for anything that is not already a LiteLLM exception.
    """
    return exception is not None and (
        getattr(exception, _MCP_TOOLS_EXECUTED_ATTR, False) is True
        or mcp_tools_executed(exception.__cause__)
        or mcp_tools_executed(exception.__context__)
    )


class MCPToolReplayGuard:
    """Tags errors raised inside ``with guard:`` once the request has executed an MCP tool call."""

    __slots__ = ("_tool_executed",)

    def __init__(self) -> None:
        self._tool_executed: bool = False

    def record_tool_execution(self) -> None:
        self._tool_executed = True

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._tool_executed and isinstance(exc, Exception):
            mark_mcp_tools_executed(exc)

from collections.abc import Iterator
from types import TracebackType
from typing import Final

_MCP_TOOLS_EXECUTED_ATTR: Final = "_litellm_mcp_tools_executed"

# Real exception chains are a few links deep; the cap also makes the walk cycle-safe.
_MAX_EXCEPTION_CHAIN_DEPTH: Final = 20


def mark_mcp_tools_executed(exception: BaseException) -> None:
    """Tag an error raised after an MCP gateway loop already executed tool calls.

    Retrying or falling back re-runs the whole loop, which executes those tool calls
    again, so the router surfaces a tagged error instead of replaying the request. The
    exception is tagged rather than wrapped so the client sees the same error.
    """
    setattr(exception, _MCP_TOOLS_EXECUTED_ATTR, True)


def _exception_chain(exception: BaseException) -> Iterator[BaseException]:
    current = exception  # rebind-ok: advances one link per iteration of the bounded walk
    for _ in range(_MAX_EXCEPTION_CHAIN_DEPTH):
        yield current
        following = current.__cause__ or current.__context__
        if following is None:
            return
        current = following


def mcp_tools_executed(exception: BaseException | None) -> bool:
    """Whether ``exception`` or an exception in its chain was tagged by ``mark_mcp_tools_executed``.

    The chain matters because a caller can re-raise a tagged error as a new exception, as
    ``exception_type`` does for anything that is not already a LiteLLM exception.
    """
    return exception is not None and any(
        getattr(link, _MCP_TOOLS_EXECUTED_ATTR, False) is True for link in _exception_chain(exception)
    )


class MCPToolReplayGuard:
    """Tags errors raised inside ``with guard:`` once the request has sent an MCP tool call to a server."""

    __slots__ = ("_tool_dispatched",)

    def __init__(self) -> None:
        self._tool_dispatched: bool = False

    def record_tool_dispatch(self) -> None:
        self._tool_dispatched = True

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._tool_dispatched and isinstance(exc, Exception):
            mark_mcp_tools_executed(exc)

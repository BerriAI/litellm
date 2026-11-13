from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

from litellm.litellm_core_utils.agentic_followup_failure import mark_failure_after_tool_calls_sent


class ToolDispatch:
    """Whether one request handed an MCP tool call to its server's transport.

    Once one has, the tool may have run whatever happens next, so a later failure is marked to be
    returned as it is: a router retry or fallback would run the whole tool loop, and the tool, again.
    """

    def __init__(self) -> None:
        self._sent: bool = False

    @property
    def sent(self) -> bool:
        return self._sent

    def record_sent(self) -> None:
        self._sent = True

    @contextmanager
    def recording(self) -> Generator[None]:
        token: Final = _ACTIVE_TOOL_DISPATCH.set(self)
        try:
            yield
        finally:
            _ACTIVE_TOOL_DISPATCH.reset(token)

    @contextmanager
    def failures_not_replayed(self) -> Generator[None]:
        try:
            yield
        except Exception as error:
            if self._sent:
                mark_failure_after_tool_calls_sent(error)
            raise


_ACTIVE_TOOL_DISPATCH: Final[ContextVar[ToolDispatch | None]] = ContextVar("litellm_mcp_tool_dispatch", default=None)


def active_tool_dispatch() -> ToolDispatch | None:
    return _ACTIVE_TOOL_DISPATCH.get()

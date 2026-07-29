import asyncio
import weakref
from collections.abc import Awaitable, Callable

from mcp.server.session import ServerSession

from litellm._logging import verbose_logger


class MCPListChangeNotifier:
    def __init__(self) -> None:
        self._tool_sessions: weakref.WeakSet[ServerSession] = weakref.WeakSet()
        self._resource_sessions: weakref.WeakSet[ServerSession] = weakref.WeakSet()

    def remember_tool_list_session(self, session: ServerSession) -> None:
        self._tool_sessions.add(session)

    def remember_resource_list_session(self, session: ServerSession) -> None:
        self._resource_sessions.add(session)

    async def notify_lists_changed(self) -> None:
        await asyncio.gather(
            self._notify_sessions(
                tuple(self._tool_sessions),
                lambda session: session.send_tool_list_changed(),
                self._tool_sessions,
                "tool",
            ),
            self._notify_sessions(
                tuple(self._resource_sessions),
                lambda session: session.send_resource_list_changed(),
                self._resource_sessions,
                "resource",
            ),
        )

    async def _notify_sessions(
        self,
        sessions: tuple[ServerSession, ...],
        send_notification: Callable[[ServerSession], Awaitable[None]],
        registered_sessions: weakref.WeakSet[ServerSession],
        list_name: str,
    ) -> None:
        results = await asyncio.gather(
            *(send_notification(session) for session in sessions),
            return_exceptions=True,
        )
        for session, result in zip(sessions, results):
            if isinstance(result, BaseException):
                registered_sessions.discard(session)
                verbose_logger.warning(
                    "Failed to send MCP %s list-changed notification: %s",
                    list_name,
                    result,
                )


mcp_list_change_notifier = MCPListChangeNotifier()

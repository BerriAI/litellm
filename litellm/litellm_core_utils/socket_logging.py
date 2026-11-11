import asyncio
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, TypeVar

from litellm._logging import verbose_logger
from litellm.constants import WEBSOCKET_TURN_LOGGING_WAIT_SECONDS

_T: Final = TypeVar("_T")


class SocketLogging:
    """The logging a WebSocket's turns started, so its close can settle the socket's token reservation after its
    turns charged their usage."""

    def __init__(self, wait_seconds: float = WEBSOCKET_TURN_LOGGING_WAIT_SECONDS) -> None:
        self._wait_seconds: Final = wait_seconds
        self._pending: tuple[asyncio.Future[None], ...] = ()

    def tracked(self, coroutine: Coroutine[object, object, _T]) -> Coroutine[object, object, _T]:
        logged: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()
        self._pending = (*(pending for pending in self._pending if not pending.done()), logged)
        return _resolve_when_done(coroutine, logged)

    async def finished(self) -> None:
        pending: Final = tuple(logged for logged in self._pending if not logged.done())
        if not pending:
            return
        _, unfinished = await asyncio.wait(pending, timeout=self._wait_seconds)
        if unfinished:
            verbose_logger.warning(
                "%d WebSocket turn logging tasks were still running after %ss", len(unfinished), self._wait_seconds
            )


async def _resolve_when_done(coroutine: Coroutine[object, object, _T], logged: asyncio.Future[None]) -> _T:
    try:
        return await coroutine
    finally:
        if not logged.done():
            logged.set_result(None)


_socket_logging: Final[ContextVar[SocketLogging | None]] = ContextVar("socket_logging", default=None)


@contextmanager
def collect_socket_logging() -> Generator[SocketLogging, None, None]:
    socket_logging: Final = SocketLogging()
    token: Final = _socket_logging.set(socket_logging)
    try:
        yield socket_logging
    finally:
        _socket_logging.reset(token)


def track_socket_logging(coroutine: Coroutine[object, object, _T]) -> Coroutine[object, object, _T]:
    socket_logging: Final = _socket_logging.get()
    return coroutine if socket_logging is None else socket_logging.tracked(coroutine)

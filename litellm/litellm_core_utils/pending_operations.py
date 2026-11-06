from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final


class PendingOperations:
    def __init__(self, completed: Callable[[], None]) -> None:
        self._completed = completed
        self._pending = 0
        self._closed = False
        self._released = False

    def started(self) -> bool:
        if self._released:
            return False
        self._pending += 1
        return True

    def finished(self) -> None:
        self._pending -= 1
        if self._closed and self._pending == 0:
            self._release()

    def close(self) -> None:
        self._closed = True
        if self._pending == 0:
            self._release()

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._completed()


_PENDING: Final[ContextVar[PendingOperations | None]] = ContextVar("litellm_pending_operations", default=None)


@contextmanager
def defer_until_operations_finish(completed: Callable[[], None]) -> Generator[None, None, None]:
    operations: Final = PendingOperations(completed)
    token: Final = _PENDING.set(operations)
    try:
        yield
    finally:
        _PENDING.reset(token)
        operations.close()


def register_pending_operation() -> PendingOperations | None:
    operations: Final = _PENDING.get()
    return operations if operations is not None and operations.started() else None


def finish_pending_operation(operations: PendingOperations | None) -> None:
    if operations is not None:
        operations.finished()

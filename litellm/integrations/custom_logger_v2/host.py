from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

from litellm.integrations.custom_logger_v2.dispatcher import Dispatcher, RequestScope
from litellm.integrations.custom_logger_v2.outcomes import Outcome, Patch
from litellm.integrations.custom_logger_v2.registry import CallbackRegistry
from litellm.integrations.custom_logger_v2.selection import RequestSelection

_NO_SELECTION: Final = RequestSelection()
_CURRENT_SCOPE: Final[ContextVar[RequestScope | None]] = ContextVar("custom_logger_v2_scope", default=None)


class V2Outcome(Exception):
    def __init__(self, outcome: Outcome | None, patches: tuple[Patch, ...]) -> None:
        super().__init__(f"a V2 callback ended the hook with {type(outcome).__name__}")
        self.outcome: Final = outcome
        self.patches: Final = patches


class _Host:
    def __init__(self) -> None:
        self._dispatcher: Dispatcher | None = None

    def dispatcher(self) -> Dispatcher:
        if self._dispatcher is None:
            self._dispatcher = Dispatcher(CallbackRegistry())
        return self._dispatcher

    def reset(self) -> None:
        self._dispatcher = None


_HOST: Final = _Host()


def get_dispatcher() -> Dispatcher:
    return _HOST.dispatcher()


def reset_host_for_tests() -> None:
    _HOST.reset()


def current_scope() -> RequestScope | None:
    return _CURRENT_SCOPE.get()


@contextmanager
def bind_scope(scope: RequestScope) -> Generator[RequestScope]:
    token: Final = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


def enter_request_scope(selection: RequestSelection) -> RequestScope:
    scope: Final = _HOST.dispatcher().open_scope(selection)
    _ = _CURRENT_SCOPE.set(scope)
    return scope


@contextmanager
def request_scope(selection: RequestSelection = _NO_SELECTION) -> Generator[RequestScope]:
    with _HOST.dispatcher().open_scope(selection) as scope, bind_scope(scope):
        yield scope

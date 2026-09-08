from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Final

import pytest

from litellm.rust_bridge._lifecycle import NativeOutcome, drive_async, drive_sync


class _Machine:
    def __init__(self, replace: bool = False) -> None:
        self.outcome: NativeOutcome | None = None
        self.replace: Final = replace

    def complete(self) -> bool | None:
        if self.outcome is None:
            return None
        return self.outcome is NativeOutcome.SUCCESS

    def advance(self, outcome: int, logger_available: bool, has_fallbacks: bool) -> bool:
        self.outcome = NativeOutcome(outcome)
        return self.replace


class _Host:
    def __init__(self, invoke: Callable[[], tuple[bool, object]], replace: bool = False) -> None:
        self.machine: Final = _Machine(replace)
        self._invoke: Final = invoke
        self.error: BaseException | None = None

    def invoke(self) -> tuple[bool, object]:
        return self._invoke()

    def advance(self, outcome: NativeOutcome, error: BaseException | None = None) -> None:
        if self.machine.advance(outcome, True, False):
            self.error = error

    def result(self) -> object:
        if self.machine.complete():
            return "complete"
        if self.error is None:
            raise RuntimeError("missing test error")
        raise self.error


def test_drive_sync_returns_terminal_result() -> None:
    host: Final = _Host(lambda: (False, None))

    assert drive_sync(host) == "complete"
    assert host.machine.outcome is NativeOutcome.SUCCESS


def test_drive_sync_replaces_an_ordinary_failure() -> None:
    failure: Final = ValueError("failed")
    host: Final = _Host(lambda: (_ for _ in ()).throw(failure), replace=True)

    with pytest.raises(ValueError, match="failed") as raised:
        drive_sync(host)

    assert raised.value is failure
    assert host.machine.outcome is NativeOutcome.FAILURE


def test_drive_sync_classifies_base_exception_as_abort() -> None:
    class Abort(BaseException):
        pass

    failure: Final = Abort("aborted")
    host: Final = _Host(lambda: (_ for _ in ()).throw(failure), replace=True)

    with pytest.raises(Abort, match="aborted"):
        drive_sync(host)

    assert host.machine.outcome is NativeOutcome.ABORT


@pytest.mark.asyncio
async def test_drive_async_awaits_the_selected_operation() -> None:
    completed: Final = asyncio.Event()

    async def operation() -> None:
        await asyncio.sleep(0)
        completed.set()

    host: Final = _Host(lambda: (True, operation()))

    assert await drive_async(host) == "complete"
    assert completed.is_set()
    assert host.machine.outcome is NativeOutcome.SUCCESS

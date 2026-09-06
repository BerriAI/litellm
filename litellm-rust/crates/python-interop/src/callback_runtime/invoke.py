import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Final, cast


def invoke_direct(callback: Callable[[object], object], payload: object) -> object:
    result: Final = callback(payload)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("direct hook returned an awaitable")
    return result


def invoke_direct_zero(callback: Callable[[], object]) -> object:
    result: Final = callback()
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("direct hook returned an awaitable")
    return result


class Invocation:
    def __init__(
        self,
        callback: Callable[[object], object],
        payload: object,
        returns_awaitable: bool,
        admission: object,
        zero_arguments: bool = False,
    ) -> None:
        self.callback = callback
        self.payload = payload
        self.returns_awaitable = returns_awaitable
        self.admission: object | None = admission
        self.zero_arguments = zero_arguments
        self.task: asyncio.Task[object] | None = None
        self.cancelled = False

    async def run(self) -> object:
        self.task = asyncio.current_task()
        try:
            if self.cancelled:
                raise asyncio.CancelledError()
            if not self.returns_awaitable:
                if self.zero_arguments:
                    return invoke_direct_zero(self.callback)
                return invoke_direct(self.callback, self.payload)
            result: Final = self.callback() if self.zero_arguments else self.callback(self.payload)
            if not inspect.isawaitable(result):
                raise TypeError("awaitable hook returned a non-awaitable")
            return await cast(Awaitable[object], result)
        finally:
            self.admission = None
            self.task = None

    def cancel(self) -> None:
        self.cancelled = True
        if self.task is not None:
            self.task.cancel()

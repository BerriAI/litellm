import asyncio
from typing import Final

from litellm.constants import PROXY_DB_LOOKUP_MAX_CONCURRENCY


class LoopBoundSemaphore:
    __slots__ = ("_loop", "_semaphore", "_value")

    def __init__(self, value: int) -> None:
        self._value: Final = value
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def current(self) -> asyncio.Semaphore:
        loop: Final = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self._value)
            self._loop = loop
        return self._semaphore


db_lookup_gate: Final = LoopBoundSemaphore(PROXY_DB_LOOKUP_MAX_CONCURRENCY)

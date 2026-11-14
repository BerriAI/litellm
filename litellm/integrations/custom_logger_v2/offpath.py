import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Protocol


class OffPathSink(Protocol):
    def submit(self, delivery: Coroutine[object, object, None]) -> None: ...


class InProcessSink:
    def __init__(self) -> None:
        self._pending: set[asyncio.Task[None]] = set()  # mutable-ok: strong references keep running tasks alive
        self._loopless: Final = ThreadPoolExecutor(max_workers=1, thread_name_prefix="custom-logger-v2-off-path")

    def submit(self, delivery: Coroutine[object, object, None]) -> None:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            _ = self._loopless.submit(asyncio.run, delivery)
            return
        task: Final[asyncio.Task[None]] = loop.create_task(delivery)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        await asyncio.gather(*self._pending)

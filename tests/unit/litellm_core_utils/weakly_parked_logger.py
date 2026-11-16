import asyncio
import gc
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.responses.streaming_iterator import drain_logging_tasks

_DEADLOCK_GUARD_SECONDS: Final = 5


class WeaklyParkedLogger(CustomLogger):
    """Parks the success callback on a future only its logging task reaches, which asyncio's weak task refs lose."""

    def __init__(self) -> None:
        super().__init__()
        self.entered: Final = asyncio.Event()
        self.finished: Final = asyncio.Event()
        self.parks = 0
        self._release: weakref.ref[asyncio.Future[None]] | None = None
        self._task: weakref.ref[asyncio.Task[object]] | None = None

    async def park(self) -> None:
        release: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()
        current: Final = asyncio.current_task()
        self._release = weakref.ref(release)
        self._task = None if current is None else weakref.ref(current)
        self.parks += 1
        self.entered.set()
        await release
        self.finished.set()

    async def async_log_success_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self.park()

    async def async_log_stream_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self.park()

    def _release_parked(self) -> bool:
        release: Final = None if self._release is None else self._release()
        if release is None:
            return False
        if not release.done():
            release.set_result(None)
        return True

    async def survives_collection(self) -> bool:
        await asyncio.wait_for(self.entered.wait(), _DEADLOCK_GUARD_SECONDS)
        _ = gc.collect()
        return self._release_parked()

    async def settle(self) -> bool:
        _ = self._release_parked()
        task: Final = None if self._task is None else self._task()
        if task is not None and not task.done():
            _, pending = await asyncio.wait((task,), timeout=_DEADLOCK_GUARD_SECONDS)
            if pending:
                _ = task.cancel()
                _ = await asyncio.wait((task,))
                return False
        await drain_logging_tasks(timeout=_DEADLOCK_GUARD_SECONDS)
        return True


@asynccontextmanager
async def weakly_parked_logger() -> AsyncIterator[WeaklyParkedLogger]:
    logger: Final = WeaklyParkedLogger()
    try:
        yield logger
    except BaseException:
        _ = await logger.settle()
        raise
    assert await logger.settle(), "the released logging callback was still pending and had to be cancelled"

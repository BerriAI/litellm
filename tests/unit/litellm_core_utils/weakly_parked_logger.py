import asyncio
import gc
import weakref
from typing import Final

from litellm.integrations.custom_logger import CustomLogger

_DEADLOCK_GUARD_SECONDS: Final = 5


class WeaklyParkedLogger(CustomLogger):
    """Parks the success callback on a future only the logging task can reach, the shape asyncio's weak task refs lose."""

    def __init__(self) -> None:
        super().__init__()
        self.entered: Final = asyncio.Event()
        self.finished: Final = asyncio.Event()
        self._release: weakref.ref[asyncio.Future[None]] | None = None

    async def park(self) -> None:
        release: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()
        self._release = weakref.ref(release)
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

    async def survives_collection(self) -> bool:
        await asyncio.wait_for(self.entered.wait(), _DEADLOCK_GUARD_SECONDS)
        _ = gc.collect()
        release: Final = None if self._release is None else self._release()
        if release is None:
            return False
        release.set_result(None)
        return True

import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Final

from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.errors import exhausted

_WINDOW_SECONDS: Final = 60.0
_TRACKED_CALLERS: Final = 10_000


class EventBudget:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()

    def take(self, caller: str, limit: int) -> bool:
        now: Final = self._clock()
        window: Final = self._windows.get(caller)
        start, used = window if window is not None and now - window[0] < _WINDOW_SECONDS else (now, 0)
        if used >= limit:
            return False
        self._windows[caller] = (start, used + 1)
        self._windows.move_to_end(caller)
        while len(self._windows) > _TRACKED_CALLERS:
            self._windows.popitem(last=False)
        return True


class StreamSlots:
    def __init__(self) -> None:
        self._per_caller: OrderedDict[str, int] = OrderedDict()
        self._total = 0

    @asynccontextmanager
    async def hold(self, caller: str, settings: EventsSettings) -> AsyncIterator[None]:
        if self._total >= settings.max_streams:
            raise exhausted("streams", settings.max_streams).to_mcp()
        if self._per_caller.get(caller, 0) >= settings.max_streams_per_caller:
            raise exhausted("streams", settings.max_streams_per_caller).to_mcp()
        self._total += 1
        self._per_caller[caller] = self._per_caller.get(caller, 0) + 1
        try:
            yield
        finally:
            self._total -= 1
            remaining: Final = self._per_caller.get(caller, 1) - 1
            if remaining:
                self._per_caller[caller] = remaining
            else:
                self._per_caller.pop(caller, None)

    def live(self, caller: str) -> int:
        return self._per_caller.get(caller, 0)

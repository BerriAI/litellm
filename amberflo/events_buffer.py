from asyncio import Lock
from datetime import datetime, timezone

from .logging import get_logger


logger = get_logger(__name__)


class EventsBuffer:
    """
    Manages event buffering and thread-safe operations.
    """

    def __init__(self, max_buffer_size: int = 10000) -> None:
        self.buffer: list[dict] = []
        self.max_buffer_size = max_buffer_size
        self.first_entry_time: datetime | None = None
        self._buffer_lock: Lock | None = None

    @property
    def buffer_lock(self) -> Lock:
        """
        Lazy initialization of asyncio.Lock to avoid event loop issues.
        """
        if self._buffer_lock is None:
            self._buffer_lock = Lock()
        return self._buffer_lock

    async def add_events(self, events: list[dict]) -> int:
        """
        Add events to buffer. Return buffer size to allow client to trigger a
        flush based on buffer size.
        """
        async with self.buffer_lock:
            if len(self.buffer) + len(events) > self.max_buffer_size:
                logger.warning(
                    f"Dropping {len(events)} events due to buffer being full (capacity: {self.max_buffer_size}, size: {len(self.buffer)}"
                )
                return len(self.buffer)

            if not self.first_entry_time:
                self.first_entry_time = datetime.now(timezone.utc)

            self.buffer.extend(events)
            return len(self.buffer)

    async def extract_all(self) -> tuple[list[dict], datetime | None]:
        """
        Extract all events from buffer and reset state.
        """
        async with self.buffer_lock:
            events = self.buffer.copy()
            first_time = self.first_entry_time

            self.buffer.clear()
            self.first_entry_time = None

            return events, first_time

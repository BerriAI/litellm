import asyncio
import json
import gzip
from datetime import datetime
from typing import cast

from .blob_client import BlobWriter
from .events_buffer import EventsBuffer
from .utils import get_env
from .utils import positive_int
from .logging import get_logger


logger = get_logger(__name__)


class AsyncEventsWriter:
    """
    Wraps a synchronous blob client so that the writes can happen
    asynchronously.

    Also:
    - formats the "key" and "body" of the blob to be written;
    - manages the buffering of events, and graceful shutdown.
    """

    def __init__(
        self,
        client: BlobWriter,
        buffer: EventsBuffer,
        flush_interval=60,
        batch_size=200,
    ):
        self.client = client
        self.buffer = buffer

        self.flush_interval = int(
            get_env("AFLO_FLUSH_INTERVAL", flush_interval, validate=positive_int)
        )
        self.batch_size = int(
            get_env("AFLO_BATCH_SIZE", batch_size, validate=positive_int)
        )

        # Coordination elements will be lazily initialized to avoid issues with
        # event loop not being ready
        self.flush_task: asyncio.Task | None = None
        self._flush_lock: asyncio.Lock | None = None
        self._initialized = False
        logger.debug(
            "Async events writer initialized: flush_interval: %s, batch_size: %s",
            self.flush_interval,
            self.batch_size,
        )

    @property
    def flush_lock(self) -> asyncio.Lock:
        """
        Lazy initialization of asyncio.Lock to avoid event loop issues.
        """
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()
        return self._flush_lock

    async def async_init(self) -> None:
        if not self._initialized:
            self._initialized = True

            loop = asyncio.get_running_loop()
            self.flush_task = loop.create_task(self._periodic_flush())
            logger.debug("Async event writer async initialization completed")

    async def async_write(self, events):
        """
        Adds the event to a buffer for asynchronously sending them.
        """
        await self.async_init()

        logger.debug(f"Received {len(events)} events to write asynchronously")

        try:
            buffer_size = await self.buffer.add_events(events)

            if buffer_size >= self.batch_size:
                logger.debug(
                    f"Buffer reached batch size ({buffer_size} >= {self.batch_size}), flushing..."
                )
                async with self.flush_lock:
                    await self._flush_buffer()

        except Exception:
            logger.exception("Failed to write events async")

    async def _periodic_flush(self) -> None:
        """
        Background task for periodic flushing.
        """
        while True:
            try:
                await asyncio.sleep(self.flush_interval)

                logger.debug("Periodic flush timer triggered, flushing...")

                async with self.flush_lock:
                    await self._flush_buffer()

            except asyncio.CancelledError:
                logger.debug("Periodic flush task cancelled, flushing...")

                # Flush remaining events in the queue before cancellation
                async with self.flush_lock:
                    await self._flush_buffer()

                # Propagate cancellation
                raise

            except Exception:
                logger.exception("Error in async writer periodic flush")

    async def _flush_buffer(self) -> None:
        """
        Flush the current contents of the buffer.
        """
        key = None

        try:
            events, first_entry_time = await self.buffer.extract_all()
            if not events:
                logger.debug("No events to flush")
                return

            key = self.client.make_key(cast(datetime, first_entry_time))

            logger.info(f"Flushing {len(events)} events to: {key}")

            body = _prepare_body(events)

            # Avoid blocking
            asyncio.create_task(self.client.put_object(key, body))

        except asyncio.CancelledError:
            logger.warning(f"Flushing cancelled: {key}")
            raise


def _prepare_body(events) -> bytes:
    json_bytes = json.dumps(events).encode()
    compressed = gzip.compress(json_bytes)

    file_size = len(compressed)
    ratio = len(compressed) / len(json_bytes)
    logger.debug(f"Events file size: {file_size} (compression ratio: {ratio:.2f})")

    return compressed

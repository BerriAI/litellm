"""
Shared base for everything LiteLLM writes to ClickHouse.

Built on `CustomBatchLogger`: rows accumulate in `log_queue` and are flushed as one
gzip JSONEachRow insert, either every `CLICKHOUSE_FLUSH_INTERVAL_SECONDS` or as soon as
`batch_size` rows are queued. Subclasses only pick the table and build rows:

- `ClickHouseSpendLogger`  -> spend_logs   (LiteLLM requests when tracing is enabled)
"""

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import ClassVar, Final

from litellm._logging import verbose_logger
from litellm.constants import (
    CLICKHOUSE_BATCH_SIZE,
    CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
    CLICKHOUSE_MAX_BUFFERED_ROWS,
    CLICKHOUSE_MAX_RETRIES,
)
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.rust_bridge.clickhouse import ClickHouseSpendStorage, spend_storage_config


def clickhouse_storage_from_env() -> ClickHouseSpendStorage:
    return ClickHouseSpendStorage(spend_storage_config())


class ClickHouseBatchLogger(CustomBatchLogger):
    table: ClassVar[str]

    def __init__(self, storage: ClickHouseSpendStorage | None = None) -> None:
        self.storage = storage or clickhouse_storage_from_env()
        self.rows_written = 0
        self.rows_dropped = 0
        self._failed_attempts = 0
        super().__init__(
            flush_lock=asyncio.Lock(),
            batch_size=CLICKHOUSE_BATCH_SIZE,
            flush_interval=CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
        )
        self._flush_task: asyncio.Task[None] | None = None
        self._stop: Final = asyncio.Event()

    def start(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.get_running_loop().create_task(self.periodic_flush())

    async def aclose(self) -> None:
        self._stop.set()
        if self._flush_task is not None:
            await self._flush_task
        while self.log_queue:
            await self.flush_queue()

    async def periodic_flush(self) -> None:
        while True:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.flush_interval)
            if self._stop.is_set():
                return
            await self.flush_queue()

    def is_full(self) -> bool:
        """Backpressure signal: producers should reject (429) instead of enqueueing."""
        return len(self.log_queue) >= CLICKHOUSE_MAX_BUFFERED_ROWS

    def enqueue(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Never awaits ClickHouse. Kicks off an early flush once a full batch is queued."""
        self.start()
        self.log_queue.extend(rows)
        if len(self.log_queue) >= self.batch_size:
            asyncio.get_running_loop().create_task(self.flush_queue())

    async def flush_queue(self) -> None:
        # Swap the queue under the lock so rows enqueued during the insert are kept.
        if self.flush_lock is None:
            return
        async with self.flush_lock:
            while self.log_queue:
                batch = self.log_queue[: self.batch_size]
                self.log_queue = self.log_queue[len(batch) :]
                if not await self._insert(batch):
                    break

    async def async_send_batch(self) -> None:
        await self.flush_queue()

    async def _insert(self, batch: list[dict[str, object]]) -> bool:
        try:
            await self.storage.insert_rows(self.table, batch)
            self.rows_written += len(batch)
            self._failed_attempts = 0
            return True
        except Exception as e:
            self._failed_attempts += 1
            if self._failed_attempts >= CLICKHOUSE_MAX_RETRIES:
                self.rows_dropped += len(batch)
                self._failed_attempts = 0
                verbose_logger.error(
                    "ClickHouse: dropped %s rows for %s after %s attempts: %s",
                    len(batch),
                    self.table,
                    CLICKHOUSE_MAX_RETRIES,
                    e,
                )
            else:
                # put it back; the next periodic flush retries it
                self.log_queue = batch + self.log_queue
                verbose_logger.warning("ClickHouse: insert into %s failed, will retry: %s", self.table, e)
            return False

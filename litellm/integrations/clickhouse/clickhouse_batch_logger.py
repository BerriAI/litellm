"""
Shared base for everything LiteLLM writes to ClickHouse.

Built on `CustomBatchLogger`: rows accumulate in `log_queue` and are flushed as one
gzip JSONEachRow insert, either every `CLICKHOUSE_FLUSH_INTERVAL_SECONDS` or as soon as
`batch_size` rows are queued. Subclasses only pick the table and build rows:

- `ClickHouseSpendLogger`  -> spend_logs   (LiteLLM requests, via the `clickhouse` callback)
"""

import asyncio
import os
from typing import Any, ClassVar

from litellm._logging import verbose_logger
from litellm.constants import (
    CLICKHOUSE_BATCH_SIZE,
    CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
    CLICKHOUSE_MAX_BUFFERED_ROWS,
    CLICKHOUSE_MAX_RETRIES,
)
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.rust_bridge.traces import TraceStorage


def clickhouse_storage_from_env() -> TraceStorage:
    return TraceStorage(
        database=os.getenv("CLICKHOUSE_DATABASE", "litellm"),
        url=os.getenv("CLICKHOUSE_URL", ""),
    )


class ClickHouseBatchLogger(CustomBatchLogger):
    table: ClassVar[str]

    def __init__(self, storage: TraceStorage | None = None, **kwargs: Any) -> None:
        self.storage = storage or clickhouse_storage_from_env()
        self.rows_written = 0
        self.rows_dropped = 0
        self._failed_attempts = 0
        super().__init__(
            flush_lock=asyncio.Lock(),
            batch_size=CLICKHOUSE_BATCH_SIZE,
            flush_interval=CLICKHOUSE_FLUSH_INTERVAL_SECONDS,  # type: ignore[arg-type]
            **kwargs,
        )
        try:
            asyncio.get_running_loop().create_task(self.periodic_flush())
        except RuntimeError:  # no loop yet (e.g. sync config load); proxy startup calls start()
            pass

    def start(self) -> None:
        asyncio.get_running_loop().create_task(self.periodic_flush())

    def is_full(self) -> bool:
        """Backpressure signal: producers should reject (429) instead of enqueueing."""
        return len(self.log_queue) >= CLICKHOUSE_MAX_BUFFERED_ROWS

    def enqueue(self, rows: list[dict[str, Any]]) -> None:
        """Never awaits ClickHouse. Kicks off an early flush once a full batch is queued."""
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

    async def async_send_batch(self, *args: Any, **kwargs: Any) -> None:
        await self.flush_queue()

    async def _insert(self, batch: list[dict[str, Any]]) -> bool:
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

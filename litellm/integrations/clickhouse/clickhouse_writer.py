"""
Bounded, batched writer: many producers enqueue rows, one background task flushes.

One instance per worker process. Producers never await ClickHouse — they either
enqueue or are told the buffer is full (the OTLP endpoint turns that into a 429).
"""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List

from litellm._logging import verbose_logger
from litellm.constants import (
    CLICKHOUSE_BATCH_SIZE,
    CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
    CLICKHOUSE_MAX_BUFFERED_ROWS,
    CLICKHOUSE_MAX_RETRIES,
)
from litellm.integrations.clickhouse.clickhouse_client import ClickHouseClient


class ClickHouseWriter:
    def __init__(self, client: ClickHouseClient):
        self.client = client
        self.buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.rows_written: Dict[str, int] = defaultdict(int)
        self.rows_dropped: Dict[str, int] = defaultdict(int)
        self._flush_lock = asyncio.Lock()
        self._task: "asyncio.Task | None" = None

    def is_full(self, table: str) -> bool:
        return len(self.buffers[table]) >= CLICKHOUSE_MAX_BUFFERED_ROWS

    def enqueue(self, table: str, rows: List[Dict[str, Any]]) -> None:
        self.buffers[table].extend(rows)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self) -> None:
        while True:
            await asyncio.sleep(CLICKHOUSE_FLUSH_INTERVAL_SECONDS)
            await self.flush()

    async def flush(self) -> None:
        async with self._flush_lock:
            for table in list(self.buffers):
                while self.buffers[table]:
                    batch = self.buffers[table][:CLICKHOUSE_BATCH_SIZE]
                    del self.buffers[table][: len(batch)]
                    await self._insert_with_retry(table, batch)

    async def _insert_with_retry(self, table: str, batch: List[Dict[str, Any]]) -> None:
        for attempt in range(CLICKHOUSE_MAX_RETRIES):
            try:
                await self.client.insert_json_each_row(table, batch)
                self.rows_written[table] += len(batch)
                return
            except Exception as e:
                verbose_logger.warning(
                    "ClickHouseWriter: insert into %s failed (attempt %s): %s",
                    table,
                    attempt + 1,
                    e,
                )
                await asyncio.sleep(2**attempt)
        self.rows_dropped[table] += len(batch)
        verbose_logger.error(
            "ClickHouseWriter: dropped %s rows for %s after %s retries",
            len(batch),
            table,
            CLICKHOUSE_MAX_RETRIES,
        )

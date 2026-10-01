import asyncio
import concurrent.futures
import io
import json
import os
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import accumulate, islice, takewhile
from typing import ClassVar, Final, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import (
    CLICKHOUSE_BATCH_SIZE,
    CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
    CLICKHOUSE_MAX_BUFFERED_ROWS,
    CLICKHOUSE_MAX_RETRIES,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge.traces import TraceStorage

_ROW: Final = TypeAdapter(dict[str, JsonValue])
_MAX_ROW_BYTES: Final = 1024 * 1024
_MAX_BATCH_BYTES: Final = 8 * 1024 * 1024
_MAX_BUFFER_BYTES: Final = 64 * 1024 * 1024


class SpendStorage(Protocol):
    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None: ...


def clickhouse_storage_from_env() -> TraceStorage:
    return TraceStorage(database=os.getenv("CLICKHOUSE_DATABASE", "litellm"), url=os.getenv("CLICKHOUSE_URL", ""))


def _encode_row(row: Mapping[str, object], limit: int) -> bytes | None:
    buffer: Final = io.BytesIO()
    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(_ROW.validate_python(row)):
        if len(chunk) > limit:
            return None
        encoded: Final = chunk.encode()
        if buffer.tell() + len(encoded) > limit:
            return None
        buffer.write(encoded)
    return buffer.getvalue()


class ClickHouseBatchLogger(CustomLogger):
    table: ClassVar[str]

    def __init__(
        self,
        storage: SpendStorage | None = None,
        *,
        batch_size: int = CLICKHOUSE_BATCH_SIZE,
        flush_interval: float = CLICKHOUSE_FLUSH_INTERVAL_SECONDS,
        max_rows: int = CLICKHOUSE_MAX_BUFFERED_ROWS,
        max_bytes: int = _MAX_BUFFER_BYTES,
        max_row_bytes: int = _MAX_ROW_BYTES,
        max_batch_bytes: int = _MAX_BATCH_BYTES,
        max_retries: int = CLICKHOUSE_MAX_RETRIES,
        prepare: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self.storage: Final = storage if storage is not None else clickhouse_storage_from_env()
        self.batch_size: Final = batch_size
        self.flush_interval: Final = flush_interval
        self.max_rows: Final = max_rows
        self.max_bytes: Final = max_bytes
        self.max_row_bytes: Final = min(max_row_bytes, max_batch_bytes, max_bytes)
        self.max_batch_bytes: Final = max_batch_bytes
        self.max_retries: Final = max_retries
        self._prepare: Final = prepare
        self._prepared = prepare is None
        self._queue: Final = deque[bytes]()
        self._lock: Final = threading.Lock()
        self._started: Final = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._flush_lock: asyncio.Lock | None = None
        self._notified = False
        self._closing = False
        self._closed = False
        self._inflight: tuple[bytes, ...] = ()
        self._attempts = 0
        self._buffered_bytes = 0
        self.rows_written = 0
        self.rows_dropped = 0

    @property
    def buffered_rows(self) -> int:
        with self._lock:
            return len(self._queue) + len(self._inflight)

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return self._buffered_bytes

    def is_full(self) -> bool:
        with self._lock:
            return len(self._queue) + len(self._inflight) >= self.max_rows or self._buffered_bytes >= self.max_bytes

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._closing:
                return
            self._thread = threading.Thread(target=self._run, name="clickhouse-spend", daemon=True)
            self._thread.start()

    def _notify(self) -> None:
        with self._lock:
            if self._notified or self._loop is None or self._wakeup is None or self._closed:
                return
            self._notified = True
            self._loop.call_soon_threadsafe(self._wakeup.set)

    def enqueue(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.start()
        for row in rows:
            encoded: Final = _encode_row(row, self.max_row_bytes)
            with self._lock:
                if (
                    encoded is None
                    or self._closing
                    or len(self._queue) + len(self._inflight) >= self.max_rows
                    or self._buffered_bytes + len(encoded) > self.max_bytes
                ):
                    self.rows_dropped += 1
                    verbose_logger.warning("ClickHouse: dropped spend row because the writer is closed or at capacity")
                    continue
                self._queue.append(encoded)
                self._buffered_bytes += len(encoded)
        if self.buffered_rows >= self.batch_size:
            self._notify()

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except asyncio.CancelledError:
            pass

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._flush_lock = asyncio.Lock()
        self._worker = asyncio.current_task()
        self._started.set()
        if self._closing or self.buffered_rows >= self.batch_size:
            self._wakeup.set()
        try:
            while True:
                if self._closing and not self.buffered_rows:
                    return
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=self.flush_interval)
                except TimeoutError:
                    pass
                self._wakeup.clear()
                with self._lock:
                    self._notified = False
                await self._flush()
                if self._closing and self.buffered_rows:
                    self._wakeup.set()
        finally:
            with self._lock:
                remaining: Final = len(self._queue) + len(self._inflight)
                self.rows_dropped += remaining
                self._queue.clear()
                self._inflight = ()
                self._buffered_bytes = 0
                self._closed = True
            if remaining:
                verbose_logger.error("ClickHouse: dropped %s spend rows at shutdown deadline", remaining)

    def _batch(self) -> tuple[bytes, ...]:
        with self._lock:
            if self._inflight:
                return self._inflight
            candidates: Final = tuple(islice(self._queue, self.batch_size))
            sizes: Final = accumulate(len(row) + 1 for row in candidates)
            self._inflight = tuple(
                row for row, _ in takewhile(lambda pair: pair[1] <= self.max_batch_bytes + 1, zip(candidates, sizes))
            )
            for _ in self._inflight:
                self._queue.popleft()
            return self._inflight

    def _finish(self, written: bool) -> None:
        with self._lock:
            if written:
                self.rows_written += len(self._inflight)
            else:
                self.rows_dropped += len(self._inflight)
            self._buffered_bytes -= sum(map(len, self._inflight))
            self._inflight = ()
            self._attempts = 0

    async def _flush(self) -> None:
        if self._flush_lock is None:
            return
        async with self._flush_lock:
            try:
                if not self._prepared and self._prepare is not None:
                    await self._prepare()
                    self._prepared = True
                while batch := self._batch():
                    await self.storage.insert_rows(self.table, tuple(_ROW.validate_json(row) for row in batch))
                    self._finish(True)
            except Exception as error:
                self._attempts += 1
                if self._attempts >= self.max_retries and self._inflight:
                    verbose_logger.error("ClickHouse: dropped %s rows after retries: %s", len(self._inflight), error)
                    self._finish(False)
                else:
                    verbose_logger.warning("ClickHouse: spend delivery delayed: %s", error)

    async def flush_queue(self) -> None:
        self.start()
        if self._closed or self._closing:
            return
        await asyncio.to_thread(self._started.wait)
        if self._loop is not None and not self._closed:
            future: Final[concurrent.futures.Future[None]] = asyncio.run_coroutine_threadsafe(self._flush(), self._loop)
            await asyncio.wrap_future(future)

    async def aclose(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closing = True
            thread: Final = self._thread
        if thread is None:
            return
        self._notify()
        await asyncio.to_thread(thread.join, timeout)
        if thread.is_alive() and self._loop is not None and self._worker is not None:
            self._loop.call_soon_threadsafe(self._worker.cancel)
            await asyncio.to_thread(thread.join, 1.0)

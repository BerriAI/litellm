import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from enum import Enum
from io import BytesIO
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeIs

from litellm._logging import verbose_proxy_logger
from litellm.integrations.clickhouse.clickhouse_spend_logger import spend_log_row_from_payload
from litellm.integrations.custom_logger import CustomLogger
from litellm.tracing.types import SpendLogPayload

MAX_EVENT_BYTES: Final = 1024 * 1024
MAX_BUFFER_BYTES: Final = 32 * 1024 * 1024
MAX_BUFFER_EVENTS: Final = 1000
MAX_BATCH_BYTES: Final = 4 * 1024 * 1024
SHUTDOWN_SECONDS: Final = 3.0
_PAYLOAD: Final = TypeAdapter(SpendLogPayload)


class ExportFailure(Enum):
    TOO_LARGE = "record exceeds the export budget"
    INVALID = "record cannot be serialized"


def _is_mapping(
    value: object,
) -> TypeIs[Mapping[object, object]]:  # guard-ok: bounds arbitrary callback mappings before validation
    return isinstance(value, Mapping)


def _is_sequence(
    value: object,
) -> TypeIs[Sequence[object]]:  # guard-ok: bounds arbitrary callback sequences before validation
    return isinstance(value, (tuple, list))


def _check_size(value: object, remaining: int, depth: int = 0) -> int | ExportFailure:
    if remaining <= 0 or depth > 32:
        return ExportFailure.TOO_LARGE
    if isinstance(value, str):
        if len(value) > remaining:
            return ExportFailure.TOO_LARGE
        try:
            return remaining - len(value.encode())
        except UnicodeError:
            return ExportFailure.INVALID
    if _is_mapping(value):
        return _check_sequence(value.items(), remaining, depth)
    if _is_sequence(value):
        return _check_sequence(value, remaining, depth)
    return remaining - 32


def _check_sequence(values: Iterable[object], remaining: int, depth: int) -> int | ExportFailure:
    budget = remaining  # rebind-ok: consumes a finite serialization budget
    for value in values:
        match _check_size(value, budget - 8, depth + 1):
            case ExportFailure() as failure:
                return failure
            case int() as checked:
                budget = checked
    if budget < 0:
        return ExportFailure.TOO_LARGE
    return budget


def encode_record(value: Mapping[str, object]) -> bytes | ExportFailure:
    checked: Final = _check_size(value, MAX_EVENT_BYTES)
    if isinstance(checked, ExportFailure):
        return checked
    try:
        with BytesIO() as output:
            parts: Final = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(
                dict(value)
            )
            for encoded in (part.encode() for part in parts):
                if output.tell() + len(encoded) > MAX_EVENT_BYTES:
                    return ExportFailure.TOO_LARGE
                output.write(encoded)
            return output.getvalue()
    except (ValueError, TypeError, OverflowError, RecursionError):
        return ExportFailure.INVALID


class LensExporter(CustomLogger):
    def __init__(self, client: httpx.AsyncClient, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        super().__init__()
        self.client: Final = client
        self.sleep: Final = sleep
        self.queue: Final[deque[bytes]] = deque()  # mutable-ok: bounded producer-consumer queue
        self.wake: Final = asyncio.Event()
        self.closed = False
        self.buffered_bytes = 0
        self.buffered_events = 0
        self.rows_written = 0
        self.rows_dropped = 0
        self.last_error = ""
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    def enqueue(self, record: bytes) -> bool:
        if (
            self.closed
            or len(record) > MAX_EVENT_BYTES
            or self.buffered_events >= MAX_BUFFER_EVENTS
            or self.buffered_bytes + len(record) > MAX_BUFFER_BYTES
        ):
            self.rows_dropped += 1
            return False
        self.queue.append(record)
        self.buffered_events += 1
        self.buffered_bytes += len(record)
        self.wake.set()
        return True

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._log(kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._log(kwargs)

    def _log(self, kwargs: Mapping[str, object]) -> None:
        raw: Final = kwargs.get("standard_logging_object")
        if raw is None or self.closed:
            return
        if self.buffered_events >= MAX_BUFFER_EVENTS or self.buffered_bytes >= MAX_BUFFER_BYTES:
            self.rows_dropped += 1
            return
        try:
            checked: Final = _check_size(raw, MAX_EVENT_BYTES)
            if isinstance(checked, ExportFailure):
                self.rows_dropped += 1
                self._warn(checked.value)
                return
            payload: Final = _PAYLOAD.validate_python(raw)
            if str(payload.get("call_type", "")).startswith(("/v1/traces", "/v1/logs")):
                return
            row: Final = spend_log_row_from_payload(payload, kwargs)
            record: Final = encode_record(row)
            if isinstance(record, ExportFailure):
                self.rows_dropped += 1
                self._warn(record.value)
                return
            self.enqueue(record)
        except ValidationError as error:
            self.rows_dropped += 1
            fields: Final = tuple(
                str(issue["loc"][0]) if issue["loc"] else "$"
                for issue in error.errors(include_input=False, include_context=False, include_url=False)[:5]
            )
            self._warn("invalid request record fields: " + ", ".join(fields))
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            self.rows_dropped += 1
            self._warn(type(error).__name__)

    def _warn(self, reason: str) -> None:
        if reason != self.last_error:
            verbose_proxy_logger.warning("Lens request export failed (%s); model requests continue", reason)
            self.last_error = reason

    def _batch(self) -> tuple[bytes, ...]:
        size = 2  # rebind-ok: count bytes in a bounded batch without copying records
        records: Final[deque[bytes]] = deque()  # mutable-ok: finite batch drained from the queue
        while self.queue and size + len(self.queue[0]) + 1 <= MAX_BATCH_BYTES:
            record: Final = self.queue.popleft()
            size += len(record) + 1
            records.append(record)
        return tuple(records)

    async def _send(self, records: tuple[bytes, ...]) -> bool:
        body: Final = b"[" + b",".join(records) + b"]"
        for attempt in range(3):
            try:
                async with self.client.stream(
                    "POST",
                    "/internal/spend",
                    content=body,
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                ) as response:
                    if response.status_code == 204:
                        self.last_error = ""
                        return True
                    if response.status_code not in (429, 502, 503, 504):
                        self._warn(f"HTTP {response.status_code}")
                        return False
            except httpx.HTTPError:
                pass
            if attempt < 2:
                await self.sleep(float(1 << attempt))
        self._warn("retry limit reached")
        return False

    async def _run(self) -> None:
        while not self.closed or self.queue:
            if not self.queue:
                self.wake.clear()
                await self.wake.wait()
                continue
            await self._drain_batch()

    async def _drain_batch(self) -> None:
        batch: Final = self._batch()
        try:
            if await self._send(batch):
                self.rows_written += len(batch)
            else:
                self.rows_dropped += len(batch)
        except asyncio.CancelledError:
            self.rows_dropped += len(batch)
            raise
        finally:
            self.buffered_events -= len(batch)
            self.buffered_bytes -= sum(len(record) for record in batch)

    async def aclose(self) -> None:
        self.closed = True
        self.wake.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=SHUTDOWN_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.task
        self.rows_dropped += len(self.queue)
        self.queue.clear()
        self.buffered_bytes = 0
        self.buffered_events = 0

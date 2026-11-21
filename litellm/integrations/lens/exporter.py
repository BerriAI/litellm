import asyncio
import hashlib
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.clickhouse.clickhouse_spend_logger import spend_log_row_from_payload
from litellm.integrations.clickhouse.spend_log_types import SpendLogPayload, SpendLogRecord
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.lens.content import (
    MAX_CONTENT_JOBS,
    MAX_NATIVE_UPLOADS,
    MAX_PENDING_CONTENT_BYTES,
    MAX_SERIALIZATIONS,
    REFERENCE_KEY,
    UPLOAD_TIMEOUT_SECONDS,
    ContentLimits,
    ContentState,
    ContentStore,
    Owner,
    Stored,
    Unavailable,
    object_name,
    signed_reference,
)

if TYPE_CHECKING:
    from prometheus_client import Counter as PrometheusCounter

MAX_EVENT_BYTES: Final = 1024 * 1024
MAX_BUFFER_BYTES: Final = 32 * 1024 * 1024
MAX_BUFFER_EVENTS: Final = 1000
MAX_BATCH_BYTES: Final = 4 * 1024 * 1024
SHUTDOWN_SECONDS: Final = 3.0
SMALL_CONTENT_BYTES: Final = 64 * 1024
_PAYLOAD: Final = TypeAdapter(SpendLogPayload)
_METADATA: Final = TypeAdapter(dict[str, JsonValue])
_CONTENT_FIELDS: Final = frozenset(("messages", "response"))
_UNEXPORTED_FIELDS: Final = _CONTENT_FIELDS | {"hidden_params"}
_CONTENT_ENCODER: Final = json.JSONEncoder(default=str)


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


def encode_record(value: Mapping[str, object], limit: int = MAX_EVENT_BYTES) -> bytes | ExportFailure:
    checked: Final = _check_size(value, limit)
    if isinstance(checked, ExportFailure):
        return checked
    try:
        with BytesIO() as output:
            parts: Final = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(
                dict(value)
            )
            for encoded in (part.encode() for part in parts):
                if output.tell() + len(encoded) > limit:
                    return ExportFailure.TOO_LARGE
                output.write(encoded)
            return output.getvalue()
    except (ValueError, TypeError, OverflowError, RecursionError):
        return ExportFailure.INVALID


def _empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _content_text(value: object, limit: int) -> str | Unavailable:
    try:
        if _empty(value):
            return ""
        if isinstance(value, str):
            return value if len(value) <= limit else Unavailable.TOO_LARGE
        match _check_size(value, limit):
            case ExportFailure.TOO_LARGE:
                return Unavailable.TOO_LARGE
            case ExportFailure.INVALID:
                return Unavailable.UNSERIALIZABLE
            case int():
                pass
        with StringIO() as output:
            for part in _CONTENT_ENCODER.iterencode(value):
                if output.tell() + len(part) > limit:
                    return Unavailable.TOO_LARGE
                output.write(part)
            return output.getvalue()
    except (ValueError, TypeError, OverflowError, RecursionError, RuntimeError):
        return Unavailable.UNSERIALIZABLE


@dataclass(frozen=True, slots=True)
class _Spill:
    body: str
    size: int
    sha256: str


def _prepare(
    bare: SpendLogRecord, messages: object, response: object, limits: ContentLimits, spill: bool
) -> bytes | _Spill | Unavailable:
    limit: Final = limits.object_bytes if spill else limits.inline_bytes
    text: Final = _content_text(messages, limit)
    if isinstance(text, Unavailable):
        return _unstored(text, spill)
    reply: Final = _content_text(response, limit - len(text))
    if isinstance(reply, Unavailable):
        return _unstored(reply, spill)
    record: Final = encode_record({**bare, "messages": text, "response": reply}, limits.inline_bytes)
    if isinstance(record, bytes):
        return record
    if record is ExportFailure.INVALID:
        return Unavailable.UNSERIALIZABLE
    if not spill:
        return Unavailable.STORAGE_NOT_CONFIGURED
    try:
        body: Final = json.dumps({"messages": text, "response": reply}, ensure_ascii=False)
        data: Final = body.encode()
    except UnicodeEncodeError:
        return Unavailable.UNSERIALIZABLE
    if len(data) > limits.object_bytes:
        return Unavailable.TOO_LARGE
    return _Spill(body=body, size=len(data), sha256=hashlib.sha256(data).hexdigest())


def _unstored(reason: Unavailable, spill: bool) -> Unavailable:
    return Unavailable.STORAGE_NOT_CONFIGURED if reason is Unavailable.TOO_LARGE and not spill else reason


def _owner(row: SpendLogRecord) -> Owner:
    return Owner(team_id=row["team_id"], api_key=row["api_key"], request_id=row["request_id"])


def _with_reference(row: SpendLogRecord, signing_key: bytes, state: ContentState) -> Mapping[str, object]:
    try:
        metadata: Final = _METADATA.validate_json(row["metadata"] or "{}")
    except ValueError:
        return row
    reference: Final = dict(signed_reference(signing_key, _owner(row), state))
    return {**row, "metadata": json.dumps({**metadata, REFERENCE_KEY: reference}, default=str)}


def _exported(raw: Mapping[object, object]) -> Mapping[str, object]:
    return {
        key: value for key, value in raw.items() if isinstance(key, str) and key in _PAYLOAD_KEYS - _UNEXPORTED_FIELDS
    }


_PAYLOAD_KEYS: Final = frozenset(SpendLogPayload.__annotations__)


@dataclass(frozen=True, slots=True)
class ExportCounters:
    written: "PrometheusCounter | None"
    dropped: "PrometheusCounter | None"

    def add_written(self, count: int) -> None:
        if self.written is not None:
            self.written.inc(count)

    def add_dropped(self, count: int) -> None:
        if self.dropped is not None:
            self.dropped.inc(count)


@cache
def lens_export_counters() -> ExportCounters:
    try:
        from prometheus_client import Counter
    except ImportError:
        return ExportCounters(written=None, dropped=None)
    return ExportCounters(
        written=Counter(
            "litellm_lens_export_rows_written_total",
            "Request records the Lens exporter delivered to the Lens service",
        ),
        dropped=Counter(
            "litellm_lens_export_rows_dropped_total",
            "Request records the Lens exporter dropped before the Lens service accepted them",
        ),
    )


class _RawReservation:
    def __init__(self, exporter: "LensExporter", raw: int) -> None:
        self.exporter: Final = exporter
        self.raw: Final = raw
        self.holders = 1
        exporter.pending_content_bytes += raw

    def hold(self) -> None:
        self.holders += 1

    def release(self) -> None:
        self.holders -= 1
        if self.holders == 0:
            self.exporter.pending_content_bytes -= self.raw


class LensExporter(CustomLogger):
    def __init__(
        self,
        client: httpx.AsyncClient,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        counters: ExportCounters | None = None,
        content_store: ContentStore | None = None,
        signing_key: bytes = b"",
        limits: ContentLimits = ContentLimits(),
        upload_timeout: float = UPLOAD_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self.client: Final = client
        self.sleep: Final = sleep
        self.counters: Final = counters if counters is not None else lens_export_counters()
        self.content_store: Final = content_store
        self.signing_key: Final = signing_key
        self.limits: Final = limits
        self.upload_timeout: Final = upload_timeout
        self.serialize_slots: Final = asyncio.Semaphore(MAX_SERIALIZATIONS)
        self.content_tasks: Final[set[asyncio.Task[bytes | ContentState]]] = set()  # mutable-ok: bounded content jobs
        self.native_uploads = 0
        self.pending_content_bytes = 0
        self.closing = False
        self.last_content_error = ""
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

    def _wrote(self, count: int) -> None:
        self.rows_written += count
        self.counters.add_written(count)

    def _drop(self, count: int) -> None:
        self.rows_dropped += count
        self.counters.add_dropped(count)

    def enqueue(self, record: bytes) -> bool:
        if (
            self.closed
            or len(record) > MAX_EVENT_BYTES
            or self.buffered_events >= MAX_BUFFER_EVENTS
            or self.buffered_bytes + len(record) > MAX_BUFFER_BYTES
        ):
            self._drop(1)
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
            self._drop(1)
            return
        try:
            if _is_mapping(raw):
                checked: Final = _check_size(_exported(raw), MAX_EVENT_BYTES)
                if isinstance(checked, ExportFailure):
                    self._reject(checked)
                    return
            payload: Final = _PAYLOAD.validate_python(
                {key: value for key, value in raw.items() if key not in _CONTENT_FIELDS} if _is_mapping(raw) else raw
            )
            if str(payload.get("call_type", "")).startswith(("/v1/traces", "/v1/logs")):
                return
            bare: Final = spend_log_row_from_payload(payload, kwargs)
            messages: Final = raw.get("messages") if _is_mapping(raw) else None
            response: Final = raw.get("response") if _is_mapping(raw) else None
            if litellm.turn_off_message_logging is True or (_empty(messages) and _empty(response)):
                self._export(bare)
                return
            if isinstance(_check_size((messages, response), SMALL_CONTENT_BYTES), int):
                prepared: Final = _prepare(bare, messages, response, self.limits, spill=False)
                if isinstance(prepared, bytes):
                    self.enqueue(prepared)
                    return
            self._admit(bare, messages, response)
        except ValidationError as error:
            self._drop(1)
            fields: Final = tuple(
                str(issue["loc"][0]) if issue["loc"] else "$"
                for issue in error.errors(include_input=False, include_context=False, include_url=False)[:5]
            )
            self._log_failure("invalid request record fields: " + ", ".join(fields))
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            self._drop(1)
            self._log_failure(type(error).__name__)

    def _reject(self, failure: ExportFailure) -> None:
        self._drop(1)
        self._log_failure(failure.value)

    def _export(self, row: Mapping[str, object]) -> None:
        record: Final = encode_record(row)
        if isinstance(record, ExportFailure):
            self._reject(record)
            return
        self.enqueue(record)

    def _admit(self, bare: SpendLogRecord, messages: object, response: object) -> None:
        spill: Final = self.content_store is not None
        limit: Final = self.limits.object_bytes if spill else self.limits.inline_bytes
        measured: Final = _check_size((messages, response), limit)
        if isinstance(measured, ExportFailure):
            too_large: Final = Unavailable.TOO_LARGE if spill else Unavailable.STORAGE_NOT_CONFIGURED
            self._finish(bare, too_large if measured is ExportFailure.TOO_LARGE else Unavailable.UNSERIALIZABLE)
            return
        raw: Final = limit - measured
        if (
            self.closing
            or len(self.content_tasks) >= MAX_CONTENT_JOBS
            or (spill and self.native_uploads >= MAX_NATIVE_UPLOADS)
            or self.pending_content_bytes + raw > MAX_PENDING_CONTENT_BYTES
        ):
            self._finish(bare, Unavailable.OVERLOADED)
            return
        ticket: Final = _RawReservation(self, raw)
        task: Final = asyncio.create_task(self._content_outcome(bare, messages, response, ticket))
        self.content_tasks.add(task)
        task.add_done_callback(partial(self._content_done, bare, ticket))

    async def _content_outcome(
        self, bare: SpendLogRecord, messages: object, response: object, ticket: "_RawReservation"
    ) -> bytes | ContentState:
        prepared: Final = await self._serialize(bare, messages, response, ticket)
        if isinstance(prepared, _Spill):
            return await self._upload(bare, prepared)
        return prepared

    def _content_done(
        self, bare: SpendLogRecord, ticket: "_RawReservation", task: "asyncio.Task[bytes | ContentState]"
    ) -> None:
        self.content_tasks.discard(task)
        ticket.release()
        outcome: Final = (
            Unavailable.CANCELLED
            if task.cancelled()
            else Unavailable.UPLOAD_FAILED
            if task.exception() is not None
            else task.result()
        )
        if isinstance(outcome, bytes):
            self.enqueue(outcome)
            return
        self._finish(bare, outcome)

    async def _serialize(
        self, bare: SpendLogRecord, messages: object, response: object, ticket: "_RawReservation"
    ) -> bytes | _Spill | Unavailable:
        await self.serialize_slots.acquire()
        try:
            work: Final = asyncio.get_running_loop().run_in_executor(
                None, _prepare, bare, messages, response, self.limits, self.content_store is not None
            )
        except BaseException:
            self.serialize_slots.release()
            raise
        ticket.hold()
        work.add_done_callback(partial(self._serialized, ticket))
        return await asyncio.shield(work)

    def _serialized(self, ticket: "_RawReservation", work: "asyncio.Future[bytes | _Spill | Unavailable]") -> None:
        self.serialize_slots.release()
        ticket.release()
        if not work.cancelled():
            work.exception()

    async def _upload(self, bare: SpendLogRecord, spill: _Spill) -> ContentState:
        if self.content_store is None:
            return Unavailable.STORAGE_NOT_CONFIGURED
        name: Final = object_name(_owner(bare), bare["start_time"], spill.sha256)
        if name is None:
            return Unavailable.UNSERIALIZABLE
        key: Final = self.content_store.object_key(name)
        if key is None:
            return Unavailable.STORAGE_NOT_CONFIGURED
        if (
            self.native_uploads >= MAX_NATIVE_UPLOADS
            or self.pending_content_bytes + spill.size > MAX_PENDING_CONTENT_BYTES
        ):
            return Unavailable.OVERLOADED
        self.native_uploads += 1
        self.pending_content_bytes += spill.size
        try:
            native: Final = asyncio.ensure_future(self.content_store.upload(key, spill.body))
        except Exception:  # noqa: BLE001  # a store that cannot start an upload leaves the content unavailable
            self.native_uploads -= 1
            self.pending_content_bytes -= spill.size
            return Unavailable.UPLOAD_FAILED
        native.add_done_callback(partial(self._uploaded, spill.size))
        done, _ = await asyncio.wait((native,), timeout=self.upload_timeout)
        if native not in done or native.cancelled() or native.exception() is not None or native.result() is not True:
            return Unavailable.UPLOAD_FAILED
        return Stored(object_key=key, size=spill.size, sha256=spill.sha256)

    def _uploaded(self, size: int, native: "asyncio.Future[bool]") -> None:
        self.native_uploads -= 1
        self.pending_content_bytes -= size
        if not native.cancelled():
            native.exception()

    def _finish(self, bare: SpendLogRecord, state: ContentState) -> None:
        record: Final = encode_record(_with_reference(bare, self.signing_key, state))
        if isinstance(record, ExportFailure):
            self._reject(record)
            return
        self.enqueue(record)
        reason: Final = state.value if isinstance(state, Unavailable) else ""
        if reason and reason != self.last_content_error:
            verbose_proxy_logger.warning(
                "Agent tracing store lens exported a request record without its content (%s)", reason
            )
        self.last_content_error = reason

    def _log_failure(self, reason: str) -> None:
        if reason != self.last_error:
            verbose_proxy_logger.error(
                "Agent tracing store lens dropped request records (%s); model requests continue", reason
            )
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
                        self._log_failure(f"HTTP {response.status_code}")
                        return False
            except httpx.HTTPError:
                pass
            if attempt < 2:
                await self.sleep(float(1 << attempt))
        self._log_failure("retry limit reached")
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
                self._wrote(len(batch))
            else:
                self._drop(len(batch))
        except asyncio.CancelledError:
            self._drop(len(batch))
            raise
        finally:
            self.buffered_events -= len(batch)
            self.buffered_bytes -= sum(len(record) for record in batch)

    async def aclose(self) -> None:
        self.closing = True
        jobs: Final = tuple(self.content_tasks)
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self.closed = True
        self.wake.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=SHUTDOWN_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.task
        self._drop(len(self.queue))
        self.queue.clear()
        self.buffered_bytes = 0
        self.buffered_events = 0

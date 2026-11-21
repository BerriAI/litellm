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
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

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
from litellm.integrations.lens.ledger import LedgerHandoff, Release

if TYPE_CHECKING:
    from prometheus_client import Counter as PrometheusCounter

MAX_EVENT_BYTES: Final = 1024 * 1024
MAX_BUFFER_BYTES: Final = 32 * 1024 * 1024
MAX_BUFFER_EVENTS: Final = 1000
MAX_BATCH_BYTES: Final = 4 * 1024 * 1024
SHUTDOWN_SECONDS: Final = 3.0
SMALL_CONTENT_BYTES: Final = 64 * 1024
MAX_PENDING_COMMITS: Final = 10_000
_HELD_ROW_BYTES: Final = 4096
_CONTENT_STATE: Final = "_lens_content_state"
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


def _texts(messages: object, response: object, limits: ContentLimits, spill: bool) -> tuple[str, str] | Unavailable:
    limit: Final = limits.object_bytes if spill else limits.inline_bytes
    text: Final = _content_text(messages, limit)
    if isinstance(text, Unavailable):
        return _unstored(text, spill)
    reply: Final = _content_text(response, limit - len(text))
    if isinstance(reply, Unavailable):
        return _unstored(reply, spill)
    return text, reply


def _prepare(
    bare: SpendLogRecord, messages: object, response: object, limits: ContentLimits, spill: bool
) -> bytes | _Spill | Unavailable:
    texts: Final = _texts(messages, response, limits, spill)
    return texts if isinstance(texts, Unavailable) else _packed(bare, *texts, limits, spill)


def _packed(
    bare: Mapping[str, object], text: str, reply: str, limits: ContentLimits, spill: bool
) -> bytes | _Spill | Unavailable:
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


def _held(bare: SpendLogRecord, content: tuple[str, str] | Unavailable) -> tuple[Mapping[str, object], int]:
    """A ledger-backed record waiting for its commit: content serialized, nothing uploaded or signed yet."""
    if isinstance(content, Unavailable):
        return {**bare, _CONTENT_STATE: content.value}, _HELD_ROW_BYTES
    text, reply = content
    return {**bare, "messages": text, "response": reply}, _HELD_ROW_BYTES + len(text) + len(reply)


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


_DEFAULT_LIMITS: Final = ContentLimits()
_PENDING: Final = TypeAdapter(SpendLogRecord)
_Outcome: TypeAlias = bytes | ContentState | tuple[str, str]
_Result: Final = TypeVar("_Result")


def _confirmed(native: "asyncio.Future[bool]") -> bool:
    if not native.done() or native.cancelled():
        return False
    if native.exception() is not None:
        return False
    return native.result() is True


class _RawReservation:
    def __init__(self, raw: int, released: Callable[[int], None]) -> None:
        self.raw: Final = raw
        self.released: Final = released
        self.holders = 1

    def hold(self) -> None:
        self.holders += 1

    def release(self) -> None:
        self.holders -= 1
        if self.holders == 0:
            self.released(self.raw)


class LensExporter(CustomLogger):
    def __init__(
        self,
        client: httpx.AsyncClient,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        counters: ExportCounters | None = None,
        content_store: ContentStore | None = None,
        signing_key: bytes = b"",
        limits: ContentLimits = _DEFAULT_LIMITS,
        upload_timeout: float = UPLOAD_TIMEOUT_SECONDS,
        ledger: LedgerHandoff | None = None,
        ledger_key: Callable[[Mapping[str, object], object], str | None] | None = None,
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
        self.content_tasks: Final[set[asyncio.Task[_Outcome]]] = set()  # mutable-ok: bounded content jobs
        self.ledger: Final = ledger
        self.ledger_key: Final = ledger_key
        self.commits: Final[deque[Mapping[str, object]]] = deque()  # mutable-ok: bounded confirmed deliveries
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
        self._log(kwargs, response_obj)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._log(kwargs, response_obj)

    def _log(self, kwargs: Mapping[str, object], response_obj: object = None) -> None:
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
            key: Final = self._ledger_request_id(kwargs, response_obj)
            if key is not False:
                self._log_ledger_backed(key, bare, messages, response)
                return
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

    def _ledger_request_id(self, kwargs: Mapping[str, object], response_obj: object) -> str | None | Literal[False]:
        """The request id a ledger commit will confirm, None when the ledger state is unknown, False when no commit follows."""
        if self.ledger is None or self.ledger_key is None:
            return False
        try:
            key: Final = self.ledger_key(kwargs, response_obj)
        except Exception as error:  # noqa: BLE001  # an unknown ledger state never exports a record early
            self._log_failure(f"ledger state unknown: {type(error).__name__}")
            return None
        return False if key is None else key

    def _log_ledger_backed(self, key: str | None, bare: SpendLogRecord, messages: object, response: object) -> None:
        if key is None or self.ledger is None:
            self._drop(1)
            return
        if litellm.turn_off_message_logging is True or (_empty(messages) and _empty(response)):
            self._hold(key, bare, ("", ""))
            return
        if isinstance(_check_size((messages, response), SMALL_CONTENT_BYTES), int):
            self._hold(key, bare, _texts(messages, response, self.limits, self.content_store is not None))
            return
        self._admit(bare, messages, response, key)

    def _hold(self, key: str, bare: SpendLogRecord, content: tuple[str, str] | Unavailable) -> None:
        if self.ledger is None or self.closed:
            self._drop(1)
            return
        row, size = _held(bare, content)
        self._release(self.ledger.hold(key, row, size))

    def ledger_committed(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Admits confirmed deliveries for the export task; the ledger writer that reports them never waits."""
        if self.ledger is None or self.closed:
            return
        if len(self.commits) + len(rows) > MAX_PENDING_COMMITS:
            self._drop(len(rows))
            self._log_failure("too many confirmed deliveries awaiting export")
            return
        self.commits.extend(rows)
        self.wake.set()

    def _apply_commits(self) -> None:
        if self.ledger is None or not self.commits:
            return
        rows: Final = tuple(self.commits)
        self.commits.clear()
        self._release(self.ledger.commit(rows))

    def _release(self, release: Release) -> None:
        for row in release.ready:
            self._emit_finalized(row)
        if release.dropped:
            self._drop(release.dropped)
            self._log_failure("records whose ledger commit was not confirmed in time")

    def _emit_finalized(self, row: Mapping[str, object]) -> None:
        """The one exit for a record whose committed accounting owner and cost are final.

        Content handling runs only here, so an object key and its signed reference always name the
        committed owner, never the callback's.
        """
        state: Final = row.get(_CONTENT_STATE)
        final: Final = _PENDING.validate_python({key: value for key, value in row.items() if key != _CONTENT_STATE})
        if isinstance(state, str):
            self._finish(final, Unavailable(state))
            return
        if not final["messages"] and not final["response"]:
            self._export(final)
            return
        record: Final = encode_record(final, self.limits.inline_bytes)
        if isinstance(record, bytes):
            self.enqueue(record)
            return
        if record is ExportFailure.INVALID or self.content_store is None:
            unavailable: Final = (
                Unavailable.UNSERIALIZABLE if record is ExportFailure.INVALID else Unavailable.STORAGE_NOT_CONFIGURED
            )
            self._finish({**final, "messages": "", "response": ""}, unavailable)
            return
        self._admit_packing(final)

    def _admit_packing(self, final: SpendLogRecord) -> None:
        reserved: Final = len(final["messages"]) + len(final["response"])
        bare: Final[SpendLogRecord] = {**final, "messages": "", "response": ""}
        if self._saturated(True, reserved):
            self._finish(bare, Unavailable.OVERLOADED)
            return
        self.pending_content_bytes += reserved
        ticket: Final = _RawReservation(reserved, self._release_raw)
        work: Final = partial(_packed, bare, final["messages"], final["response"], self.limits, True)
        task: Final = asyncio.create_task(self._packing_outcome(bare, work, ticket))
        self.content_tasks.add(task)
        task.add_done_callback(partial(self._content_done, bare, ticket, None))

    async def _packing_outcome(
        self, bare: SpendLogRecord, work: Callable[[], bytes | _Spill | Unavailable], ticket: "_RawReservation"
    ) -> _Outcome:
        prepared: Final = await self._run_serialization(work, ticket)
        if isinstance(prepared, _Spill):
            return await self._upload(bare, prepared)
        return prepared

    def _reject(self, failure: ExportFailure) -> None:
        self._drop(1)
        self._log_failure(failure.value)

    def _export(self, row: Mapping[str, object]) -> None:
        record: Final = encode_record(row)
        if isinstance(record, ExportFailure):
            self._reject(record)
            return
        self.enqueue(record)

    def _admit(self, bare: SpendLogRecord, messages: object, response: object, key: str | None = None) -> None:
        spill: Final = self.content_store is not None
        limit: Final = self.limits.object_bytes if spill else self.limits.inline_bytes
        measured: Final = _check_size((messages, response), limit)
        if isinstance(measured, ExportFailure):
            too_large: Final = Unavailable.TOO_LARGE if spill else Unavailable.STORAGE_NOT_CONFIGURED
            self._settle(key, bare, too_large if measured is ExportFailure.TOO_LARGE else Unavailable.UNSERIALIZABLE)
            return
        raw: Final = limit - measured
        if self._saturated(spill, raw):
            self._settle(key, bare, Unavailable.OVERLOADED)
            return
        self.pending_content_bytes += raw
        ticket: Final = _RawReservation(raw, self._release_raw)
        task: Final = asyncio.create_task(self._content_outcome(bare, messages, response, ticket, key))
        self.content_tasks.add(task)
        task.add_done_callback(partial(self._content_done, bare, ticket, key))

    def _settle(self, key: str | None, bare: SpendLogRecord, outcome: _Outcome) -> None:
        if key is not None:
            self._hold(key, bare, outcome if isinstance(outcome, (tuple, Unavailable)) else Unavailable.UPLOAD_FAILED)
        elif isinstance(outcome, bytes):
            self.enqueue(outcome)
        elif not isinstance(outcome, tuple):
            self._finish(bare, outcome)

    async def _content_outcome(
        self,
        bare: SpendLogRecord,
        messages: object,
        response: object,
        ticket: "_RawReservation",
        key: str | None,
    ) -> _Outcome:
        spill: Final = self.content_store is not None
        if key is not None:
            return await self._run_serialization(partial(_texts, messages, response, self.limits, spill), ticket)
        prepared: Final = await self._run_serialization(
            partial(_prepare, bare, messages, response, self.limits, spill), ticket
        )
        if isinstance(prepared, _Spill):
            return await self._upload(bare, prepared)
        return prepared

    def _content_done(
        self, bare: SpendLogRecord, ticket: "_RawReservation", key: str | None, task: "asyncio.Task[_Outcome]"
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
        self._settle(key, bare, outcome)

    async def _run_serialization(self, work: Callable[[], _Result], ticket: "_RawReservation") -> _Result:
        await self.serialize_slots.acquire()
        try:
            future: Final = asyncio.get_running_loop().run_in_executor(None, work)
        except BaseException:
            self.serialize_slots.release()
            raise
        ticket.hold()
        future.add_done_callback(partial(self._serialized, ticket))
        return await asyncio.shield(future)

    def _serialized(self, ticket: "_RawReservation", work: "asyncio.Future[_Result]") -> None:
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
        await asyncio.wait((native,), timeout=self.upload_timeout)
        if not _confirmed(native):
            return Unavailable.UPLOAD_FAILED
        return Stored(object_key=key, size=spill.size, sha256=spill.sha256)

    def _saturated(self, spill: bool, raw: int) -> bool:
        if self.closing or len(self.content_tasks) >= MAX_CONTENT_JOBS:
            return True
        if spill and self.native_uploads >= MAX_NATIVE_UPLOADS:
            return True
        return self.pending_content_bytes + raw > MAX_PENDING_CONTENT_BYTES

    def _release_raw(self, raw: int) -> None:
        self.pending_content_bytes -= raw

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
        while not self.closed or self.queue or self.commits:
            if self.commits:
                self._apply_commits()
                continue
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
        if self.ledger is not None:
            self._apply_commits()
            self._release(self.ledger.drain())
            leftovers: Final = tuple(self.content_tasks)
            for job in leftovers:
                job.cancel()
            await asyncio.gather(*leftovers, return_exceptions=True)
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

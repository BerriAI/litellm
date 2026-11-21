"""
Reports of committed ledger deliveries, for observers that act only once accounting is final.

Each report is a bounded summary projected from the accepted delivery's accounting payload. Reports
leave the ledger path only after its transaction committed, and an observer failure never reaches the
write that triggered it.
"""

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger

MAX_SUMMARY_BYTES: Final = 16 * 1024
MAX_FIELD_BYTES: Final = 512
MAX_PENDING_NATIVE_SUMMARIES: Final = 10_000
ACCOUNTING_FIELDS: Final = (
    "request_id",
    "call_type",
    "api_key",
    "user",
    "team_id",
    "organization_id",
    "end_user",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "model",
    "model_group",
    "custom_llm_provider",
    "status",
    "request_duration_ms",
)
_SUMMARY: Final = TypeAdapter(dict[str, object])


@runtime_checkable
class LedgerCommitObserver(Protocol):
    def ledger_committed(self, rows: Sequence[Mapping[str, object]]) -> None: ...


def _fits(value: object) -> bool:
    if isinstance(value, str):
        return len(value) <= MAX_FIELD_BYTES
    return value is None or isinstance(value, (int, float, bool))


def unreportable(delivery_id: str, request_id: object) -> Mapping[str, object]:
    """Stands in for a summary too large to report, so observers still count the committed delivery."""
    return {
        "delivery_id": delivery_id[:MAX_FIELD_BYTES],
        "request_id": request_id if isinstance(request_id, str) and len(request_id) <= MAX_FIELD_BYTES else None,
        "unreportable": True,
    }


def committed_summary(delivery_id: str, accounting: Mapping[str, object], request_id: object) -> Mapping[str, object]:
    """The canonical accounting fields of an accepted delivery, keyed by its delivery and request ids."""
    canonical_request: Final = accounting.get("request_id", request_id)
    values: Final = tuple(accounting.get(field) for field in ACCOUNTING_FIELDS)
    if len(delivery_id) > MAX_FIELD_BYTES or not all(_fits(value) for value in (*values, canonical_request)):
        return unreportable(delivery_id, canonical_request)
    return {**dict(zip(ACCOUNTING_FIELDS, values)), "request_id": canonical_request, "delivery_id": delivery_id}


def _observers() -> tuple[LedgerCommitObserver, ...]:
    unique: Final = {
        id(callback): callback for callback in tuple(litellm.callbacks) if isinstance(callback, LedgerCommitObserver)
    }
    return tuple(unique.values())


def has_commit_observers() -> bool:
    return any(isinstance(callback, LedgerCommitObserver) for callback in tuple(litellm.callbacks))


def notify_ledger_commit(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    for observer in _observers():
        try:
            observer.ledger_committed(rows)
        except Exception as error:  # noqa: BLE001  # an observer never fails the ledger write it heard about
            verbose_proxy_logger.warning("Spend tracking - a ledger commit observer failed: %s", error)


def report_ledger_commit(prepare: Callable[[], Sequence[Mapping[str, object]]]) -> None:
    """Prepares and dispatches a commit report, never failing the ledger write that already committed."""
    try:
        if has_commit_observers():
            notify_ledger_commit(prepare())
    except Exception as error:  # noqa: BLE001  # accounting is final; a report failure only loses the report
        verbose_proxy_logger.warning("Spend tracking - could not report a committed ledger write: %s", error)


class NativeCommitReports:
    """Moves the native writer's commit summaries onto the event loop within a fixed budget.

    The native writer calls `receive` on its own thread with the GIL held, so it only admits the
    texts and schedules them. Decoding and every observer run later on the loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, max_pending: int = MAX_PENDING_NATIVE_SUMMARIES) -> None:
        self.loop: Final = loop
        self.max_pending: Final = max_pending
        self.lock: Final = threading.Lock()
        self.pending = 0
        self.dropped = 0

    def receive(self, texts: Sequence[str]) -> None:
        if not has_commit_observers():
            return
        with self.lock:
            if self.loop.is_closed() or self.pending + len(texts) > self.max_pending:
                self.dropped += len(texts)
                return
            self.pending += len(texts)
        try:
            self.loop.call_soon_threadsafe(self._deliver, tuple(texts))
        except RuntimeError:
            with self.lock:
                self.pending -= len(texts)
                self.dropped += len(texts)

    def _deliver(self, texts: tuple[str, ...]) -> None:
        with self.lock:
            self.pending -= len(texts)
        notify_ledger_commit(tuple(row for row in (_decoded(text) for text in texts) if row is not None))


def _decoded(text: str) -> Mapping[str, object] | None:
    if len(text) > MAX_SUMMARY_BYTES:
        return None
    try:
        return _SUMMARY.validate_json(text)
    except ValidationError:
        return None

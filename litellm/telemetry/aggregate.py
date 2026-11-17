import logging
import time
from collections.abc import Callable
from typing import Final, TypeVar

from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, UIEvent
from litellm.telemetry.report import AttemptKey, AttemptMetrics, Report, RequestKey, RequestMetrics
from litellm.telemetry.sink import Exporter, ExportOutcome

DEFAULT_MAX_ROWS: Final = 2000
QUIET_RETRIES: Final = 3

_LOGGER: Final = logging.getLogger(__name__)

_Key: Final = TypeVar("_Key")
_Row: Final = TypeVar("_Row")


class AggregatingSink:
    """Folds records into per-window rows and hands each window to an exporter as one ``Report``"""

    def __init__(
        self,
        exporter: Exporter,
        *,
        clock: Callable[[], float] = time.time,
        max_rows: int = DEFAULT_MAX_ROWS,
        logger: logging.Logger = _LOGGER,
    ) -> None:
        self._exporter: Final = exporter
        self._clock: Final = clock
        self._max_rows: Final = max_rows
        self._logger: Final = logger
        self._instance: InstanceInfo | None = None
        self._window_start: float = clock()
        self._requests: dict[RequestKey, RequestMetrics] = {}  # mutable-ok: bounded fold, drained per flush
        self._attempts: dict[AttemptKey, AttemptMetrics] = {}  # mutable-ok: bounded fold, drained per flush
        self._ui_events: dict[UIEvent, int] = {}  # mutable-ok: bounded fold, drained per flush
        self._dropped: int = 0
        self._flushing: bool = False
        self._consecutive_retries: int = 0

    def _row_count(self) -> int:
        return len(self._requests) + len(self._attempts) + len(self._ui_events)

    def _row(
        self,
        rows: dict[_Key, _Row],  # mutable-ok: one of this sink's bounded row folds, shared cap check
        key: _Key,
        empty: Callable[[], _Row],
        weight: int,
    ) -> _Row | None:
        existing: Final = rows.get(key)
        if existing is not None:
            return existing
        if self._row_count() >= self._max_rows:
            self._dropped += weight
            return None
        row: Final = empty()
        rows[key] = row  # rebind-ok: inserts the new row into this sink's own bounded fold
        return row

    def _add_ui_events(self, event: UIEvent, count: int) -> None:
        existing: Final = self._ui_events.get(event)
        if existing is None and self._row_count() >= self._max_rows:
            self._dropped += count
            return
        self._ui_events[event] = (existing or 0) + count

    def set_instance(self, info: InstanceInfo) -> None:
        self._instance = info

    def record_request(self, record: RequestRecord) -> None:
        row: Final = self._row(self._requests, RequestKey.of(record), RequestMetrics.empty, 1)
        if row is not None:
            row.add(record)

    def record_attempt(self, record: AttemptRecord) -> None:
        row: Final = self._row(self._attempts, AttemptKey.of(record), AttemptMetrics.empty, 1)
        if row is not None:
            row.add(record)

    def record_ui_event(self, event: UIEvent) -> None:
        self._add_ui_events(event, 1)

    def _restore(self, report: Report) -> None:
        self._window_start = report.window_start
        self._dropped += report.dropped_records
        for request_key, request_metrics in report.requests:
            request_row = self._row(self._requests, request_key, RequestMetrics.empty, request_metrics.request_count)
            if request_row is not None:
                request_row.add_metrics(request_metrics)
        for attempt_key, attempt_metrics in report.attempts:
            attempt_row = self._row(self._attempts, attempt_key, AttemptMetrics.empty, attempt_metrics.attempt_count)
            if attempt_row is not None:
                attempt_row.add_metrics(attempt_metrics)
        for event, count in report.ui_events:
            self._add_ui_events(event, count)

    async def flush(self) -> None:
        instance: Final = self._instance
        if instance is None or self._flushing:
            return
        self._flushing = True
        try:
            await self._export_window(instance)
        finally:
            self._flushing = False

    async def _export_window(self, instance: InstanceInfo) -> None:
        window_end: Final = self._clock()
        report: Final = Report(
            instance=instance,
            window_start=self._window_start,
            window_end=window_end,
            requests=tuple(self._requests.items()),
            attempts=tuple(self._attempts.items()),
            ui_events=tuple(self._ui_events.items()),
            dropped_records=self._dropped,
        )
        self._requests, self._attempts, self._ui_events, self._dropped = {}, {}, {}, 0
        self._window_start = window_end
        outcome: Final = await self._exporter.export(report)
        if outcome is not ExportOutcome.RETRY:
            self._consecutive_retries = 0
            return
        self._consecutive_retries += 1
        if self._consecutive_retries >= QUIET_RETRIES:
            self._logger.error(
                "telemetry: report export failed %d times in a row, keeping the window for the next flush",
                self._consecutive_retries,
            )
        self._restore(report)

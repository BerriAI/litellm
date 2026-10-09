import time
from collections.abc import Callable
from typing import Final

from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, UIEvent
from litellm.telemetry.report import AttemptKey, AttemptMetrics, Report, RequestKey, RequestMetrics
from litellm.telemetry.sink import Exporter, ExportOutcome

DEFAULT_MAX_ROWS: Final = 2000


class AggregatingSink:
    """Folds records into per-window rows and hands each window to an exporter as one ``Report``"""

    def __init__(
        self,
        exporter: Exporter,
        *,
        clock: Callable[[], float] = time.time,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self._exporter: Final = exporter
        self._clock: Final = clock
        self._max_rows: Final = max_rows
        self._instance: InstanceInfo | None = None
        self._window_start: float = clock()
        self._requests: dict[RequestKey, RequestMetrics] = {}  # mutable-ok: bounded fold, drained per flush
        self._attempts: dict[AttemptKey, AttemptMetrics] = {}  # mutable-ok: bounded fold, drained per flush
        self._ui_events: dict[UIEvent, int] = {}  # mutable-ok: bounded fold, drained per flush
        self._dropped: int = 0

    def _row_count(self) -> int:
        return len(self._requests) + len(self._attempts) + len(self._ui_events)

    def _has_room_for(self, is_new_row: bool) -> bool:
        if not is_new_row or self._row_count() < self._max_rows:
            return True
        self._dropped += 1
        return False

    def set_instance(self, info: InstanceInfo) -> None:
        self._instance = info

    def record_request(self, record: RequestRecord) -> None:
        key: Final = RequestKey.of(record)
        existing: Final = self._requests.get(key)
        if not self._has_room_for(existing is None):
            return
        self._add_request_metrics(key, RequestMetrics.of(record))

    def record_attempt(self, record: AttemptRecord) -> None:
        key: Final = AttemptKey.of(record)
        existing: Final = self._attempts.get(key)
        if not self._has_room_for(existing is None):
            return
        self._add_attempt_metrics(key, AttemptMetrics.of(record))

    def record_ui_event(self, event: UIEvent) -> None:
        existing: Final = self._ui_events.get(event)
        if not self._has_room_for(existing is None):
            return
        self._add_ui_event_count(event, 1)

    def _add_request_metrics(self, key: RequestKey, metrics: RequestMetrics) -> None:
        existing: Final = self._requests.get(key)
        self._requests[key] = metrics if existing is None else existing.merge(metrics)

    def _add_attempt_metrics(self, key: AttemptKey, metrics: AttemptMetrics) -> None:
        existing: Final = self._attempts.get(key)
        self._attempts[key] = metrics if existing is None else existing.merge(metrics)

    def _add_ui_event_count(self, event: UIEvent, count: int) -> None:
        self._ui_events[event] = self._ui_events.get(event, 0) + count

    def _restore(self, report: Report) -> None:
        self._window_start = report.window_start
        self._dropped += report.dropped_records
        for request_key, request_metrics in report.requests:
            self._add_request_metrics(request_key, request_metrics)
        for attempt_key, attempt_metrics in report.attempts:
            self._add_attempt_metrics(attempt_key, attempt_metrics)
        for event, count in report.ui_events:
            self._add_ui_event_count(event, count)

    async def flush(self) -> None:
        instance: Final = self._instance
        if instance is None:
            return
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
            return
        self._restore(report)

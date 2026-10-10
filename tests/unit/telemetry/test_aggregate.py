import asyncio
import logging
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.histogram import ATTEMPT_BOUNDS, BLOCK_COUNT_BOUNDS, LATENCY_BOUNDS_MS, Histogram
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryGroup,
    TokenCounts,
    UIAction,
    UIEvent,
)
from litellm.telemetry.report import Report, RequestKey
from litellm.telemetry.sink import Exporter, ExportOutcome


@dataclass
class _FakeExporter:
    outcome: ExportOutcome = ExportOutcome.SENT
    reports: tuple[Report, ...] = ()

    async def export(self, report: Report) -> ExportOutcome:
        self.reports = (*self.reports, report)
        return self.outcome


@dataclass
class _Clock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now


_INSTANCE: Final = InstanceInfo(instance_id="i", litellm_version="1.0.0", groups=frozenset(TelemetryGroup))


def _request(
    *, provider: str = "openai", latency_ms: float = 80.0, tokens: TokenCounts = TokenCounts(), attempts: int = 1
) -> RequestRecord:
    return RequestRecord(
        endpoint="/chat/completions",
        stream=False,
        litellm_status=StatusClass.SUCCESS,
        latency_to_first_byte_ms=latency_ms,
        provider=provider,
        provider_status=StatusClass.SUCCESS,
        provider_attempts=attempts,
        tokens=tokens,
        blocks=BlockCounts(total=2, by_type=((BlockType.TEXT, 2),)),
        header_keys=frozenset({"anthropic-beta"}),
    )


def _histogram(bounds: tuple[float, ...], *values: float) -> Histogram:
    histogram: Final = Histogram.empty(bounds)
    for value in values:
        histogram.add(value)
    return histogram


def _sink(exporter: Exporter, clock: _Clock, max_rows: int = 100) -> AggregatingSink:
    sink: Final = AggregatingSink(exporter, clock=clock, max_rows=max_rows)
    sink.set_instance(_INSTANCE)
    return sink


@pytest.mark.asyncio
async def test_requests_with_the_same_dimensions_fold_into_one_row() -> None:
    exporter: Final = _FakeExporter()
    clock: Final = _Clock()
    sink: Final = _sink(exporter, clock)
    sink.record_request(_request(latency_ms=80.0, tokens=TokenCounts(input=10, output=3, cache_read=4), attempts=1))
    sink.record_request(_request(latency_ms=600.0, tokens=TokenCounts(input=5, output=2, cache_read=7), attempts=2))
    clock.now = 1060.0
    await sink.flush()

    (report,) = exporter.reports
    assert (report.window_start, report.window_end) == (1000.0, 1060.0)
    ((key, metrics),) = report.requests
    assert key == RequestKey.of(_request())
    assert (metrics.request_count, metrics.input_tokens, metrics.output_tokens, metrics.cache_read_tokens) == (
        2,
        15,
        5,
        11,
    )
    assert metrics.latency_to_first_byte == _histogram(LATENCY_BOUNDS_MS, 80.0, 600.0)
    assert metrics.provider_attempts == _histogram(ATTEMPT_BOUNDS, 1, 2)
    assert metrics.block_count == _histogram(BLOCK_COUNT_BOUNDS, 2, 2)
    assert metrics.block_types == {BlockType.TEXT: 4}
    assert metrics.header_keys == {"anthropic-beta": 2}
    assert metrics.latency_to_headers == Histogram.empty(LATENCY_BOUNDS_MS)


@pytest.mark.asyncio
async def test_each_window_starts_empty_after_a_successful_export() -> None:
    exporter: Final = _FakeExporter()
    clock: Final = _Clock()
    sink: Final = _sink(exporter, clock)
    sink.record_request(_request(provider="openai"))
    sink.record_request(_request(provider="anthropic"))
    await sink.flush()
    clock.now = 1100.0
    await sink.flush()

    first, second = exporter.reports
    assert {key.provider for key, _ in first.requests} == {"openai", "anthropic"}
    assert second.requests == ()
    assert second.window_start == first.window_end


@pytest.mark.asyncio
async def test_a_retryable_export_failure_carries_the_window_into_the_next_report() -> None:
    exporter: Final = _FakeExporter(outcome=ExportOutcome.RETRY)
    clock: Final = _Clock()
    sink: Final = _sink(exporter, clock)
    sink.record_request(_request())
    sink.record_attempt(AttemptRecord(provider="openai", provider_status=StatusClass.SUCCESS, stream=False))
    sink.record_ui_event(UIEvent(page="models", action=UIAction.VIEW))
    await sink.flush()
    sink.record_request(_request())
    sink.record_ui_event(UIEvent(page="models", action=UIAction.VIEW))
    exporter.outcome = ExportOutcome.SENT
    clock.now = 2000.0
    await sink.flush()

    retried: Final = exporter.reports[-1]
    assert retried.window_start == 1000.0
    assert retried.requests[0][1].request_count == 2
    assert retried.attempts[0][1].attempt_count == 1
    assert retried.ui_events == ((UIEvent(page="models", action=UIAction.VIEW), 2),)


@pytest.mark.asyncio
async def test_a_rejected_report_is_not_resent() -> None:
    exporter: Final = _FakeExporter(outcome=ExportOutcome.REJECTED)
    sink: Final = _sink(exporter, _Clock())
    sink.record_request(_request())
    await sink.flush()
    await sink.flush()
    assert exporter.reports[-1].requests == ()


@pytest.mark.asyncio
async def test_new_rows_past_the_cap_are_dropped_and_counted_but_existing_rows_keep_folding() -> None:
    exporter: Final = _FakeExporter()
    sink: Final = _sink(exporter, _Clock(), max_rows=1)
    sink.record_request(_request(provider="openai"))
    sink.record_request(_request(provider="anthropic"))
    sink.record_attempt(AttemptRecord(provider="openai", provider_status=StatusClass.SUCCESS, stream=False))
    sink.record_request(_request(provider="openai"))
    await sink.flush()

    (report,) = exporter.reports
    assert [(key.provider, metrics.request_count) for key, metrics in report.requests] == [("openai", 2)]
    assert report.attempts == ()
    assert report.dropped_records == 2


@pytest.mark.asyncio
async def test_nothing_is_exported_before_the_instance_is_known() -> None:
    exporter: Final = _FakeExporter()
    sink: Final = AggregatingSink(exporter, clock=_Clock())
    sink.record_request(_request())
    await sink.flush()
    assert exporter.reports == ()
    sink.set_instance(_INSTANCE)
    await sink.flush()
    assert exporter.reports[0].requests[0][1].request_count == 1


@dataclass
class _RecordsDuringExport:
    records: tuple[RequestRecord, ...]
    sink: AggregatingSink | None = None
    outcome: ExportOutcome = ExportOutcome.RETRY
    reports: tuple[Report, ...] = ()

    async def export(self, report: Report) -> ExportOutcome:
        self.reports = (*self.reports, report)
        sink: Final = self.sink
        if sink is not None:
            for record in self.records:
                sink.record_request(record)
        return self.outcome


@pytest.mark.asyncio
async def test_a_retried_window_restores_only_up_to_the_row_cap_and_counts_the_rest_as_dropped() -> None:
    exporter: Final = _RecordsDuringExport(records=(_request(provider="bedrock"), _request(provider="vertex_ai")))
    sink: Final = _sink(exporter, _Clock(), max_rows=2)
    exporter.sink = sink
    sink.record_request(_request(provider="openai"))
    sink.record_request(_request(provider="openai"))
    sink.record_request(_request(provider="anthropic"))
    await sink.flush()
    exporter.outcome, exporter.records = ExportOutcome.SENT, ()
    await sink.flush()

    retried: Final = exporter.reports[-1]
    assert [(key.provider, metrics.request_count) for key, metrics in retried.requests] == [
        ("bedrock", 1),
        ("vertex_ai", 1),
    ]
    assert retried.dropped_records == 3


@dataclass
class _BlockingExporter:
    release: asyncio.Event
    reports: tuple[Report, ...] = ()

    async def export(self, report: Report) -> ExportOutcome:
        self.reports = (*self.reports, report)
        await self.release.wait()
        return ExportOutcome.SENT


@pytest.mark.asyncio
async def test_a_flush_while_another_flush_is_exporting_does_nothing() -> None:
    exporter: Final = _BlockingExporter(release=asyncio.Event())
    sink: Final = AggregatingSink(exporter, clock=_Clock())
    sink.set_instance(_INSTANCE)
    sink.record_request(_request())
    first: Final = asyncio.create_task(sink.flush())
    await asyncio.sleep(0)
    sink.record_request(_request())
    await sink.flush()
    exporter.release.set()
    await first

    assert len(exporter.reports) == 1
    await sink.flush()
    assert exporter.reports[-1].requests[0][1].request_count == 1


@pytest.mark.asyncio
async def test_each_retry_past_the_quiet_limit_logs_an_error_until_an_export_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exporter: Final = _FakeExporter(outcome=ExportOutcome.RETRY)
    sink: Final = AggregatingSink(exporter, clock=_Clock(), logger=logging.getLogger("telemetry-test"))
    sink.set_instance(_INSTANCE)
    with caplog.at_level(logging.ERROR, logger="telemetry-test"):
        for _ in range(4):
            await sink.flush()
        exporter.outcome = ExportOutcome.SENT
        await sink.flush()
        exporter.outcome = ExportOutcome.RETRY
        await sink.flush()

    assert [record.getMessage() for record in caplog.records] == [
        "telemetry: report export failed 3 times in a row, keeping the window for the next flush",
        "telemetry: report export failed 4 times in a row, keeping the window for the next flush",
    ]

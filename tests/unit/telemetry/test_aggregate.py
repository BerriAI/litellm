from dataclasses import dataclass
from typing import Final

import pytest

from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.histogram import ATTEMPT_BOUNDS, LATENCY_BOUNDS_MS, Histogram
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
from litellm.telemetry.sink import ExportOutcome


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


def _sink(exporter: _FakeExporter, clock: _Clock, max_rows: int = 100) -> AggregatingSink:
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
    assert metrics.request_count == 2
    assert metrics.tokens == TokenCounts(input=15, output=5, cache_read=11)
    assert metrics.latency_to_first_byte == Histogram.of(LATENCY_BOUNDS_MS, 80.0).merge(
        Histogram.of(LATENCY_BOUNDS_MS, 600.0)
    )
    assert metrics.provider_attempts == Histogram.of(ATTEMPT_BOUNDS, 1).merge(Histogram.of(ATTEMPT_BOUNDS, 2))
    assert metrics.block_types == ((BlockType.TEXT, 4),)
    assert metrics.header_keys == (("anthropic-beta", 2),)
    assert metrics.latency_to_headers.counts == (0,) * (len(LATENCY_BOUNDS_MS) + 1)


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

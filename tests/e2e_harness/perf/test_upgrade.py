from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from cases.small_bodies_5rps import CASE
from memory_readings import RssCapture, RssReading
from setup.observe import Observation, evaluate
from setup.upgrade import (
    MAX_TRAFFIC_SECONDS,
    UpgradeMonitor,
    matched_response_ids,
    postgres_failures,
    upgrade_case,
    upgrade_properties,
)
from setup.upgrade_worker import UpgradeState
from setup.wire import PlanProgress, WireRecord, response_id


def record(seq: int = 0, *, started_at: float = 100) -> WireRecord:
    return WireRecord(
        seq=seq,
        marker=f"request-{seq}",
        caller="key",
        endpoint="chat_completions",
        stream=False,
        blocked=False,
        started_at=started_at,
        status=200,
        header_s=1,
        first_byte_s=1,
        first_token_s=None,
        total_s=1,
        text="answer",
        ok=True,
        mock_delay_s=1,
        call_id=f"call-{seq}",
        error=None,
        response_id=f"response-{seq}",
    )


def test_spend_logs_match_call_ids_when_response_ids_are_transformed() -> None:
    records: Final = (record(0), record(1), record(2))
    rows: Final = (("provider-response-id", "call-0"), ("response-1", "unrelated-call"))
    assert matched_response_ids(records, rows) == frozenset({"response-0"})


def state(phase: str = "schema_ready", *, pending: bool = True) -> UpgradeState:
    return UpgradeState.model_validate(
        {
            "phase": phase,
            "started_at": 0,
            "required_indexes": ["index"],
            "required_migrations": ["migration"],
            "pending_indexes": ["index"] if pending else [],
            "baseline_rows": {"seed-row": "hash"},
        }
    )


class Probe:
    def __init__(self) -> None:
        self.now = 0.0
        self.current = state()
        self.builds: list[int] = []
        self.found: frozenset[str] = frozenset()
        self.releases = 0
        self.metric_checks = 0
        self.errors = False

    def check_metrics(self) -> None:
        self.metric_checks += 1
        assert not self.errors, "Failed PostgreSQL operation"

    def persisted(self, records: tuple[WireRecord, ...]) -> frozenset[str]:
        return frozenset(row.response_id for row in records if row.response_id in self.found and row.response_id)

    def release(self) -> None:
        self.releases += 1

    def monitor(self) -> UpgradeMonitor:
        return UpgradeMonitor(
            state=lambda: self.current,
            check_metrics=self.check_metrics,
            read_activity=lambda: (self.builds, 2),
            read_persisted=self.persisted,
            start_indexes=self.release,
            clock=lambda: self.now,
            wall_clock=lambda: self.now,
        )


def test_stop_drains_issued_requests_without_issuing_the_rest() -> None:
    progress: Final = PlanProgress(100)
    assert progress.take(stop=False)
    assert progress.take(stop=False)
    progress.completed = 1
    assert not progress.take(stop=True)
    assert not progress.drained
    assert not progress.take(stop=False)
    progress.completed = 2
    assert progress.drained
    assert progress.issued == 2


def test_finite_plan_still_waits_for_its_last_response() -> None:
    progress: Final = PlanProgress(1)
    assert progress.take(stop=False)
    assert not progress.take(stop=False)
    assert not progress.drained
    progress.completed = 1
    assert progress.drained


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"id": "chat-id"}, "chat-id"),
        ({"type": "response.created", "response": {"id": "response-id"}}, "response-id"),
        ({"type": "message_start", "message": {"id": "message-id"}}, "message-id"),
        ({"type": "response.output_item.added", "item": {"id": "item-id"}}, None),
        ({"type": "content_block_delta", "delta": {"text": "hello"}}, None),
    ],
)
def test_response_id_uses_response_not_output_item(payload: Mapping[str, object], expected: str | None) -> None:
    assert response_id(payload) == expected


def metrics(*, failure: str = "0", batch: str = "0") -> str:
    return f"""# TYPE litellm_postgres_failed_requests_total counter
litellm_postgres_failed_requests_total{{function_name="query",error_class="Timeout"}} {failure}
# TYPE litellm_batch_write_to_db_failed_requests_total counter
litellm_batch_write_to_db_failed_requests_total{{function_name="flush",error_class="DatabaseError"}} {batch}
"""


def test_failed_sql_and_async_batch_writes_are_both_counted() -> None:
    assert postgres_failures(metrics(failure="2", batch="3")) == 5
    assert postgres_failures(metrics()) == 0


def test_no_error_label_series_is_a_valid_zero() -> None:
    assert (
        postgres_failures("""# TYPE litellm_postgres_failed_requests_total counter
# TYPE litellm_batch_write_to_db_failed_requests_total counter
""")
        == 0
    )


@pytest.mark.parametrize("service", [None, "postgres", "batch_write_to_db"])
def test_multiprocess_metrics_omit_failure_families_until_a_failure(service: str | None) -> None:
    total: Final = """# TYPE litellm_postgres_total_requests_total counter
litellm_postgres_total_requests_total{postgres="postgres"} 1
"""
    failure: Final = (
        f"# TYPE litellm_{service}_failed_requests_total counter\n"
        f'litellm_{service}_failed_requests_total{{function_name="query",error_class="Timeout"}} 1\n'
        if service is not None
        else ""
    )
    assert postgres_failures(total + failure) == (1 if service is not None else 0)


@pytest.mark.parametrize("total", ["0", "-1", "NaN", "+Inf"])
def test_missing_failure_families_require_observed_database_activity(total: str) -> None:
    with pytest.raises(AssertionError):
        _ = postgres_failures(
            "# TYPE litellm_postgres_total_requests_total counter\n"
            f'litellm_postgres_total_requests_total{{postgres="postgres"}} {total}\n'
        )


@pytest.mark.parametrize("body", ["", "# TYPE unrelated counter\n", metrics(failure="NaN"), metrics(batch="-1")])
def test_missing_or_invalid_counters_do_not_pass_as_zero(body: str) -> None:
    with pytest.raises(AssertionError):
        _ = postgres_failures(body)


def test_index_completion_stops_after_the_post_upgrade_window() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    assert not monitor.poll((record(),))
    assert probe.releases == 1
    probe.builds = [17]
    probe.now = 101
    assert not monitor.poll(())
    probe.found = frozenset({"response-0", "response-1"})
    probe.now = 103
    assert not monitor.poll((record(1, started_at=102),))
    assert monitor.overlap_writes == 1
    assert monitor.persisted_count == 2
    assert not monitor.pending
    assert monitor.max_lock_waiters == 2
    probe.current = state("complete")
    assert not monitor.poll(())
    probe.now = 162
    assert not monitor.poll(())
    probe.now = 163
    assert not monitor.poll(())
    probe.found = frozenset(f"response-{i}" for i in range(300))
    assert monitor.poll(tuple(record(i) for i in range(2, 300)))
    assert probe.releases == 1
    assert probe.metric_checks == 7


def test_http_200_without_spend_row_cannot_complete() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    records: Final = tuple(record(i) for i in range(300))
    _ = monitor.poll(records)
    with pytest.raises(AssertionError, match="no matching spend log"):
        monitor.assert_complete(state("complete", pending=False))


def test_no_pending_index_work_still_requires_300_persisted_requests() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    records: Final = tuple(record(i) for i in range(300))
    probe.found = frozenset(f"response-{i}" for i in range(300))
    _ = monitor.poll(records)
    monitor.assert_complete(state("complete", pending=False))
    assert monitor.successes == monitor.persisted_count == 300
    with pytest.raises(AssertionError, match="during index work"):
        monitor.assert_complete(state("complete", pending=True))


def test_postgres_failure_fails_even_when_inference_succeeded() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    probe.errors = True
    with pytest.raises(AssertionError, match="PostgreSQL"):
        _ = monitor.poll((record(),))


def test_failed_migration_and_operational_timeout_are_not_success() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    probe.current = state("failed")
    with pytest.raises(AssertionError, match="worker failed"):
        _ = monitor.poll(())
    probe.current = state()
    probe.now = MAX_TRAFFIC_SECONDS
    with pytest.raises(AssertionError, match="operational timeout"):
        _ = monitor.poll(())


def test_upgrade_case_uses_5rps_with_the_same_request_and_memory_limits() -> None:
    case: Final = upgrade_case()
    assert case.traffic.rps == 5
    assert not {"cpu_seconds_per_request", "postgres_calls_per_request", "redis_calls_per_request"}.intersection(
        limit.metric for limit in case.limits
    )
    assert case.limits == tuple(
        limit
        for limit in CASE.limits
        if limit.metric
        in {
            "unexpected_status_count",
            "gateway_added_seconds",
            "time_to_first_token_seconds",
            "memory_growth_megabytes",
        }
    )


def failed_record(seq: int) -> WireRecord:
    return record(seq).model_copy(update={"status": 503, "ok": False, "text": "", "response_id": None, "call_id": None})


def test_three_non_200s_do_not_start_indexes_or_require_spend_rows() -> None:
    probe: Final = Probe()
    monitor: Final = probe.monitor()
    assert not monitor.poll(tuple(failed_record(i) for i in range(3)))
    assert probe.releases == 0
    assert monitor.processed == 3
    assert monitor.successes == 0
    assert not monitor.pending
    probe.found = frozenset(f"response-{i}" for i in range(3, 303))
    assert not monitor.poll(tuple(record(i) for i in range(3, 303)))
    monitor.assert_complete(state("complete", pending=False))
    assert probe.releases == 1
    assert monitor.processed == 303
    assert monitor.successes == monitor.persisted_count == 300


def test_fourth_non_200_fails_across_poll_batches() -> None:
    monitor: Final = Probe().monitor()
    assert not monitor.poll(tuple(failed_record(i) for i in range(3)))
    with pytest.raises(AssertionError, match="Too many unexpected non-200"):
        _ = monitor.poll((failed_record(3),))


@pytest.mark.parametrize(
    "failed", [record().model_copy(update={"text": ""}), record().model_copy(update={"ok": False})]
)
def test_invalid_200_is_not_an_allowed_non_200(failed: WireRecord) -> None:
    with pytest.raises(AssertionError, match="Failed or empty inference"):
        _ = Probe().monitor().poll((failed,))


def _ceiling(metric: str, percentile: int | None = None) -> float:
    return next(
        limit.max
        for limit in upgrade_case().limits
        if limit.metric == metric and limit.percentile == percentile and limit.max is not None
    )


@pytest.mark.parametrize(
    "over,expected",
    [
        ("", ()),
        ("unexpected_status_count", ("unexpected_status_count",)),
        ("gateway_added_seconds", ("gateway_added_seconds percentile 50",)),
        ("time_to_first_token_seconds", ("time_to_first_token_seconds percentile 99",)),
        ("memory_growth_megabytes", ("memory_growth_megabytes",)),
    ],
)
def test_upgrade_verdict_uses_regular_limits_without_cpu_or_datastore_samples(
    over: str, expected: tuple[str, ...]
) -> None:
    """Each sample sits exactly on the case's ceiling, so only the one nudged past it goes over."""
    failures: Final = int(_ceiling("unexpected_status_count")) + (over == "unexpected_status_count")
    overhead: Final = _ceiling("gateway_added_seconds", 50) * (1.01 if over == "gateway_added_seconds" else 1)
    ttft: Final = _ceiling("time_to_first_token_seconds", 99) * (1.01 if over == "time_to_first_token_seconds" else 1)
    growth: Final = _ceiling("memory_growth_megabytes") + (over == "memory_growth_megabytes")
    nonstream: Final = tuple(record(i).model_copy(update={"total_s": overhead, "mock_delay_s": 0}) for i in range(100))
    stream: Final = record(100).model_copy(update={"stream": True, "first_token_s": ttft})
    observed: Final = Observation(
        records=(*nonstream, stream, *(failed_record(i) for i in range(101, 101 + failures))),
        mock_requests=(),
        guardrail_calls=(),
        rss_before=RssCapture((RssReading("gateway", "host", 1, 1000),), ()),
        rss_after=RssCapture((RssReading("gateway", "host", 1, 1000 + growth),), ()),
        wall_seconds=60,
    )
    rows, exceeded = evaluate(upgrade_case(), observed)
    assert tuple(row.label for row in rows if row.over or row.missing) == expected
    assert len(exceeded) == len(expected)


def test_upgrade_properties_carry_the_upgrade_timings_and_what_the_monitor_saw() -> None:
    state: Final = UpgradeState(
        phase="complete",
        started_at=0,
        schema_seconds=12.5,
        index_seconds=240.25,
        required_indexes=("idx_a", "idx_b"),
        required_migrations=("20260101_a",),
        pending_indexes=("idx_b",),
        baseline_rows={},
    )
    monitor: Final = UpgradeMonitor(
        state=lambda: state,
        check_metrics=lambda: None,
        read_activity=lambda: ([], 0),
        read_persisted=lambda records: frozenset(),
        start_indexes=lambda: None,
    )
    monitor.overlap_writes = 37
    monitor.max_lock_waiters = 3
    monitor.pending = (record(0), record(1))

    assert dict(upgrade_properties(state, monitor)) == {
        "perf_upgrade_schema_seconds": 12.5,
        "perf_upgrade_index_seconds": 240.25,
        "perf_upgrade_overlap_writes": 37,
        "perf_upgrade_max_lock_waiters": 3,
        "perf_upgrade_pending_indexes": 1,
        "perf_upgrade_unmatched_requests": 2,
    }

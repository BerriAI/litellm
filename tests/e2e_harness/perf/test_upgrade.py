from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from setup.upgrade import MAX_TRAFFIC_SECONDS, UpgradeMonitor, postgres_failures, upgrade_case
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
    assert monitor.poll(())
    assert probe.releases == 1
    assert probe.metric_checks == 6


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


def test_upgrade_case_uses_5rps_without_datastore_count_limits() -> None:
    case: Final = upgrade_case()
    assert case.traffic.rps == 5
    assert not {"postgres_calls_per_request", "redis_calls_per_request"}.intersection(
        limit.metric for limit in case.limits
    )
    assert next(limit.max for limit in case.limits if limit.metric == "unexpected_status_count") == 0

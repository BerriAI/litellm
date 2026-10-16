from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Final

import psycopg
from cases.small_bodies_5rps import CASE
from e2e_config import MASTER_KEY
from e2e_http import send_text_external
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from setup.performance_test_case import PerformanceTestCase
from setup.upgrade_worker import UpgradeState, fingerprints, missing_indexes, read_state, release_indexes
from setup.wire import WireRecord

MAX_TRAFFIC_SECONDS: Final = 6 * 3600
_ROWS: Final = TypeAdapter(tuple[tuple[str, str | None], ...])
_ACTIVITY: Final = TypeAdapter(tuple[list[int], int])
_COUNT: Final = TypeAdapter(tuple[int])
_SERVICES: Final = ("postgres", "batch_write_to_db")


def upgrade_case() -> PerformanceTestCase:
    return replace(
        CASE,
        name="database_upgrade_5rps",
        traffic=replace(CASE.traffic, total_requests=int(CASE.traffic.rps * MAX_TRAFFIC_SECONDS)),
        limits=tuple(
            limit
            for limit in CASE.limits
            if limit.metric not in {"cpu_seconds_per_request", "postgres_calls_per_request", "redis_calls_per_request"}
        ),
    )


class MetricsHeaders(BaseModel):
    model_config = ConfigDict(frozen=True)
    authorization: str = Field(default=f"Bearer {MASTER_KEY}", alias="Authorization")


def postgres_failures(text: str) -> float:
    families: Final = tuple(text_string_to_metric_families(text))
    names: Final = frozenset(family.name for family in families)
    required: Final = frozenset(f"litellm_{service}_failed_requests" for service in _SERVICES)
    assert required <= names, f"Missing database service counters: {required - names}"
    samples: Final = tuple(sample for family in families for sample in family.samples)
    failures: Final = tuple(sample.value for sample in samples if sample.name in {f"{name}_total" for name in required})
    assert all(math.isfinite(value) and value >= 0 for value in failures), "Invalid database failure counter"
    return sum(failures)


def assert_database_metrics(urls: tuple[str, ...]) -> None:
    assert urls, "E2E_DATABASE_METRICS_URLS must name every gateway/backend /metrics endpoint"
    for url in urls:
        assert_database_metrics_url(url)


def assert_database_metrics_url(url: str) -> None:
    result: Final = send_text_external("GET", url, headers=MetricsHeaders(), timeout=10)
    assert result.status_code == 200, f"Database metrics unavailable at {url}: {result.status_code}"
    assert postgres_failures(result.body) == 0, f"Failed PostgreSQL operations reported at {url}"


def persisted(connection: psycopg.Connection[tuple[object, ...]], records: tuple[WireRecord, ...]) -> frozenset[str]:
    ids: Final = tuple(record.response_id for record in records if record.response_id)
    if not ids:
        return frozenset()
    rows: Final = _ROWS.validate_python(
        connection.execute(
            'SELECT request_id, litellm_call_id FROM "LiteLLM_SpendLogs" WHERE request_id=ANY(%s)', (list(ids),)
        ).fetchall()
    )
    expected: Final = {record.response_id: record.call_id for record in records}
    return frozenset(identifier for identifier, call_id in rows if expected[identifier] == call_id)


def activity(connection: psycopg.Connection[tuple[object, ...]]) -> tuple[list[int], int]:
    return _ACTIVITY.validate_python(
        connection.execute("""SELECT
        ARRAY(SELECT pid FROM pg_stat_progress_create_index WHERE datname=current_database()),
        (SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock')
    """).fetchone()
    )


def assert_finished(connection: psycopg.Connection[tuple[object, ...]], state: UpgradeState) -> None:
    assert state.phase == "complete", f"Upgrade did not finish: {state.phase}"
    assert not missing_indexes(connection, state.required_indexes), "Required target indexes are missing or invalid"
    (missing,) = _COUNT.validate_python(
        connection.execute(
            """
        SELECT count(*) FROM unnest(%s::text[]) name WHERE NOT EXISTS (
            SELECT 1 FROM "_prisma_migrations" WHERE migration_name=name
            AND finished_at IS NOT NULL AND rolled_back_at IS NULL)""",
            (list(state.required_migrations),),
        ).fetchone()
    )
    assert missing == 0, f"{missing} target migrations were not applied"
    (unfinished,) = _COUNT.validate_python(
        connection.execute("""
        SELECT count(*) FROM "_prisma_migrations" WHERE finished_at IS NULL AND rolled_back_at IS NULL
    """).fetchone()
    )
    assert unfinished == 0, f"{unfinished} unfinished migrations"
    assert fingerprints(connection, tuple(state.baseline_rows)) == state.baseline_rows, (
        "Baseline rows changed or disappeared"
    )


class UpgradeMonitor:
    def __init__(
        self,
        *,
        state: Callable[[], UpgradeState],
        check_metrics: Callable[[], None],
        read_activity: Callable[[], tuple[list[int], int]],
        read_persisted: Callable[[tuple[WireRecord, ...]], frozenset[str]],
        start_indexes: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.state: Final = state
        self.check_metrics: Final = check_metrics
        self.read_activity: Final = read_activity
        self.read_persisted: Final = read_persisted
        self.start_indexes: Final = start_indexes
        self.clock: Final = clock
        self.wall_clock: Final = wall_clock
        self.started: Final = clock()
        self.status_limit: Final = next(limit.max for limit in CASE.limits if limit.metric == "unexpected_status_count")
        self.pending: tuple[WireRecord, ...] = ()
        self.processed = 0
        self.successes = 0
        self.persisted_count = 0
        self.overlap_writes = 0
        self.max_lock_waiters = 0
        self.previous_builds: frozenset[int] = frozenset()
        self.build_seen_at: float | None = None
        self.released = False
        self.completed_at: float | None = None

    def poll(self, records: tuple[WireRecord, ...]) -> bool:
        self.check_metrics()
        current: Final = self.state()
        assert current.phase != "failed", "Target migration/index worker failed"
        assert self.clock() - self.started < MAX_TRAFFIC_SECONDS, "Upgrade exceeded its six-hour operational timeout"
        successful: Final = tuple(record for record in records if record.status == 200)
        self.processed += len(records)
        self.successes += len(successful)
        assert self.processed - self.successes <= self.status_limit, "Too many unexpected non-200 responses"
        assert all(record.ok and record.text for record in successful), "Failed or empty inference response"
        assert all(record.response_id and record.call_id for record in successful), (
            "Response missing spend-log identifiers"
        )
        builds, lock_waiters = self.read_activity()
        if not self.previous_builds.intersection(builds):
            self.build_seen_at = self.wall_clock() if builds else None
        self.max_lock_waiters = max(self.max_lock_waiters, lock_waiters)
        self.pending += successful
        found: Final = self.read_persisted(self.pending)
        self.persisted_count += len(found)
        still_building, _ = self.read_activity()
        since: Final = self.build_seen_at
        if since is not None and frozenset(builds).intersection(still_building):
            self.overlap_writes += sum(
                1 for record in self.pending if record.response_id in found and record.started_at >= since
            )
        self.previous_builds = frozenset(builds)
        self.pending = tuple(record for record in self.pending if record.response_id not in found)
        if self.successes and not self.released:
            self.start_indexes()
            self.released = True
        if current.phase == "complete" and self.completed_at is None:
            self.completed_at = self.clock()
        return self.completed_at is not None and self.clock() - self.completed_at >= 60 and self.successes >= 300

    def assert_complete(self, state: UpgradeState) -> None:
        assert self.successes >= 300, f"Only {self.successes} requests completed"
        assert not self.pending, f"{len(self.pending)} successful requests have no matching spend log"
        assert self.persisted_count == self.successes, "Spend-log count does not match successful requests"
        assert not state.pending_indexes or self.overlap_writes > 0, "No persisted requests observed during index work"


def monitor_for(connection: psycopg.Connection[tuple[object, ...]], urls: tuple[str, ...]) -> UpgradeMonitor:
    return UpgradeMonitor(
        state=lambda: read_state(connection),
        check_metrics=lambda: assert_database_metrics(urls),
        read_activity=lambda: activity(connection),
        read_persisted=lambda records: persisted(connection, records),
        start_indexes=lambda: release_indexes(connection),
    )

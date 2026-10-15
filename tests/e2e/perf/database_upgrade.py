from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from threading import Event
from typing import Final, Literal, TextIO
from uuid import uuid4

import psycopg
from e2e_http import NoBody, Success, unwrap, without_retries
from models import KeyGenerateBody, KeyGenerateResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from transport import HttpTransport, Transport

from .database_observe import (
    DatabaseSample,
    IndexSample,
    await_spend_logs,
    count_logs,
    fingerprint,
    history,
    index_sample,
    log_sample,
    manifest,
    observations,
)
from .database_queries import (
    QueryCase,
    QuerySample,
    QuerySummary,
    latency_regressed,
    query_cases,
    query_once,
    summarize,
)
from .database_seed import TARGET_BYTES
from .database_traffic import (
    TrafficPhase,
    TrafficSample,
    TrafficSummary,
    request,
    traffic,
    traffic_summary,
)


class RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline_version: str
    target_version: str
    model: str
    target_indexes: tuple[str, ...] = Field(min_length=1)
    expected_bytes: int = Field(default=TARGET_BYTES, gt=0)
    repeats: int = Field(default=10, ge=2)
    upgrade_timeout: float = Field(default=3600, gt=0)
    max_traffic_gap: float = Field(default=10, gt=0)
    query_regression_ratio: float = Field(default=2, ge=1)


class Readiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    db: str
    litellm_version: str


class QueryRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: QueryCase
    samples: tuple[QuerySample, ...]
    summary: QuerySummary


class UpgradeWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    started: float
    finished: float
    exit_code: int
    traffic: tuple[TrafficSample, ...]
    database: tuple[DatabaseSample, ...]
    indexes_before: IndexSample
    indexes: tuple[IndexSample, ...]


class RunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: RunSpec
    manifest_sha256: str
    baseline_reported_version: str
    target_reported_version: str
    cache_classification: Literal["first-request-and-repeated-not-proven-cold"] = (
        "first-request-and-repeated-not-proven-cold"
    )
    migration_names: tuple[str, ...]
    migration_seconds: float
    upgrade_window_seconds: float
    rows_before: int
    rows_after: int
    traffic_logs_written: int
    traffic: tuple[TrafficSummary, ...]
    queries_before: tuple[QuerySummary, ...]
    queries_after: tuple[QuerySummary, ...]
    max_blocked_sessions: int
    max_lock_wait_seconds: float
    max_read_seconds: float
    background_index_overlap: Literal["passed", "failed", "no-pending-index-work"]
    failures: tuple[str, ...]


def save(path: Path, value: BaseModel | tuple[BaseModel, ...]) -> None:
    content: Final = (
        value.model_dump_json(indent=2)
        if isinstance(value, BaseModel)
        else ("[\n" + ",\n".join(item.model_dump_json() for item in value) + "\n]")
    )
    _ = path.write_text(content)


def benchmark(transport: Transport, cases: tuple[QueryCase, ...], repeats: int) -> tuple[QueryRun, ...]:
    return tuple(benchmark_case(transport, case, repeats) for case in cases)


def benchmark_case(transport: Transport, case: QueryCase, repeats: int) -> QueryRun:
    samples: Final = tuple(query_once(transport, case, repeated=index > 0) for index in range(repeats + 1))
    return QueryRun(case=case, samples=samples, summary=summarize(samples))


def ready(transport: Transport) -> Readiness:
    with without_retries():
        result: Final = unwrap(
            transport.get(
                "/health/readiness/details",
                headers=transport.master,
                params=NoBody(),
                response_type=Readiness,
                timeout=15,
            )
        )
    assert result.db == "connected", "Proxy cannot reach the seeded database"
    return result


def collect_traffic(
    transport: Transport,
    key: str,
    model: str,
    marker: str,
    stop: Event,
    started: Event,
    phase: TrafficPhase = "stable-during",
) -> tuple[TrafficSample, ...]:
    return tuple(traffic(transport, key, model, marker, phase, stop, started))


def collect_database(url: str, stop: Event) -> tuple[DatabaseSample, ...]:
    return tuple(observations(url, stop))


def run_command(command: tuple[str, ...], timeout: float, output: Path) -> int:
    assert command, "An upgrade command is required"
    with output.open("w") as log:
        try:
            return subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            log.write("Upgrade command timed out\n")
            return 124
        except OSError as exc:
            log.write(f"Upgrade command failed to start ({type(exc).__name__})\n")
            return 127


def during_upgrade(
    transport: Transport,
    candidate: Transport,
    key: str,
    target_key: str,
    database_url: str,
    spec: RunSpec,
    command: tuple[str, ...],
    output: Path,
) -> UpgradeWindow:
    stop: Final = Event()
    ready_event: Final = Event()
    with (
        psycopg.connect(database_url, autocommit=True, connect_timeout=5) as connection,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        connection.execute("SET statement_timeout = '5s'")
        indexes_before: Final = index_sample(connection, spec.target_indexes)
        requests: Final = pool.submit(collect_traffic, transport, key, spec.model, uuid4().hex, stop, ready_event)
        database: Final = pool.submit(collect_database, database_url, stop)
        try:
            assert ready_event.wait(30), "Stable traffic did not start before migration"
            started: Final = time.time()
            upgrade: Final = pool.submit(run_command, command, spec.upgrade_timeout, output / "upgrade.log")
            target_traffic, indexes = target_during_upgrade(
                connection,
                candidate,
                target_key,
                spec,
                upgrade,
                output,
            )
            exit_code: Final = upgrade.result()
            finished: Final = time.time()
        finally:
            stop.set()
            save(output / "during-traffic.json", requests.result())
            save(output / "database-observations.json", database.result())
    return UpgradeWindow(
        started=started,
        finished=finished,
        exit_code=exit_code,
        traffic=(*requests.result(), *target_traffic),
        database=database.result(),
        indexes_before=indexes_before,
        indexes=indexes,
    )


def target_during_upgrade(
    connection: psycopg.Connection[tuple[object, ...]],
    candidate: Transport,
    key: str,
    spec: RunSpec,
    upgrade: Future[int],
    output: Path,
) -> tuple[tuple[TrafficSample, ...], tuple[IndexSample, ...]]:
    deadline: Final = time.monotonic() + spec.upgrade_timeout
    while time.monotonic() < deadline:
        with without_retries():
            if (
                isinstance(
                    result := candidate.get(
                        "/health/readiness/details",
                        headers=candidate.master,
                        params=NoBody(),
                        response_type=Readiness,
                        timeout=2,
                    ),
                    Success,
                )
                and result.data.db == "connected"
                and (result.data.litellm_version.removeprefix("v") == spec.target_version.removeprefix("v"))
            ):
                break
        assert not upgrade.done() or upgrade.result() == 0, "Upgrade command failed before target readiness"
        time.sleep(0.2)
    else:
        raise AssertionError("Target release did not become ready before the upgrade deadline")
    stop: Final = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        requests: Final = pool.submit(
            collect_traffic,
            candidate,
            key,
            spec.model,
            uuid4().hex,
            stop,
            Event(),
            "target-during",
        )
        try:
            probe, before_probe = target_index_probe(connection, candidate, key, spec, upgrade, deadline)
            save(output / "target-index-probe.json", probe)
            with (output / "index-observations.jsonl").open("w") as log:
                _ = log.write(before_probe.model_dump_json() + "\n")
                log.flush()
                indexes: Final = (before_probe, *index_progress(connection, spec, probe, upgrade, deadline, log))
        finally:
            stop.set()
            save(output / "target-during-traffic.json", requests.result())
    return (probe, *requests.result()), indexes


def target_index_probe(
    connection: psycopg.Connection[tuple[object, ...]],
    candidate: Transport,
    key: str,
    spec: RunSpec,
    upgrade: Future[int],
    deadline: float,
) -> tuple[TrafficSample, IndexSample]:
    while not (sample := index_sample(connection, spec.target_indexes)).active_builds and not (
        upgrade.done() and not sample.pending
    ):
        assert not upgrade.done() or upgrade.result() == 0, "Upgrade command failed before index builds started"
        assert time.monotonic() < deadline, "Target request-log index builds did not start"
        time.sleep(0.2)
    return request(candidate, key, spec.model, uuid4().hex, "target-during"), sample


def index_progress(
    connection: psycopg.Connection[tuple[object, ...]],
    spec: RunSpec,
    probe: TrafficSample,
    upgrade: Future[int],
    deadline: float,
    log: TextIO,
) -> Iterator[IndexSample]:
    while sample := index_sample(connection, spec.target_indexes, probe.request_id):
        _ = log.write(sample.model_dump_json() + "\n")
        log.flush()
        yield sample
        if upgrade.done() and (upgrade.result() != 0 or (not sample.pending and sample.active_builds == 0)):
            return
        assert time.monotonic() < deadline, "Target request-log indexes did not finish before the upgrade deadline"
        time.sleep(0.2)


def background_overlap(window: UpgradeWindow) -> Literal["passed", "failed", "no-pending-index-work"]:
    if any(sample.pending and sample.active_builds and sample.target_probe_persisted for sample in window.indexes):
        return "passed"
    if window.indexes_before.pending or any(sample.pending or sample.active_builds for sample in window.indexes):
        return "failed"
    return "no-pending-index-work"


def fixed_traffic(
    transport: Transport, key: str, model: str, phase: Literal["before", "after"]
) -> tuple[TrafficSample, ...]:
    return tuple(islice(traffic(transport, key, model, uuid4().hex, phase, Event(), Event()), 20))


def run(
    connection: psycopg.Connection[tuple[object, ...]],
    stable: Transport,
    candidate: Transport,
    spec: RunSpec,
    command: tuple[str, ...],
    output: Path,
    *,
    database_url: str,
) -> RunReport:
    output.mkdir(parents=True, exist_ok=False)
    dataset: Final = manifest(connection, spec.expected_bytes)
    save(output / "dataset.json", dataset)
    assert dataset.spec.baseline_version == spec.baseline_version, "Snapshot does not match the resolved stable release"
    assert int(dataset.postgres_version.split(".")[0]) == connection.info.server_version // 10000, (
        "Snapshot PostgreSQL major version changed"
    )
    assert dataset.spec.anchor <= datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0), (
        "Historical query window must exclude today's traffic"
    )
    old_ready: Final = ready(stable)
    assert old_ready.litellm_version.removeprefix("v") == spec.baseline_version.removeprefix("v"), (
        "The stable proxy is not running the snapshot's release"
    )
    old_history: Final = history(connection)
    old_data: Final = fingerprint(connection, dataset.progress.rows)
    old_count: Final = count_logs(connection)
    assert old_count == dataset.progress.rows, "The snapshot copy already contains non-seed traffic"
    cases: Final = query_cases(dataset, log_sample(connection))
    queries_before: Final = benchmark(stable, cases, spec.repeats)
    save(output / "before-queries.json", queries_before)
    assert all(query.summary.errors == 0 for query in queries_before), "Baseline dashboard queries failed"
    with without_retries():
        key: Final = unwrap(
            stable.post(
                "/key/generate",
                headers=stable.master,
                json=KeyGenerateBody(models=[spec.model], duration="2h", key_alias="perf-upgrade-stable"),
                response_type=KeyGenerateResponse,
            )
        ).key
        target_key: Final = unwrap(
            stable.post(
                "/key/generate",
                headers=stable.master,
                json=KeyGenerateBody(models=[spec.model], duration="2h", key_alias="perf-upgrade-target"),
                response_type=KeyGenerateResponse,
            )
        ).key
    before: Final = fixed_traffic(stable, key, spec.model, "before")
    save(output / "before-traffic.json", before)
    assert all(sample.failure is None for sample in before), "Baseline traffic failed"
    _ = await_spend_logs(connection, tuple(sample.request_id for sample in before if sample.request_id is not None))
    window: Final = during_upgrade(stable, candidate, key, target_key, database_url, spec, command, output)
    save(output / "upgrade-window.json", window)
    assert window.exit_code == 0, f"Upgrade command exited {window.exit_code}, see upgrade.log"
    new_ready: Final = ready(candidate)
    assert new_ready.litellm_version.removeprefix("v") == spec.target_version.removeprefix("v"), (
        "Target version changed after the upgrade"
    )
    after: Final = fixed_traffic(candidate, target_key, spec.model, "after")
    save(output / "after-traffic.json", after)
    all_traffic: Final = (*before, *window.traffic, *after)
    request_ids: Final = tuple(
        sample.request_id for sample in all_traffic if sample.failure is None and sample.request_id
    )
    written: Final = await_spend_logs(connection, request_ids)
    queries_after: Final = benchmark(candidate, cases, spec.repeats)
    save(output / "after-queries.json", queries_after)
    new_history: Final = history(connection)
    assert all(row in new_history for row in old_history), "Existing migration history changed"
    assert old_data == fingerprint(connection, dataset.progress.rows), "Seed rows changed during the upgrade"
    new_count: Final = count_logs(connection)
    added: Final = tuple(row for row in new_history if row not in old_history)
    ends: Final = tuple(row.finished_at.timestamp() for row in added if row.finished_at is not None)
    starts: Final = tuple(row.started_at.timestamp() for row in added)
    traffic_stats: Final = tuple(
        traffic_summary(phase, all_traffic) for phase in ("before", "stable-during", "target-during", "after")
    )
    failures: Final = (
        *(
            f"{stat.phase} traffic failed or stalled"
            for stat in traffic_stats
            if stat.errors or stat.longest_gap_seconds > spec.max_traffic_gap
        ),
        *(f"{query.case.name} failed" for query in queries_after if query.summary.errors),
        *(
            f"{old.case.name} p95 regressed"
            for old, new in zip(queries_before, queries_after)
            if latency_regressed(
                old.summary.repeated_p95_seconds, new.summary.repeated_p95_seconds, spec.query_regression_ratio
            )
        ),
        *(
            ("Spend-log count did not grow by the successful traffic count",)
            if new_count != old_count + written
            else ()
        ),
        *(("Database reads failed during migration",) if any(sample.failure for sample in window.database) else ()),
        *(("No database observation during migration",) if not window.database else ()),
        *(
            ("Target traffic and spend writes did not overlap background index builds",)
            if background_overlap(window) == "failed"
            else ()
        ),
    )
    report: Final = RunReport(
        spec=spec,
        manifest_sha256=hashlib.sha256(dataset.model_dump_json().encode()).hexdigest(),
        baseline_reported_version=old_ready.litellm_version,
        target_reported_version=new_ready.litellm_version,
        migration_names=tuple(row.migration_name for row in added),
        migration_seconds=max(ends) - min(starts) if ends else 0,
        upgrade_window_seconds=window.finished - window.started,
        rows_before=old_count,
        rows_after=new_count,
        traffic_logs_written=written,
        traffic=traffic_stats,
        queries_before=tuple(query.summary for query in queries_before),
        queries_after=tuple(query.summary for query in queries_after),
        max_blocked_sessions=max((sample.blocked_sessions for sample in window.database), default=0),
        max_lock_wait_seconds=max((sample.longest_lock_wait_seconds for sample in window.database), default=0),
        max_read_seconds=max((sample.read_seconds for sample in window.database), default=0),
        background_index_overlap=background_overlap(window),
        failures=failures,
    )
    save(output / "report.json", report)
    return report


def main() -> None:
    assert os.environ.get("E2E_PERF_TESTS") == "1", "Set E2E_PERF_TESTS=1 for this opt-in benchmark"
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument(
        "--target-index",
        dest="target_indexes",
        action="append",
        required=True,
        help="Expected request-log index name from the target release, repeat for every index",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--stable-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--expected-bytes", type=int, default=TARGET_BYTES)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--upgrade-timeout", type=float, default=3600)
    parser.add_argument("--max-traffic-gap", type=float, default=10)
    parser.add_argument("--query-regression-ratio", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-disposable", action="store_true", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments: Final = parser.parse_args()
    values: Final = vars(arguments)
    spec: Final = RunSpec.model_validate(values)
    text: Final = TypeAdapter(str)
    command: Final = TypeAdapter(tuple[str, ...]).validate_python(values["command"])
    assert command and command != ("--",), "An upgrade command is required"
    assert values["stable_url"] != values["target_url"], "Stable and target need independently addressed proxies"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=5) as connection:
        connection.execute("SET statement_timeout = '20min'")
        result: Final = run(
            connection,
            HttpTransport(
                text.validate_python(values["stable_url"]), os.environ["LITELLM_MASTER_KEY"], request_timeout=60
            ),
            HttpTransport(
                text.validate_python(values["target_url"]), os.environ["LITELLM_MASTER_KEY"], request_timeout=60
            ),
            spec,
            command[1:] if command and command[0] == "--" else command,
            TypeAdapter(Path).validate_python(values["output"]),
            database_url=os.environ["DATABASE_URL"],
        )
    assert not result.failures, result.failures


if __name__ == "__main__":
    main()

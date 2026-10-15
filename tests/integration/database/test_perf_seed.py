import os
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import scratch_database
from integration._support.process import MIGRATE_DEPLOY, owned_proxy
from litellm_proxy_extras.request_log_indexes import ensure_request_log_indexes
from psycopg import sql

from tests.e2e.perf.database_observe import index_sample, persisted
from tests.e2e.perf.database_queries import LogRow, query_cases, query_once
from tests.e2e.perf.database_seed import SeedManifest, SeedSpec, initialize, measure, seed, seed_batch
from tests.e2e.perf.database_traffic import TrafficSample
from tests.e2e.perf.database_upgrade import RunSpec, ready, run
from tests.e2e.transport import HttpTransport

SPEC: Final = SeedSpec(baseline_version="v0.0.0", target_bytes=1, anchor=datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def seed_database_url(gateway: Gateway, tmp_path: Path) -> Generator[str]:
    with scratch_database("integration_perf_seed") as url:
        with owned_proxy(gateway, tmp_path, {"DATABASE_URL": url}, database_setup=MIGRATE_DEPLOY):
            pass
        yield url


@pytest.fixture
def seed_database(seed_database_url: str) -> Generator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(seed_database_url, autocommit=True) as connection:
        yield connection


def test_metadata_seed_resumes_without_duplicates_and_rolls_up_exact_totals(
    seed_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    initialize(seed_database, SPEC)
    assert seed_batch(seed_database, SPEC, 1000) == 1000
    assert seed_batch(seed_database, SPEC, 1000) == 2000
    seeded: Final = seed(seed_database, SPEC)
    assert seeded.rows == 2000
    assert seed(seed_database, SPEC).rows == 2000
    assert seeded.total_bytes == seeded.table_bytes + seeded.index_bytes
    assert seeded.total_bytes > 0 and seeded.index_bytes > 0
    stored: Final = seed_database.execute("SELECT manifest FROM litellm_perf_seed.checkpoint").fetchone()
    assert stored is not None
    manifest: Final = SeedManifest.model_validate(stored[0])
    assert manifest.progress.rows == 2000
    assert manifest.progress.total_bytes == seeded.total_bytes
    assert manifest.heap_bytes > 0 and manifest.database_bytes > seeded.total_bytes
    assert manifest.spec == SPEC
    assert manifest.store_prompts_in_spend_logs is False
    assert manifest.indexes
    assert seed_database.execute(
        "SELECT count(*), count(DISTINCT request_id), "
        "count(*) FILTER (WHERE status='failure'), count(*) FILTER (WHERE cache_hit='True'), "
        "count(*) FILTER (WHERE messages <> '{}'::jsonb OR response <> '{}'::jsonb "
        "OR proxy_server_request <> '{}'::jsonb), "
        'count(*) FILTER (WHERE total_tokens <> prompt_tokens + completion_tokens) FROM "LiteLLM_SpendLogs"'
    ).fetchone() == (2000, 2000, 40, 200, 0, 0)
    totals: Final = seed_database.execute(
        "SELECT count(*), sum(prompt_tokens), sum(completion_tokens), round(sum(spend)::numeric, 6) "
        'FROM "LiteLLM_SpendLogs"'
    ).fetchone()
    for table in (
        "LiteLLM_DailyUserSpend",
        "LiteLLM_DailyTeamSpend",
        "LiteLLM_DailyGlobalSpend",
        "LiteLLM_DailyEndUserSpend",
    ):
        assert (
            seed_database.execute(
                sql.SQL(
                    "SELECT sum(api_requests), sum(prompt_tokens), sum(completion_tokens), round(sum(spend)::numeric, 6) FROM {}"
                ).format(sql.Identifier(table))
            ).fetchone()
            == totals
        )
    assert seed_database.execute('SELECT sum(api_requests) FROM "LiteLLM_DailyTagSpend"').fetchone() == (4000,)
    assert seed_database.execute(
        'SELECT count(*) FROM "LiteLLM_SpendLogs" l '
        'LEFT JOIN "LiteLLM_VerificationToken" k ON l.api_key = k.token '
        'LEFT JOIN "LiteLLM_TeamTable" t ON l.team_id = t.team_id '
        'LEFT JOIN "LiteLLM_UserTable" u ON l."user" = u.user_id '
        "WHERE k.token IS NULL OR t.team_id IS NULL OR u.user_id IS NULL"
    ).fetchone() == (0,)
    assert seed_database.execute(
        'SELECT EXISTS(SELECT 1 FROM "LiteLLM_SpendLogs" l '
        'JOIN "LiteLLM_VerificationToken" k ON l.api_key = k.token '
        "WHERE l.team_id NOT IN ('perf-team-0', 'perf-team-1') AND right(k.key_alias, 1) NOT IN ('8', '9'))"
    ).fetchone() == (True,)


def test_dashboard_queries_return_seeded_logs_and_exact_daily_totals(
    seed_database: psycopg.Connection[tuple[object, ...]], seed_database_url: str, gateway: Gateway, tmp_path: Path
) -> None:
    assert seed(seed_database, SPEC, batch_rows=2000).rows == 2000
    stored: Final = seed_database.execute("SELECT manifest FROM litellm_perf_seed.checkpoint").fetchone()
    sample: Final = seed_database.execute(
        'SELECT row_to_json(l) FROM "LiteLLM_SpendLogs" l WHERE request_id = %s', ("perf-seed-17",)
    ).fetchone()
    assert stored is not None and sample is not None
    cases: Final = query_cases(SeedManifest.model_validate(stored[0]), LogRow.model_validate(sample[0]))
    with owned_proxy(
        gateway, tmp_path / "dashboard", {"DATABASE_URL": seed_database_url}, database_setup=MIGRATE_DEPLOY
    ) as proxy:
        transport: Final = HttpTransport(str(proxy.client.base_url), proxy.key)
        results: Final = tuple(query_once(transport, case) for case in cases)
        assert tuple((result.name, result.failure) for result in results if result.failure) == ()
        assert all(result.status == 200 for result in results)


@pytest.fixture
def perf_upgrade_proxy(
    seed_database: psycopg.Connection[tuple[object, ...]],
    seed_database_url: str,
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[HttpTransport, str]]:
    monkeypatch.setenv("PERF_TEST_DATABASE_URL", seed_database_url)
    with owned_proxy(
        gateway, tmp_path / "traffic", {"DATABASE_URL": seed_database_url}, database_setup=MIGRATE_DEPLOY
    ) as proxy:
        transport: Final = HttpTransport(str(proxy.client.base_url), proxy.key)
        version: Final = "v" + ready(transport).litellm_version.removeprefix("v")
        assert seed(seed_database, SPEC.model_copy(update={"baseline_version": version}), batch_rows=2000).rows == 2000
        proxy.post(
            "/model/new",
            {
                "model_name": "perf-upgrade-test",
                "litellm_params": {
                    "model": "openai/perf-upgrade-test",
                    "api_key": "integration-provider-key",
                    "api_base": f"{gateway.upstream_url}/v1",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
        )
        yield transport, version


def test_upgrade_measures_blocked_reads_while_traffic_and_spend_writes_survive(
    seed_database: psycopg.Connection[tuple[object, ...]],
    seed_database_url: str,
    perf_upgrade_proxy: tuple[HttpTransport, str],
    tmp_path: Path,
) -> None:
    transport, version = perf_upgrade_proxy
    report: Final = run(
        seed_database,
        transport,
        transport,
        RunSpec(
            baseline_version=version,
            target_version=version,
            model="perf-upgrade-test",
            target_indexes=("LiteLLM_SpendLogs_pkey",),
            expected_bytes=1,
            repeats=2,
            upgrade_timeout=30,
            query_regression_ratio=100,
        ),
        (
            sys.executable,
            "-c",
            (
                "import os, psycopg; "
                "c = psycopg.connect(os.environ['PERF_TEST_DATABASE_URL']); "
                "c.execute('ALTER TABLE \"LiteLLM_SpendLogs\" ADD COLUMN perf_upgrade_probe integer'); "
                "c.execute('SELECT pg_sleep(2)'); c.commit(); c.close()"
            ),
        ),
        tmp_path / "report",
        database_url=seed_database_url,
    )
    assert report.failures == ()
    assert report.rows_before == 2000
    assert report.rows_after == 2000 + report.traffic_logs_written
    assert report.traffic_logs_written >= 41
    assert report.max_read_seconds >= 1
    assert report.upgrade_window_seconds >= 2
    assert tuple(phase.phase for phase in report.traffic) == ("before", "stable-during", "target-during", "after")
    assert report.background_index_overlap == "no-pending-index-work"
    assert all(phase.errors == 0 for phase in report.traffic)
    assert all(query.errors == 0 for query in report.queries_after)


def build_index_while_waiting_for_target_probe() -> None:
    url: Final = os.environ["PERF_TEST_DATABASE_URL"]
    output: Final = Path(os.environ["PERF_TEST_OUTPUT"])
    index: Final = "LiteLLM_SpendLogs_litellm_call_id_idx"
    with psycopg.connect(url, autocommit=True) as observing, psycopg.connect(url) as held:
        observing.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index)))
        held.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        held.execute('SELECT request_id FROM "LiteLLM_SpendLogs" LIMIT 1').fetchone()
        with ThreadPoolExecutor(max_workers=1) as pool:
            building: Final = pool.submit(ensure_request_log_indexes, url, "public")
            try:
                log: Final = output / "index-observations.jsonl"
                _ = eventually(
                    lambda: log.read_text() if log.exists() else "",
                    lambda value: '"target_probe_persisted":true' in value,
                    seconds=60,
                )
                probe: Final = TrafficSample.model_validate_json((output / "target-index-probe.json").read_text())
                assert probe.failure is None and probe.request_id is not None
                assert persisted(observing, (probe.request_id,)) == 1
                state: Final = index_sample(observing, (index,))
                assert state.pending == (index,) and state.active_builds == 1
            finally:
                held.rollback()
            assert building.result(), "Shipped request-log index builder failed"


def test_upgrade_requires_target_requests_and_persisted_spend_while_index_build_is_active(
    seed_database: psycopg.Connection[tuple[object, ...]],
    seed_database_url: str,
    perf_upgrade_proxy: tuple[HttpTransport, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, version = perf_upgrade_proxy
    output: Final = tmp_path / "report"
    monkeypatch.setenv("PERF_TEST_OUTPUT", str(output))
    report: Final = run(
        seed_database,
        transport,
        transport,
        RunSpec(
            baseline_version=version,
            target_version=version,
            model="perf-upgrade-test",
            target_indexes=("LiteLLM_SpendLogs_litellm_call_id_idx",),
            expected_bytes=1,
            repeats=2,
            upgrade_timeout=90,
            query_regression_ratio=100,
        ),
        (
            sys.executable,
            "-c",
            "from integration.database.test_perf_seed import "
            "build_index_while_waiting_for_target_probe; build_index_while_waiting_for_target_probe()",
        ),
        output,
        database_url=seed_database_url,
    )
    assert report.failures == ()
    assert report.background_index_overlap == "passed"
    assert report.rows_after == report.rows_before + report.traffic_logs_written
    assert all(phase.errors == 0 for phase in report.traffic)
    assert all(query.errors == 0 for query in report.queries_after)
    assert index_sample(seed_database, ("LiteLLM_SpendLogs_litellm_call_id_idx",)).pending == ()


def test_failed_batch_rolls_back_logs_and_checkpoint_then_resumes(
    seed_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    initialize(seed_database, SPEC)
    assert seed_batch(seed_database, SPEC, 100) == 100
    seed_database.execute(
        'ALTER TABLE "LiteLLM_DailyTeamSpend" ADD CONSTRAINT reject_batch CHECK (api_requests < 0) NOT VALID'
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        seed_batch(seed_database, SPEC, 100)
    assert seed_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (100,)
    assert seed_database.execute("SELECT rows FROM litellm_perf_seed.checkpoint").fetchone() == (100,)
    assert seed_database.execute('SELECT sum(api_requests) FROM "LiteLLM_DailyUserSpend"').fetchone() == (100,)
    seed_database.execute('ALTER TABLE "LiteLLM_DailyTeamSpend" DROP CONSTRAINT reject_batch')
    assert seed_batch(seed_database, SPEC, 100) == 200
    assert seed_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (200,)


def test_seed_refuses_partitioned_table_instead_of_measuring_only_its_parent(
    seed_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    seed_database.execute('ALTER TABLE "LiteLLM_SpendLogs" RENAME TO saved_unpartitioned_logs')
    seed_database.execute(
        'CREATE TABLE "LiteLLM_SpendLogs" (LIKE saved_unpartitioned_logs) PARTITION BY RANGE ("startTime")'
    )
    with pytest.raises(AssertionError, match="requires unpartitioned spend logs"):
        seed(seed_database, SPEC)


def test_seed_reaches_measured_size_and_refuses_a_changed_profile(
    seed_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    spec: Final = SPEC.model_copy(update={"target_bytes": measure(seed_database, 0).total_bytes + 1_000_000})
    result: Final = seed(seed_database, spec, batch_rows=100)
    assert result.total_bytes >= spec.target_bytes
    assert result.rows >= 100
    assert seed_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (result.rows,)
    with pytest.raises(AssertionError, match="same seed specification"):
        seed(seed_database, SPEC)
    assert seed_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (result.rows,)


def test_seed_refuses_existing_spend_logs(seed_database: psycopg.Connection[tuple[object, ...]]) -> None:
    seed_database.execute(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime") '
        "VALUES ('existing-request', 'acompletion', now(), now())"
    )
    with pytest.raises(AssertionError, match="existing spend-log dataset"):
        seed(seed_database, SPEC)
    assert seed_database.execute('SELECT request_id FROM "LiteLLM_SpendLogs"').fetchall() == [("existing-request",)]

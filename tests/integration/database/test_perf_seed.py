from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway
from integration._support.database import scratch_database
from integration._support.process import MIGRATE_DEPLOY, owned_proxy
from psycopg import sql

from tests.e2e.perf.database_seed import SeedManifest, SeedSpec, initialize, measure, seed, seed_batch

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


@pytest.mark.parametrize("legacy_end_user", (False, True))
def test_metadata_seed_resumes_without_duplicates_and_rolls_up_exact_totals(
    seed_database: psycopg.Connection[tuple[object, ...]],
    legacy_end_user: bool,
) -> None:
    if legacy_end_user:
        seed_database.execute('ALTER TABLE "LiteLLM_EndUserTable" DROP COLUMN models')
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
        'LEFT JOIN "LiteLLM_EndUserTable" e ON l.end_user = e.user_id '
        "WHERE k.token IS NULL OR t.team_id IS NULL OR u.user_id IS NULL OR e.user_id IS NULL"
    ).fetchone() == (0,)
    assert seed_database.execute(
        'SELECT EXISTS(SELECT 1 FROM "LiteLLM_SpendLogs" l '
        'JOIN "LiteLLM_VerificationToken" k ON l.api_key = k.token '
        "WHERE l.team_id NOT IN ('perf-team-0', 'perf-team-1') AND right(k.key_alias, 1) NOT IN ('8', '9'))"
    ).fetchone() == (True,)


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

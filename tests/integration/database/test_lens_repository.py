import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma
from psycopg import sql

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.models import (
    Check,
    Evidence,
    Execution,
    Finding,
    Job,
    Lens,
    LensSettings,
    RunAssessment,
    Sample,
    Scope,
    TraceFindingCount,
    TraceFindingsRequest,
    TraceIdentity,
    Worker,
)
from litellm.proxy.lens.repository import LensRepository, WriterDatabase
from litellm.proxy.lens.state import claim_job, queue_job


@pytest_asyncio.fixture(loop_scope="function")
async def lens_db() -> AsyncIterator[Prisma]:
    async with Prisma(datasource={"url": os.environ["DATABASE_URL"]}) as db:
        yield db


@pytest.mark.asyncio
async def test_concurrent_workers_cannot_both_acquire_the_same_job(lens_db: Prisma) -> None:
    now: Final = datetime.now(timezone.utc)
    scope: Final = Scope(team_id=uuid4().hex)
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    lens: Final = Lens(
        id=uuid4().hex,
        scope=scope,
        settings=LensSettings(name="Lease test", model="test", checks=(Check(id="c", instruction="Find retries"),)),
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    await repo.create(queue_job(lens, now, uuid4().hex))
    try:
        workers: Final = tuple(Worker(id=uuid4().hex, name="worker", scope=scope, last_seen=now) for _ in range(2))
        results: Final = await asyncio.gather(
            *(repo.update(lens.id, lambda e, w=w: claim_job(e, w, now)) for w in workers)
        )
        stored: Final = await repo.get(lens.id)
        assert stored is not None
        assert stored.jobs[0].attempts == 1
        assert stored.jobs[0].worker_id in tuple(w.id for w in workers)
        assert tuple(r.jobs[0].worker_id for r in results if r) == (stored.jobs[0].worker_id, stored.jobs[0].worker_id)
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', lens.id)


@pytest.mark.asyncio
async def test_heartbeat_never_restores_revoked_access(lens_db: Prisma) -> None:
    now: Final = datetime.now(timezone.utc)
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    worker: Final = Worker(id=uuid4().hex, name="worker", scope=Scope(team_id=uuid4().hex), last_seen=now)
    token_hash: Final = uuid4().hex
    await repo.save_worker(worker, token_hash)
    try:
        await repo.save_worker(worker.model_copy(update={"revoked": True}))
        await repo.heartbeat(worker.id, now.isoformat())
        stored: Final = await repo.worker(token_hash)
        assert stored is not None and stored.revoked is True
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_LensWorker" WHERE id=$1', worker.id)


@pytest.mark.asyncio
async def test_trace_findings_include_archived_assessments_without_counting_retries_or_counterexamples(
    lens_db: Prisma,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.lens.endpoints import trace_findings

    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=lens_db))
    now: Final = datetime.now(timezone.utc)
    prefix: Final = uuid4().hex
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    settings: Final = LensSettings(name="Finding counts", model="test", context="Answer the question")
    identities: Final = tuple(TraceIdentity(trace_id=prefix, trace_ref=f"{prefix}-{i}") for i in range(7))
    executions: Final = tuple(
        Execution(
            id=f"{prefix}-{i}",
            source="traces",
            trace_id=identity.trace_id,
            trace_ref=identity.trace_ref,
            team_id="",
            name="Run",
            start_time=now.isoformat(),
            span_count=1,
        )
        for i, identity in enumerate(identities)
    )
    finding: Final = Finding(
        id=prefix,
        title="Repeated lookup",
        description="The agent never answered the question",
        check_id="expected_behavior",
        first_seen=now,
        last_seen=now,
        revision=1,
        occurrences=(executions[0].id,),
        evidence=(
            Evidence(execution_id=executions[0].id, span_id="step", quote="no answer"),
            Evidence(execution_id=executions[1].id, span_id="step", quote="answered", role="counterexample"),
        ),
    )
    completed: Final = Job(
        id=f"{prefix}-old",
        status="completed",
        created_at=now,
        start=now,
        end=now,
        settings=settings,
        revision=1,
        sample=Sample(executions=executions[:4], eligible=4),
        assessments=(
            RunAssessment(execution_id=executions[0].id),
            RunAssessment(execution_id=executions[1].id),
            RunAssessment(execution_id=executions[2].id, cannot_assess=True),
        ),
        findings=(finding,),
    )
    lens: Final = Lens(
        id=prefix,
        scope=Scope(all_teams=True),
        settings=settings,
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
        jobs=(completed,),
    )
    await repo.create(lens)
    try:
        current: Final = completed.model_copy(update={"id": f"{prefix}-current"})
        unfinished: Final = tuple(
            completed.model_copy(
                update={
                    "id": f"{prefix}-{status}",
                    "status": status,
                    "sample": Sample(executions=(executions[index],), eligible=1),
                    "assessments": (RunAssessment(execution_id=executions[index].id),),
                    "findings": (),
                }
            )
            for index, status in enumerate(("running", "failed", "cancelled"), start=4)
        )
        await repo.update(prefix, lambda item: item.model_copy(update={"jobs": (current, *unfinished)}))
        archived: Final = await repo.job(prefix, completed.id)
        assert archived is not None and archived.status == "completed"
        expected: Final = tuple(
            TraceFindingCount(**identity.model_dump(), finding_count=1 if i == 0 else 0 if i == 1 else None)
            for i, identity in enumerate(identities)
        )
        counts: Final = await trace_findings(
            TraceFindingsRequest(traces=identities), UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        )
        assert sorted(counts, key=lambda item: item.trace_ref) == list(expected)
        await repo.update(prefix, lambda item: item.model_copy(update={"jobs": unfinished}))
        archived_counts: Final = await repo.trace_findings(identities)
        assert sorted(archived_counts, key=lambda item: item.trace_ref) == list(expected)
        assert await repo.trace_findings((TraceIdentity(trace_id=prefix),)) == (
            TraceFindingCount(trace_id=prefix, finding_count=None),
        )
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_LensRun" WHERE lens_id=$1', prefix)
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', prefix)


@pytest.mark.parametrize("populated", (False, True))
@pytest.mark.parametrize("preceding_schema", (False, True))
def test_lens_rename_preserves_saved_data_and_worker_credentials(populated: bool, preceding_schema: bool) -> None:
    migrations: Final = (
        Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
    )
    schema: Final = f"lens_migration_{uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        try:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            for name in ("20260930000000_agent_engine", "20261001000000_lens_run_history"):
                connection.execute(sql.SQL((migrations / name / "migration.sql").read_text()))
            if populated:
                connection.execute(
                    """INSERT INTO "LiteLLM_Engine" VALUES ('lens', 7, '{"findings":[{"id":"finding"}]}');
                    INSERT INTO "LiteLLM_EngineWorker" VALUES ('worker', 'token-hash', '{"analysis_key_id":"key"}');
                    INSERT INTO "LiteLLM_EngineRun" VALUES ('batch', 'lens', '2026-01-01', '{"cost":1.25}')"""
                )
            if preceding_schema:
                first_schema: Final = f"lens_first_{uuid4().hex}"
                connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(first_schema)))
                connection.execute(
                    sql.SQL("SET LOCAL search_path TO {}, {}").format(
                        sql.Identifier(first_schema), sql.Identifier(schema)
                    )
                )
            connection.execute(sql.SQL((migrations / "20261001100000_rename_lens" / "migration.sql").read_text()))
            connection.execute(sql.SQL((migrations / "20261001100000_rename_lens" / "migration.sql").read_text()))
            assert connection.execute('SELECT id, version, data FROM "LiteLLM_Lens"').fetchall() == (
                [("lens", 7, {"findings": [{"id": "finding"}]})] if populated else []
            )
            assert connection.execute('SELECT id, token_hash, data FROM "LiteLLM_LensWorker"').fetchall() == (
                [("worker", "token-hash", {"analysis_key_id": "key"})] if populated else []
            )
            assert connection.execute('SELECT id, lens_id, data FROM "LiteLLM_LensRun"').fetchall() == (
                [("batch", "lens", {"cost": 1.25})] if populated else []
            )
        finally:
            connection.rollback()


@pytest.mark.parametrize("entrypoint", ("proxy", "extras-v1", "extras-v2"))
@pytest.mark.parametrize("legacy_table", ("LiteLLM_Engine", "LiteLLM_EngineRun", "LiteLLM_EngineWorker"))
def test_db_push_refuses_legacy_lens_data(monkeypatch: pytest.MonkeyPatch, entrypoint: str, legacy_table: str) -> None:
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    from litellm.proxy.db.prisma_client import PrismaManager

    database_url: Final = os.environ["DATABASE_URL"]
    schema: Final = f"lens_push_{uuid4().hex}"
    parsed: Final = urlsplit(database_url)
    scoped: Final = urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            connection.execute(
                sql.SQL("CREATE TABLE {} (id TEXT PRIMARY KEY, data JSONB)").format(
                    sql.Identifier(schema, legacy_table)
                )
            )
            connection.execute(
                sql.SQL("INSERT INTO {} VALUES ('saved', '{{\"keep\":true}}')").format(
                    sql.Identifier(schema, legacy_table)
                )
            )
            monkeypatch.setenv("DATABASE_URL", scoped)
            setup: Final = (
                PrismaManager.setup_database if entrypoint == "proxy" else ProxyExtrasDBManager.setup_database
            )
            with pytest.raises(RuntimeError, match="Legacy Lens tables exist"):
                setup(use_migrate=False, use_v2_resolver=entrypoint == "extras-v2")
            assert connection.execute(
                sql.SQL("SELECT id, data FROM {}").format(sql.Identifier(schema, legacy_table))
            ).fetchall() == [("saved", {"keep": True})]
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_db_push_creates_fresh_lens_tables_and_preserves_them_on_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.db.prisma_client import PrismaManager

    database_url: Final = os.environ["DATABASE_URL"]
    schema: Final = f"lens_fresh_push_{uuid4().hex}"
    parsed: Final = urlsplit(database_url)
    scoped: Final = urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            monkeypatch.setenv("DATABASE_URL", scoped)
            assert PrismaManager.setup_database(use_migrate=False)
            connection.execute(
                sql.SQL("INSERT INTO {} (id, data) VALUES ('saved', '{{\"keep\":true}}')").format(
                    sql.Identifier(schema, "LiteLLM_Lens")
                )
            )
            assert PrismaManager.setup_database(use_migrate=False)
            assert connection.execute(
                sql.SQL("SELECT id, data FROM {}").format(sql.Identifier(schema, "LiteLLM_Lens"))
            ).fetchall() == [("saved", {"keep": True})]
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio
async def test_review_checkpoints_survive_new_jobs_and_only_relevant_settings_invalidate_them(lens_db: Prisma) -> None:
    from litellm.proxy.lens.models import Extraction, Review, ReviewVersion

    now: Final = datetime.now(timezone.utc)
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    settings: Final = LensSettings(name="Checkpoint test", model="analysis", context="Find blocked user requests")
    execution: Final = Execution(
        id=uuid4().hex,
        source="traces",
        trace_id=uuid4().hex,
        team_id="",
        name="task",
        start_time=now.isoformat(),
        span_count=1,
    )
    lens: Final = Lens(
        id=uuid4().hex,
        scope=Scope(all_teams=True),
        settings=settings,
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    job: Final = (
        queue_job(lens, now, uuid4().hex)
        .jobs[0]
        .model_copy(
            update={
                "sample": Sample(executions=(execution,), eligible=1),
            }
        )
    )
    checkpoint: Final = Review(
        execution_id=execution.id,
        trace_id=execution.trace_id,
        agent="agent",
        name="task",
        model=settings.model,
        duration_ms=10,
        at=now,
        content_version="version-1",
        extraction=Extraction(),
    )
    await repo.create(lens)
    try:
        await repo.save_review(lens.id, job, checkpoint)
        resumed: Final = job.model_copy(
            update={"id": uuid4().hex, "settings": settings.model_copy(update={"monthly_budget": 200})}
        )
        assert await repo.reviews(lens.id, resumed) == (checkpoint,)
        await repo.complete_reviews(
            lens.id, resumed, (ReviewVersion(execution_id=execution.id, content_version="version-1"),)
        )
        assert (await repo.reviews(lens.id, resumed))[0].consolidated
        changed: Final = resumed.model_copy(
            update={"settings": settings.model_copy(update={"context": "Find fabricated answers"})}
        )
        assert await repo.reviews(lens.id, changed) == ()
        updated: Final = checkpoint.model_copy(update={"content_version": "version-2"})
        await repo.save_review(lens.id, resumed, updated)
        await repo.complete_reviews(
            lens.id, resumed, (ReviewVersion(execution_id=execution.id, content_version="version-1"),)
        )
        assert await repo.reviews(lens.id, resumed) == (updated,)
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', lens.id)


@pytest.mark.asyncio
async def test_legacy_finding_run_provenance_is_recovered_from_archived_and_current_jobs(lens_db: Prisma) -> None:
    from litellm.proxy.lens.state import merge_finding
    from tests.unit.proxy.lens.test_state import NOW, finding, lens

    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    saved: Final = merge_finding(lens(), finding("trace"), 1, NOW)
    old: Final = (
        queue_job(lens(), NOW, uuid4().hex).jobs[0].model_copy(update={"status": "completed", "findings": (saved,)})
    )
    stored: Final = lens().model_copy(update={"id": uuid4().hex, "findings": (saved,), "jobs": (old,)})
    await repo.create(stored)
    try:
        await repo.update(stored.id, lambda value: queue_job(value, NOW, uuid4().hex))
        matches: Final = await repo.finding_runs(stored.id, (saved.id,))
        assert tuple((match.finding_id, match.job_id) for match in matches) == ((saved.id, old.id),)
        assert await repo.finding_runs(stored.id, ("unrelated",)) == ()
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', stored.id)


def test_review_migration_preserves_existing_lens_history_credentials_and_spend() -> None:
    from tests.unit.proxy.lens.test_state import NOW, lens, worker

    migrations: Final = (
        Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
    )
    schema: Final = f"lens_reviews_{uuid4().hex}"
    legacy: Final = lens().model_dump_json(exclude={"criteria_updated_at", "reservations"})
    job: Final = queue_job(lens(), NOW, "archived").jobs[0].model_dump_json(
        exclude={"review_versions": True, "coverage": {"reused"}}
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        try:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            for name in (
                "20260930000000_agent_engine",
                "20261001000000_lens_run_history",
                "20261001100000_rename_lens",
            ):
                connection.execute(sql.SQL((migrations / name / "migration.sql").read_text()))
            connection.execute('INSERT INTO "LiteLLM_Lens" VALUES (%s, 0, %s)', ("lens", legacy))
            connection.execute(
                'INSERT INTO "LiteLLM_LensRun" VALUES (%s, %s, %s, %s)', ("archived", "lens", NOW, job)
            )
            connection.execute(
                'INSERT INTO "LiteLLM_LensWorker" VALUES (%s, %s, %s)',
                ("worker", "existing-token", worker().model_dump_json()),
            )
            before: Final = tuple(
                connection.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))).fetchall()
                for table in ("LiteLLM_Lens", "LiteLLM_LensRun", "LiteLLM_LensWorker")
            )
            connection.execute(
                sql.SQL((migrations / "20261006000000_lens_review_checkpoints" / "migration.sql").read_text())
            )
            after: Final = tuple(
                connection.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))).fetchall()
                for table in ("LiteLLM_Lens", "LiteLLM_LensRun", "LiteLLM_LensWorker")
            )
            assert after == before
            assert Lens.model_validate(after[0][0][-1]) == lens()
            assert Job.model_validate(after[1][0][-1]) == queue_job(lens(), NOW, "archived").jobs[0]
            assert connection.execute('SELECT count(*) FROM "LiteLLM_LensReview"').fetchone() == (0,)
        finally:
            connection.rollback()

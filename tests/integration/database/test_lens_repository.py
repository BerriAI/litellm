import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma
from psycopg import sql

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.models import MAX_STEPS, Check, Lens, LensSettings, MetadataFilter, Scope, Step, Worker
from litellm.proxy.lens.repository import LensRepository, WriterDatabase
from litellm.proxy.lens.search import LensField, parse_search
from litellm.proxy.lens.state import claim_job, queue_job, replace_job


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


@pytest_asyncio.fixture(loop_scope="function")
async def charged_lens(lens_db: Prisma) -> AsyncIterator[tuple[LensRepository, Lens]]:
    now: Final = datetime.now(timezone.utc)
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    lens: Final = Lens(
        id=uuid4().hex,
        scope=Scope(all_teams=True),
        settings=LensSettings(name="Ledger test", model="test", context="Find failures", monthly_budget=1),
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    await repo.create(queue_job(lens, now, "run-a"))
    try:
        yield repo, lens
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_LensRun" WHERE lens_id=$1', lens.id)
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=$1', lens.id)


def model_step(at: datetime, cost: float, label: str = "Reviewed a run") -> Step:
    return Step(at=at, kind="model", label=label, model="test", purpose="extract", cost=cost)


@pytest.mark.asyncio
async def test_concurrent_model_calls_reserve_budget_without_rewriting_the_document(
    charged_lens: tuple[LensRepository, Lens],
) -> None:
    repo, lens = charged_lens
    now: Final = datetime.now(timezone.utc)
    reservations: Final = await asyncio.gather(
        *(repo.reserve(lens.id, "run-a", model_step(now + timedelta(seconds=i), 0.03)) for i in range(32))
    )
    assert all(reservation is not None for reservation in reservations)
    stored: Final = await repo.get(lens.id)
    assert stored is not None
    assert stored.version == 0
    assert stored.spent == pytest.approx(32 * 0.03)
    assert stored.jobs[0].cost == pytest.approx(32 * 0.03)
    assert len(stored.jobs[0].steps) == 32


@pytest.mark.asyncio
async def test_concurrent_reservations_admit_exactly_what_fits_the_monthly_budget(
    charged_lens: tuple[LensRepository, Lens],
) -> None:
    repo, lens = charged_lens
    now: Final = datetime.now(timezone.utc)
    reservations: Final = await asyncio.gather(
        *(repo.reserve(lens.id, "run-a", model_step(now, 0.3)) for _ in range(10))
    )
    assert sum(reservation is not None for reservation in reservations) == 3
    stored: Final = await repo.get(lens.id)
    assert stored is not None
    assert stored.spent == pytest.approx(0.9)
    assert len(stored.jobs[0].steps) == 3


@pytest.mark.asyncio
async def test_settling_and_releasing_adjust_spend_exactly_once(charged_lens: tuple[LensRepository, Lens]) -> None:
    repo, lens = charged_lens
    now: Final = datetime.now(timezone.utc)
    first: Final = await repo.reserve(lens.id, "run-a", model_step(now, 0.4))
    second: Final = await repo.reserve(lens.id, "run-a", model_step(now + timedelta(seconds=1), 0.4))
    assert first is not None and second is not None
    await repo.settle(first, model_step(now, 0.1))
    await repo.release(second)
    await repo.settle(first, model_step(now, 0.1))
    await repo.release(first)
    await repo.settle(second, model_step(now, 0.1))
    stored: Final = await repo.get(lens.id)
    assert stored is not None
    assert stored.spent == pytest.approx(0.1)
    assert stored.jobs[0].cost == pytest.approx(0.1)
    assert tuple(step.cost for step in stored.jobs[0].steps) == (pytest.approx(0.1),)


@pytest.mark.asyncio
async def test_spend_counter_survives_document_writes_and_rolls_over_each_month(
    charged_lens: tuple[LensRepository, Lens],
) -> None:
    repo, lens = charged_lens
    now: Final = datetime.now(timezone.utc)
    reserved: Final = await repo.reserve(lens.id, "run-a", model_step(now, 0.5))
    assert reserved is not None
    renamed: Final = await repo.update(
        lens.id,
        lambda e: e.model_copy(update={"settings": e.settings.model_copy(update={"name": "Renamed"}), "spent": 0}),
    )
    assert renamed is not None and renamed.spent == pytest.approx(0.5)
    assert renamed.settings.name == "Renamed"
    next_month: Final = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    rolled: Final = await repo.reserve(lens.id, "run-a", model_step(next_month, 0.2))
    assert rolled is not None
    await repo.settle(reserved, model_step(now, 0.3))
    stored: Final = await repo.get(lens.id)
    assert stored is not None
    assert stored.budget_month == next_month.strftime("%Y-%m")
    assert stored.spent == pytest.approx(0.2)
    assert stored.jobs[0].cost == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_steps_follow_the_run_into_history(charged_lens: tuple[LensRepository, Lens]) -> None:
    repo, lens = charged_lens
    now: Final = datetime.now(timezone.utc)
    reserved: Final = await repo.reserve(lens.id, "run-a", model_step(now, 0.2))
    assert reserved is not None
    await repo.settle(reserved, model_step(now, 0.1))
    for i in range(MAX_STEPS + 5):
        await repo.record_step(
            lens.id, "run-a", Step(at=now + timedelta(seconds=i + 1), kind="stage", label=f"stage {i}")
        )
    later: Final = now + timedelta(hours=1)
    await repo.update(lens.id, lambda e: replace_job(e, e.jobs[0].model_copy(update={"status": "completed"})))
    rotated: Final = await repo.update(lens.id, lambda e: queue_job(e, later, "run-b"))
    assert rotated is not None and rotated.jobs[0].id == "run-b"
    assert rotated.jobs[0].steps == () and rotated.jobs[0].cost == 0
    history: Final = await repo.jobs(lens.id)
    assert tuple(job.id for job in history) == ("run-b", "run-a")
    archived: Final = await repo.job(lens.id, "run-a")
    assert archived is not None and archived.status == "completed"
    assert archived.cost == pytest.approx(0.1)
    assert len(archived.steps) == MAX_STEPS
    assert archived.steps[0].label == "stage 5"
    assert archived.steps[-1].label == f"stage {MAX_STEPS + 4}"
    assert await repo.job(lens.id, reserved.id) is None


@pytest_asyncio.fixture(loop_scope="function")
async def searchable_lenses(lens_db: Prisma) -> AsyncIterator[tuple[LensRepository, str]]:
    tag: Final = uuid4().hex[:12]
    start: Final = datetime.now(timezone.utc)
    repo: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))

    def lens(lens_id: str, minutes: int, status: str | None, **settings: object) -> Lens:
        created: Final = start + timedelta(minutes=minutes)
        base: Final = Lens(
            id=f"{tag}-{lens_id}",
            scope=Scope(all_teams=True),
            settings=LensSettings.model_validate({"model": "test", "context": "Find failures", **settings}),
            created_at=created,
            next_run_at=created,
            budget_month=created.strftime("%Y-%m"),
        )
        if status is None:
            return base
        queued: Final = queue_job(base, created, uuid4().hex)
        return queued.model_copy(update={"jobs": (queued.jobs[0].model_copy(update={"status": status}),)})

    lenses: Final = (
        lens(
            "a",
            0,
            "failed",
            name=f"{tag} nightly",
            agent_name=f"{tag}-researcher",
            filters=(MetadataFilter(key="env", value="prod"),),
        ),
        lens("b", 1, None, name=f"{tag} weekly 100%_x", service=f"{tag}-billing", enabled=False),
        lens("c", 2, "completed", name=f"{tag} adhoc"),
    )
    for created in lenses:
        await repo.create(created)
    try:
        yield repo, tag
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id LIKE $1', f"{tag}-%")


@pytest.mark.parametrize(
    "q, expected",
    (
        ("", ("c", "b", "a")),
        ("status:failed", ("a",)),
        ("status:never", ("b",)),
        ("-status:never", ("c", "a")),
        ("status:FAIL*", ("a",)),
        ("schedule:paused", ("b",)),
        ("-schedule:paused", ("c", "a")),
        ("agent:{tag}-researcher", ("a",)),
        ("agent:{tag}-bill*", ("b",)),
        ("agent:researcher", ()),
        ("-agent:*", ("c",)),
        ("name:*nightly", ("a",)),
        ("name:nightly", ()),
        ('"env: prod"', ("a",)),
        ('"all activity"', ("c",)),
        ("100%_x", ("b",)),
        ("100%x", ()),
        ("100__x", ()),
        ("weekly status:never -schedule:watching", ("b",)),
    ),
)
@pytest.mark.asyncio
async def test_search_selects_lenses_newest_first(
    searchable_lenses: tuple[LensRepository, str], q: str, expected: tuple[str, ...]
) -> None:
    repo, tag = searchable_lenses
    found: Final = await repo.search(parse_search(f"{tag} {q.format(tag=tag)}"))
    assert tuple(lens.id.removeprefix(f"{tag}-") for lens in found if lens.id.startswith(tag)) == expected


@pytest.mark.parametrize(
    "field, needle, expected",
    (
        ("name", "{tag}", ("{tag} adhoc", "{tag} nightly", "{tag} weekly 100%_x")),
        ("name", "{tag} W", ("{tag} weekly 100%_x",)),
        ("name", "{tag}_", ()),
        ("name", "{tag} weekly 100%", ("{tag} weekly 100%_x",)),
        ("name", "{tag} weekly 1%", ()),
        ("agent", "{tag}", ("{tag}-billing", "{tag}-researcher")),
        ("agent", "{tag}-RESEARCH", ("{tag}-researcher",)),
    ),
)
@pytest.mark.asyncio
async def test_values_list_distinct_field_values_containing_the_needle(
    searchable_lenses: tuple[LensRepository, str], field: LensField, needle: str, expected: tuple[str, ...]
) -> None:
    repo, tag = searchable_lenses
    assert await repo.values(field, needle.format(tag=tag), 100) == tuple(e.format(tag=tag) for e in expected)


@pytest.mark.asyncio
async def test_values_include_derived_status_and_schedule(searchable_lenses: tuple[LensRepository, str]) -> None:
    repo, _ = searchable_lenses
    assert {"failed", "completed", "never"} <= set(await repo.values("status", "", 100))
    assert {"watching", "paused"} <= set(await repo.values("schedule", "", 100))


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

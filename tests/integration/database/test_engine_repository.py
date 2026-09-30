import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Final
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.engine.models import Check, Engine, EngineSettings, Scope, Worker
from litellm.proxy.engine.repository import EngineRepository, WriterDatabase
from litellm.proxy.engine.state import claim_job, queue_job


@pytest_asyncio.fixture(loop_scope="function")
async def engine_db() -> AsyncIterator[Prisma]:
    async with Prisma(datasource={"url": os.environ["DATABASE_URL"]}) as db:
        yield db


@pytest.mark.asyncio
async def test_concurrent_workers_cannot_both_acquire_the_same_job(engine_db: Prisma) -> None:
    now: Final = datetime.now(UTC)
    scope: Final = Scope(team_id=uuid4().hex)
    repo: Final = EngineRepository(WriterDatabase(PrismaWrapper(engine_db)))
    engine: Final = Engine(
        id=uuid4().hex,
        scope=scope,
        settings=EngineSettings(name="Lease test", model="test", checks=(Check(id="c", instruction="Find retries"),)),
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    await repo.create(queue_job(engine, now, uuid4().hex))
    try:
        workers: Final = tuple(Worker(id=uuid4().hex, name="worker", scope=scope, last_seen=now) for _ in range(2))
        results: Final = await asyncio.gather(
            *(repo.update(engine.id, lambda e, w=w: claim_job(e, w, now)) for w in workers)
        )
        stored: Final = await repo.get(engine.id)
        assert stored is not None
        assert stored.jobs[0].attempts == 1
        assert stored.jobs[0].worker_id in tuple(w.id for w in workers)
        assert tuple(r.jobs[0].worker_id for r in results if r) == (stored.jobs[0].worker_id, stored.jobs[0].worker_id)
    finally:
        await engine_db.execute_raw('DELETE FROM "LiteLLM_Engine" WHERE id=$1', engine.id)


@pytest.mark.asyncio
async def test_heartbeat_never_restores_revoked_access(engine_db: Prisma) -> None:
    now: Final = datetime.now(UTC)
    repo: Final = EngineRepository(WriterDatabase(PrismaWrapper(engine_db)))
    worker: Final = Worker(id=uuid4().hex, name="worker", scope=Scope(team_id=uuid4().hex), last_seen=now)
    token_hash: Final = uuid4().hex
    await repo.save_worker(worker, token_hash)
    try:
        await repo.save_worker(worker.model_copy(update={"revoked": True}))
        await repo.heartbeat(worker.id, now.isoformat())
        stored: Final = await repo.worker(token_hash)
        assert stored is not None and stored.revoked is True
    finally:
        await engine_db.execute_raw('DELETE FROM "LiteLLM_EngineWorker" WHERE id=$1', worker.id)

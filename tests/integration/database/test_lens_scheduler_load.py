import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Final
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma
from pydantic import TypeAdapter
from typing_extensions import LiteralString

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.endpoints import claim_due
from litellm.proxy.lens.models import Evidence, Finding, Lens, LensSettings, Scope, Worker
from litellm.proxy.lens.repository import Database, LensRepository, Row, WriterDatabase
from litellm.proxy.lens.state import current_job


@pytest_asyncio.fixture(loop_scope="function")
async def lens_db() -> AsyncIterator[Prisma]:
    async with Prisma(datasource={"url": os.environ["DATABASE_URL"]}) as db:
        yield db


class ReadMeter:
    def __init__(self) -> None:
        self.batches: tuple[tuple[int, ...], ...] = ()

    def record(self, document_sizes: tuple[int, ...]) -> None:
        self.batches = (*self.batches, document_sizes)

    @property
    def document_count(self) -> int:
        return sum(len(batch) for batch in self.batches)

    @property
    def total_bytes(self) -> int:
        return sum(sum(batch) for batch in self.batches)


class MeasuredDatabase:
    def __init__(self, database: WriterDatabase, meter: ReadMeter) -> None:
        self.database: Final = database
        self.meter: Final = meter

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        rows: Final = await self.database.query_raw(query, *args)
        if 'FROM "LiteLLM_Lens"' in query and "WHERE id" not in query:
            documents: Final = TypeAdapter(tuple[Row, ...]).validate_python(rows)
            self.meter.record(
                tuple(len(json.dumps(row.data, separators=(",", ":")).encode("utf-8")) for row in documents)
            )
        return rows

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        return await self.database.execute_raw(query, *args)

    def transaction(self) -> AbstractAsyncContextManager[Database]:
        return self.database.transaction()


def _large_lens(lens_id: str, scope: Scope, now: datetime, next_run_at: datetime) -> Lens:
    findings: Final = tuple(
        Finding(
            id=f"f{index}",
            title=f"Issue {index}",
            description="Repeated operation returns an unexpected result.",
            check_id="behavior",
            evidence=(
                Evidence(
                    execution_id=f"t{index}",
                    span_id=f"s{index}",
                    quote="Unexpected result",
                ),
            ),
            first_seen=now,
            last_seen=now,
            revision=1,
        )
        for index in range(100)
    )
    return Lens(
        id=lens_id,
        scope=scope,
        settings=LensSettings(
            name="Claim scheduler load",
            model="analysis",
            context="Find unexpected behavior",
            enabled=True,
        ),
        created_at=now,
        next_run_at=next_run_at,
        findings=findings,
        budget_month=now.strftime("%Y-%m"),
    )


def _due_lens(lens_id: str, scope: Scope, now: datetime, model: str, next_run_at: datetime) -> Lens:
    return Lens(
        id=lens_id,
        scope=scope,
        settings=LensSettings(
            name="Claim paging test",
            model=model,
            context="Find unexpected behavior",
            enabled=True,
        ),
        created_at=now,
        next_run_at=next_run_at,
        budget_month=now.strftime("%Y-%m"),
    )


async def _supports_model(_worker: Worker, _settings: LensSettings) -> bool:
    return True


async def _supports_supported_model(_worker: Worker, settings: LensSettings) -> bool:
    return settings.model == "supported"


@pytest.mark.asyncio
async def test_claim_due_reaches_a_supported_lens_behind_a_full_page_of_unsupported_ones(
    lens_db: Prisma,
) -> None:
    now: Final = datetime.now(timezone.utc).replace(microsecond=0)
    scope: Final = Scope(team_id=uuid4().hex)
    worker: Final = Worker(id=uuid4().hex, name="paging-test-worker", scope=scope, last_seen=now)
    unsupported_at: Final = now - timedelta(minutes=5)
    supported_at: Final = now - timedelta(minutes=1)
    unsupported: Final = tuple(_due_lens(uuid4().hex, scope, now, "unsupported", unsupported_at) for _ in range(25))
    supported: Final = _due_lens(uuid4().hex, scope, now, "supported", supported_at)
    candidates: Final = (*unsupported, supported)
    repository: Final = LensRepository(WriterDatabase(PrismaWrapper(lens_db)))
    await asyncio.gather(*(repository.create(candidate) for candidate in candidates))
    try:
        claim: Final = await claim_due(worker, now, repository, _supports_supported_model)
        assert claim is not None
        assert claim.lens_id == supported.id
        assert claim.job.status == "running"
    finally:
        await lens_db.execute_raw(
            'DELETE FROM "LiteLLM_Lens" WHERE id=ANY($1::text[])',
            tuple(candidate.id for candidate in candidates),
        )


@pytest.mark.asyncio
async def test_lens_claim_reads_scale_with_due_lenses_not_total_lenses(lens_db: Prisma) -> None:
    now: Final = datetime.now(timezone.utc).replace(microsecond=0)
    scope: Final = Scope(team_id=uuid4().hex)
    worker: Final = Worker(id=uuid4().hex, name="load-test-worker", scope=scope, last_seen=now)
    due_lens: Final = _large_lens(uuid4().hex, scope, now, now - timedelta(seconds=1))
    initial_future: Final = tuple(_large_lens(uuid4().hex, scope, now, now + timedelta(days=1)) for _ in range(20))
    additional_future: Final = tuple(_large_lens(uuid4().hex, scope, now, now + timedelta(days=1)) for _ in range(200))
    ids: Final = tuple(lens.id for lens in (due_lens, *initial_future, *additional_future))
    writer: Final = WriterDatabase(PrismaWrapper(lens_db))
    seed_repository: Final = LensRepository(writer)
    await asyncio.gather(*(seed_repository.create(lens) for lens in (due_lens, *initial_future)))
    try:
        before_meter: Final = ReadMeter()
        before_repository: Final = LensRepository(MeasuredDatabase(writer, before_meter))
        before_started: Final = perf_counter()
        before_claim: Final = await claim_due(worker, now, before_repository, _supports_model)
        before_seconds: Final = perf_counter() - before_started
        assert before_claim is not None
        assert before_claim.lens_id == due_lens.id
        assert before_claim.job.status == "running"
        claimed_lens: Final = await seed_repository.get(due_lens.id)
        assert claimed_lens is not None
        assert current_job(claimed_lens) == before_claim.job
        await seed_repository.update(
            due_lens.id,
            lambda lens: lens.model_copy(update={"jobs": (), "next_run_at": now - timedelta(seconds=1)}),
            attempts=1,
        )
        await asyncio.gather(*(seed_repository.create(lens) for lens in additional_future))
        after_meter: Final = ReadMeter()
        after_repository: Final = LensRepository(MeasuredDatabase(writer, after_meter))
        after_started: Final = perf_counter()
        after_claim: Final = await claim_due(worker, now, after_repository, _supports_model)
        after_seconds: Final = perf_counter() - after_started
        assert after_claim is not None
        assert after_claim.lens_id == due_lens.id
        assert after_claim.job.status == "running"
        sys.stdout.write(
            f"claim read: before={before_meter.total_bytes} bytes, {before_seconds:.4f}s; "
            f"after={after_meter.total_bytes} bytes, {after_seconds:.4f}s\n"
        )
        assert before_meter.document_count == after_meter.document_count == 1
        assert before_meter.total_bytes == after_meter.total_bytes
    finally:
        await lens_db.execute_raw('DELETE FROM "LiteLLM_Lens" WHERE id=ANY($1::text[])', ids)

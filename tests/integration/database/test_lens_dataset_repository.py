import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Final
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma
from prisma.types import DatasourceOverride

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.dataset_repository import DatasetRepository, StoredSummary
from litellm.proxy.lens.models import CaseSource, Dataset, DatasetCase, DatasetMessage, DatasetSummary
from litellm.proxy.lens.repository import WriterDatabase

SAVED_AT: Final = datetime(2026, 3, 1, 12, 0, 0, 123000, tzinfo=timezone.utc)


@pytest_asyncio.fixture(loop_scope="function")
async def lens_db() -> AsyncIterator[Prisma]:
    async with Prisma(datasource=DatasourceOverride(url=os.environ["DATABASE_URL"])) as db:
        yield db


@pytest_asyncio.fixture(loop_scope="function")
async def dataset_ids(lens_db: Prisma) -> AsyncIterator[tuple[str, ...]]:
    ids: Final = tuple(uuid4().hex for _ in range(3))
    yield ids
    await lens_db.execute_raw('DELETE FROM "LiteLLM_LensDataset" WHERE id = ANY($1::text[])', list(ids))


def _repo(db: Prisma) -> DatasetRepository:
    return DatasetRepository(WriterDatabase(PrismaWrapper(db)))


def _dataset(dataset_id: str, revision: int, case_count: int, team_id: str = "team-a") -> Dataset:
    return Dataset(
        id=dataset_id,
        name=f"Dataset r{revision}",
        agent_name="support-agent",
        team_id=team_id,
        created_at=SAVED_AT,
        revision=revision,
        created_by="user-1",
        cases=tuple(
            DatasetCase(
                id=f"case-{revision}-{i}",
                messages=(DatasetMessage(role="user", content=f"question {i}"),),
                reply=f"answer {i}",
                source=CaseSource(trace_id=f"trace-{i}"),
            )
            for i in range(case_count)
        ),
    )


@pytest.mark.asyncio
async def test_get_returns_requested_revision_and_defaults_to_latest(
    lens_db: Prisma, dataset_ids: tuple[str, ...]
) -> None:
    repo: Final = _repo(lens_db)
    dataset_id: Final = dataset_ids[0]
    revisions: Final = tuple(_dataset(dataset_id, revision, revision) for revision in (1, 3, 2))
    assert [await repo.insert(dataset, SAVED_AT) for dataset in revisions] == [True, True, True]

    assert await repo.get(dataset_id, revision=1) == revisions[0]
    assert await repo.get(dataset_id, revision=2) == revisions[2]
    assert await repo.get(dataset_id) == revisions[1]


@pytest.mark.asyncio
async def test_get_unknown_dataset_or_revision_returns_none(lens_db: Prisma, dataset_ids: tuple[str, ...]) -> None:
    repo: Final = _repo(lens_db)
    assert await repo.insert(_dataset(dataset_ids[0], 1, 1), SAVED_AT)

    assert await repo.get(dataset_ids[1]) is None
    assert await repo.get(dataset_ids[1], revision=1) is None
    assert await repo.get(dataset_ids[0], revision=2) is None


@pytest.mark.asyncio
async def test_inserting_an_existing_revision_is_rejected_and_keeps_the_first(
    lens_db: Prisma, dataset_ids: tuple[str, ...]
) -> None:
    repo: Final = _repo(lens_db)
    original: Final = _dataset(dataset_ids[0], 1, 1)
    overwrite: Final = _dataset(dataset_ids[0], 1, 4).model_copy(update={"name": "Overwritten"})

    assert await repo.insert(original, SAVED_AT) is True
    assert await repo.insert(overwrite, SAVED_AT + timedelta(days=1)) is False

    assert await repo.get(dataset_ids[0], revision=1) == original
    summaries: Final = tuple(s for s in await repo.summaries() if s.summary.id == dataset_ids[0])
    assert tuple(s.summary.updated_at for s in summaries) == (SAVED_AT,)


@pytest.mark.asyncio
async def test_summaries_list_each_dataset_once_at_its_latest_revision_newest_first(
    lens_db: Prisma, dataset_ids: tuple[str, ...]
) -> None:
    repo: Final = _repo(lens_db)
    older, newer, single = dataset_ids
    writes: Final = (
        (_dataset(older, 1, 1, "team-a"), SAVED_AT),
        (_dataset(older, 2, 3, "team-a"), SAVED_AT + timedelta(minutes=1)),
        (_dataset(newer, 1, 5, "team-b"), SAVED_AT + timedelta(minutes=2)),
        (_dataset(newer, 2, 2, "team-b"), SAVED_AT + timedelta(minutes=4)),
        (_dataset(single, 1, 0, ""), SAVED_AT + timedelta(minutes=3)),
    )
    assert [await repo.insert(dataset, saved_at) for dataset, saved_at in writes] == [True] * len(writes)

    summaries: Final = tuple(s for s in await repo.summaries() if s.summary.id in dataset_ids)

    assert summaries == (
        StoredSummary(
            team_id="team-b",
            summary=DatasetSummary(
                id=newer,
                name="Dataset r2",
                agent_name="support-agent",
                revision=2,
                case_count=2,
                updated_at=SAVED_AT + timedelta(minutes=4),
            ),
        ),
        StoredSummary(
            team_id="",
            summary=DatasetSummary(
                id=single,
                name="Dataset r1",
                agent_name="support-agent",
                revision=1,
                case_count=0,
                updated_at=SAVED_AT + timedelta(minutes=3),
            ),
        ),
        StoredSummary(
            team_id="team-a",
            summary=DatasetSummary(
                id=older,
                name="Dataset r2",
                agent_name="support-agent",
                revision=2,
                case_count=3,
                updated_at=SAVED_AT + timedelta(minutes=1),
            ),
        ),
    )

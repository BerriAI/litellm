from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy.lens.models import Check, Lens, LensSettings, Scope
from litellm.proxy.lens.repository import UPDATE_ATTEMPTS, LensRepository, Row

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)
STORED: Final = Lens(
    id="lens",
    scope=Scope(team_id="alpha"),
    settings=LensSettings(
        name="Swarm", model="cerebras/gpt-oss-120b", checks=(Check(id="c", instruction="Find loops"),)
    ),
    created_at=NOW,
    next_run_at=NOW,
    budget_month=NOW.strftime("%Y-%m"),
)


class ContendedDatabase:
    def __init__(self, losses: int) -> None:
        self.losses: Final = losses
        self.writes = 0  # rebind-ok: counts write attempts made under contention

    async def query_raw(self, query: str, *args: object) -> object:
        if query.startswith("SELECT data FROM"):
            return (Row(data=STORED.model_dump(mode="json")),)
        self.writes += 1
        return (Row(data=1 if self.writes > self.losses else 0),)

    async def execute_raw(self, query: str, *args: object) -> int:
        return 0


async def no_wait(_: float) -> None:
    return None


def renamed(lens: Lens) -> Lens:
    return lens.model_copy(update={"settings": lens.settings.model_copy(update={"name": "Swarm (renamed)"})})


@pytest.mark.asyncio
async def test_update_survives_the_contention_of_a_fast_model_writing_every_review() -> None:
    db: Final = ContendedDatabase(losses=12)
    updated: Final = await LensRepository(db, sleep=no_wait).update("lens", renamed)
    assert updated is not None
    assert updated.settings.name == "Swarm (renamed)"
    assert db.writes == 13


@pytest.mark.asyncio
async def test_update_backs_off_between_lost_writes_and_gives_up_after_the_limit() -> None:
    waits: list[float] = []  # mutable-ok: records each backoff the repository requests

    async def record(seconds: float) -> None:
        waits.append(seconds)

    db: Final = ContendedDatabase(losses=UPDATE_ATTEMPTS)
    assert await LensRepository(db, sleep=record).update("lens", renamed) is None
    assert db.writes == UPDATE_ATTEMPTS
    assert len(waits) == UPDATE_ATTEMPTS
    assert all(w >= 0 for w in waits)


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", (False, True))
async def test_checkpoint_and_progress_commit_together_or_roll_back_together(write_fails: bool) -> None:
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    from litellm.proxy.lens.models import Extraction, Progress, Review
    from litellm.proxy.lens.repository import Database
    from litellm.proxy.lens.state import claim_job, queue_job, replace_job
    from tests.unit.proxy.lens.test_state import lens, worker

    claimed: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW)
    job: Final = claimed.jobs[0].model_copy(update={"lease_until": datetime.max.replace(tzinfo=timezone.utc)})
    initial: Final = replace_job(claimed, job)
    review: Final = Review(
        execution_id="trace",
        trace_id="trace",
        agent="agent",
        name="task",
        model="analysis",
        duration_ms=1,
        at=NOW,
        content_version="content",
        extraction=Extraction(),
    )

    class CheckpointDatabase:
        def __init__(self) -> None:
            self.stored = initial
            self.checkpoint: Review | None = None

        @asynccontextmanager
        async def transaction(self) -> AsyncGenerator[Database]:
            previous: Final = self.stored
            checkpoint: Final = self.checkpoint
            try:
                yield self
            except Exception:
                self.stored = previous
                self.checkpoint = checkpoint
                raise

        async def query_raw(self, query: str, *args: object) -> tuple[Row, ...]:
            if query.startswith("SELECT data FROM"):
                return (Row(data=self.stored.model_dump(mode="json")),)
            assert isinstance(args[0], str)
            self.stored = Lens.model_validate_json(args[0])
            return (Row(data=1),)

        async def execute_raw(self, query: str, *args: object) -> int:
            assert isinstance(args[3], str)
            self.checkpoint = Review.model_validate_json(args[3])
            if write_fails:
                raise OSError("Checkpoint storage unavailable")
            return 1

    db: Final = CheckpointDatabase()
    repo: Final = LensRepository(db)
    if write_fails:
        with pytest.raises(OSError, match="Checkpoint storage unavailable"):
            await repo.progress(initial.id, job, Progress(review=review))
        assert await repo.get(initial.id) == initial
        assert db.checkpoint is None
        return
    updated: Final = await repo.progress(initial.id, job, Progress(review=review))
    assert updated is not None and updated == await repo.get(initial.id)
    assert db.checkpoint == review
    assert updated.jobs[0].reviewed == 1
    assert updated.jobs[0].reviews == (review.model_copy(update={"extraction": None, "content_version": ""}),)

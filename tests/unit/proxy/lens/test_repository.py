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

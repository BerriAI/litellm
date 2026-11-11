from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.performance_alert import (
    Claim,
    ResourceUnreadableError,
    SweepOwnershipLostError,
    SweepState,
    advance,
    new_sweep,
)
from litellm.proxy.spend_tracking.performance_alerts import PAGE_SIZE, RUN_BUDGET_SECONDS, run_performance_alerts

NOW: Final = datetime(2026, 1, 9, 2, 0, tzinfo=timezone.utc)
DAY: Final = date(2026, 1, 8)


@dataclass
class Store:
    """Each scope holds `per_scope` resources. Resources named in `unreadable` fail any page that includes them."""

    state: SweepState | None = None
    claimable: bool = True
    index_ready: bool = True
    per_scope: int = 3
    unreadable: frozenset[str] = frozenset()
    pages: list[tuple[str, int]] = field(default_factory=list)
    saved: list[SweepState] = field(default_factory=list)

    async def claim(self, interval: timedelta) -> Claim | None:
        return Claim("owner", self.state) if self.claimable else None

    async def key_index_ready(self) -> bool:
        return self.index_ready

    def _keep(self, state: SweepState) -> None:
        self.saved.append(state)
        self.state = state

    def _next(self, state: SweepState, limit: int) -> tuple[str, ...]:
        if state.scope == "gateway":
            return ("gateway",)
        names: Final = tuple(f"{state.scope}-{index}" for index in range(self.per_scope))
        after: Final = tuple(name for name in names if name > state.cursor)
        return tuple(name for name in after if not state.wrapped or name <= state.start)[:limit]

    async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState:
        self.pages.append((state.scope, limit))
        ids: Final = self._next(state, limit)
        if self.unreadable.intersection(ids):
            raise ResourceUnreadableError()
        advanced: Final = advance(state, ids, limit)
        self._keep(advanced)
        return advanced

    async def skip_next(self, owner: str, state: SweepState) -> SweepState:
        ids: Final = self._next(state, 1)
        advanced: Final = advance(state, ids, 1, skipped=len(ids))
        self._keep(advanced)
        return advanced


async def _no_sleep(seconds: float) -> None:
    return None


async def _run(store: Store, clock: list[float] | None = None) -> None:
    ticks: Final = clock if clock is not None else [0.0]
    await run_performance_alerts(store, now=lambda: NOW, clock=lambda: ticks[0], sleep=_no_sleep)


@pytest.mark.asyncio
async def test_a_fresh_sweep_checks_every_scope_and_completes() -> None:
    store: Final = Store()
    await _run(store)
    assert store.state is not None and store.state.completed
    assert (store.state.day, store.state.checked, store.state.skipped) == (DAY, 1 + 3 * 4, 0)


@pytest.mark.asyncio
async def test_no_work_happens_without_the_claim() -> None:
    store: Final = Store(claimable=False)
    await _run(store)
    assert store.pages == [] and store.saved == []


@pytest.mark.asyncio
async def test_a_completed_day_does_no_more_work() -> None:
    store: Final = Store(state=new_sweep(DAY).model_copy(update={"completed": True}))
    await _run(store)
    assert store.pages == []


@pytest.mark.asyncio
async def test_an_unfinished_previous_day_resumes_its_scope_where_it_stopped_and_still_checks_everything() -> None:
    store: Final = Store(state=SweepState(day=DAY - timedelta(days=1), scope="user", cursor="user-1"))
    await _run(store)
    assert store.saved[0].day == DAY and store.pages[0] == ("gateway", PAGE_SIZE)
    users: Final = tuple(state for state in store.saved if state.scope == "user")
    assert (users[0].cursor, users[0].wrapped) == ("user-1", False)
    assert any(state.wrapped for state in users)
    assert store.state is not None and store.state.completed and store.state.checked == 1 + 3 * 4


@pytest.mark.asyncio
async def test_the_run_resumes_the_saved_position_for_the_same_day() -> None:
    store: Final = Store(state=SweepState(day=DAY, scope="key", cursor="key-1", checked=10))
    await _run(store)
    assert store.pages[0] == ("key", PAGE_SIZE)
    assert store.state is not None and store.state.checked == 11


@pytest.mark.asyncio
async def test_an_unreadable_resource_is_retried_alone_then_skipped() -> None:
    store: Final = Store(unreadable=frozenset({"team-1"}))
    await _run(store)
    assert ("team", 1) in store.pages
    assert store.state is not None and store.state.completed
    assert (store.state.checked, store.state.skipped) == (12, 1)


@pytest.mark.asyncio
async def test_keys_wait_for_their_index_without_finishing_the_day_then_run_once_it_exists() -> None:
    store: Final = Store(index_ready=False)
    await _run(store)
    assert all(scope != "key" for scope, _ in store.pages)
    assert store.state is not None and not store.state.completed and store.state.scope == "key"
    store.index_ready = True
    await _run(store)
    assert store.state is not None and store.state.completed and store.state.checked == 1 + 3 * 4


@pytest.mark.asyncio
async def test_the_run_stops_at_its_time_budget_and_keeps_its_position() -> None:
    clock: Final = [0.0]
    store: Final = Store(per_scope=10_000)

    async def tick(seconds: float) -> None:
        clock[0] += RUN_BUDGET_SECONDS * 0.4

    await run_performance_alerts(store, now=lambda: NOW, clock=lambda: clock[0], sleep=tick)
    assert len(store.pages) == 3
    assert store.state is not None and not store.state.completed


@pytest.mark.asyncio
async def test_losing_the_claim_mid_run_stops_quietly() -> None:
    class TakenOver(Store):
        async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState:
            raise SweepOwnershipLostError()

    store: Final = TakenOver()
    await _run(store)
    assert store.saved == []


@pytest.mark.asyncio
async def test_a_database_wide_failure_ends_the_run_without_skipping_anything() -> None:
    class Unavailable(Store):
        async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState:
            from prisma.errors import RawQueryError

            raise RawQueryError({"user_facing_error": {"message": "FATAL: sorry, too many clients already"}})

    store: Final = Unavailable()
    await _run(store)
    assert store.saved == []

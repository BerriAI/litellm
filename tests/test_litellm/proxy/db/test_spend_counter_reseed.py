"""Window-spend reads in ``SpendCounterReseed``.

The maintained ``LiteLLM_BudgetWindowSpend`` row replaces a per-request
``LiteLLM_SpendLogs`` range scan, so these pin *when* the aggregate is still
allowed to run: only when the row is missing or belongs to an older window.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.constants import PROXY_DB_LOOKUP_MAX_CONCURRENCY
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed

WINDOW_START = datetime(2026, 8, 1, tzinfo=timezone.utc)


class _FakeFindUniqueTable:
    def __init__(self, row: SimpleNamespace | None, error: Exception | None = None) -> None:
        self._row = row
        self._error = error
        self.where_clauses: list[dict] = []

    async def find_unique(self, where: dict):
        self.where_clauses.append(where)
        if self._error is not None:
            raise self._error
        return self._row


class _FakeSpendLogsTable:
    def __init__(self, total: float) -> None:
        self._total = total
        self.call_count = 0

    async def group_by(self, by: list[str], where: dict, sum: dict):
        self.call_count += 1
        return [{by[0]: where.get(by[0]), "_sum": {"spend": self._total}}]


class _InFlightCountingTable:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.001)
        self.in_flight -= 1
        return SimpleNamespace(token=where["token"], spend=1.0)


class _FakePrismaClient:
    def __init__(
        self,
        row: SimpleNamespace | None = None,
        spend_logs_total: float = 0.0,
        error: Exception | None = None,
        end_user_row: SimpleNamespace | None = None,
        end_user_error: Exception | None = None,
    ) -> None:
        self.db = SimpleNamespace(
            litellm_budgetwindowspend=_FakeFindUniqueTable(row=row, error=error),
            litellm_spendlogs=_FakeSpendLogsTable(total=spend_logs_total),
            litellm_endusertable=_FakeFindUniqueTable(row=end_user_row, error=end_user_error),
            litellm_verificationtoken=_InFlightCountingTable(),
        )


def _row(window_start: datetime, spend: float) -> SimpleNamespace:
    return SimpleNamespace(window_start=window_start, spend=spend)


@pytest.mark.asyncio
async def test_window_from_table_reads_row_by_primary_key():
    """The lookup must use the table's own entity_type values ("key"), not the
    "Key"/"Team" labels the counter keys and spend-log aggregates use."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 4.5
    assert prisma.db.litellm_budgetwindowspend.where_clauses == [
        {
            "entity_type_entity_id_window_duration": {
                "entity_type": "key",
                "entity_id": "tok-1",
                "window_duration": "30d",
            }
        }
    ]


@pytest.mark.asyncio
async def test_window_from_table_maps_team_entity_type():
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 9.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Team",
        entity_id="team-1",
        window_duration="1d",
        expected_window_start=WINDOW_START,
    )

    assert result == 9.0
    inner = prisma.db.litellm_budgetwindowspend.where_clauses[0]["entity_type_entity_id_window_duration"]
    assert inner["entity_type"] == "team"


@pytest.mark.asyncio
async def test_window_from_table_trusts_row_newer_than_expected_window():
    """Regression: a pod holding a stale ``reset_at`` computes an expected start
    behind a window another pod already rolled. Trusting only an exact match
    would make it re-add the previous window's spend to the current one."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START + timedelta(days=1), 2.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 2.0


@pytest.mark.asyncio
async def test_window_from_table_rejects_row_from_previous_window():
    prisma = _FakePrismaClient(row=_row(WINDOW_START - timedelta(seconds=1), 99.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_table_treats_naive_row_timestamp_as_utc():
    """The column is ``timestamp(3)``, so a driver that hands back a naive value
    must still compare against the tz-aware expected start."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START.replace(tzinfo=None), 3.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma, entity_type",
    [
        (_FakePrismaClient(row=None), "Key"),
        (_FakePrismaClient(row=_row(WINDOW_START, 1.0)), "User"),
        (_FakePrismaClient(error=RuntimeError("connection reset")), "Key"),
        (None, "Key"),
    ],
)
async def test_window_from_table_returns_none_without_a_usable_row(prisma, entity_type):
    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type=entity_type,
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_db_prefers_the_row_over_the_spend_logs_aggregate():
    """The aggregate range-scans an unindexed table; a current row must keep it
    from running at all."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=100.0)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert prisma.db.litellm_spendlogs.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [None, _row(WINDOW_START - timedelta(seconds=1), 99.0)],
    ids=["missing_row", "previous_window_row"],
)
async def test_window_from_db_falls_back_to_spend_logs(row):
    prisma = _FakePrismaClient(row=row, spend_logs_total=7.25)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 7.25
    assert prisma.db.litellm_spendlogs.call_count == 1


@pytest.mark.asyncio
async def test_window_from_db_without_a_duration_skips_the_row_lookup():
    """Callers that cannot name the window (no PK) keep the pre-table behavior."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=7.25)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration=None,
        window_start=WINDOW_START,
    )

    assert result == 7.25
    assert prisma.db.litellm_budgetwindowspend.where_clauses == []


@pytest.mark.asyncio
async def test_window_from_db_logs_floor_only_when_requested():
    """Default keeps the steady-state short-circuit (no aggregate scan); a
    reseed (``include_logs_floor=True``) must also read the aggregate because
    the maintained row lags queued increments by up to one flush interval."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=7.25)

    default_result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )
    # Snapshot between the two calls: the default call must not scan the
    # aggregate (steady-state path has a performance pin on it).
    calls_after_default = prisma.db.litellm_spendlogs.call_count
    floored_result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
        include_logs_floor=True,
    )

    assert default_result == 4.5
    assert calls_after_default == 0
    # max(row, aggregate): the aggregate cannot lag what pods already counted,
    # so it floors a lagging row instead of the row overriding it.
    assert floored_result == 7.25
    assert prisma.db.litellm_spendlogs.call_count == calls_after_default + 1


@pytest.mark.asyncio
async def test_window_from_db_logs_floor_keeps_row_when_aggregate_lower():
    """The row is the running total once the aggregate falls behind (rolled
    window, retention cleanup): the floor must not lower the reseed either."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 9.0), spend_logs_total=4.5)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
        include_logs_floor=True,
    )

    assert result == 9.0


@pytest.mark.asyncio
async def test_coalesced_window_seeds_a_cold_counter_from_the_row():
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=100.0)
    cache = DualCache()
    counter_key = "spend:key:tok-1:window:30d"

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key=counter_key,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert cache.in_memory_cache.get_cache(key=counter_key) == 4.5
    assert prisma.db.litellm_spendlogs.call_count == 0


class _FakeRedisCache:
    """Minimal Redis stand-in: NX set behaves like the real one and can be made
    to fail to exercise the un-warmed-snapshot paths."""

    def __init__(self, *, warm_error: Exception | None = None, race_value: float | None = None,
                 nx_lost_vanished: bool = False) -> None:
        self._warm_error = warm_error
        self._race_value = race_value  # appears only after the NX attempt (winner landed mid-read)
        self._nx_lost_vanished = nx_lost_vanished  # NX fails but the value vanished (TTL race)
        self.store: dict[str, float] = {}

    async def async_increment(self, key: str, value: float) -> float:
        self.store[key] = self.store.get(key, 0.0) + float(value)
        return self.store[key]

    async def async_get_cache(self, key: str):
        if self._warm_error is not None and key not in self.store:
            raise self._warm_error
        return self.store.get(key)

    async def async_set_cache(self, key: str, value, nx: bool = False):
        if self._warm_error is not None:
            raise self._warm_error
        if self._race_value is not None:
            # Simulate losing the race: the winner seeded while our DB read was
            # in flight, so our NX fails and the winner's value is visible.
            self.store[key] = self._race_value
            return False
        if self._nx_lost_vanished:
            # NX fails but a re-read finds nothing (TTL raced the winner away):
            # the last-resort increment path must seed the counter.
            return False
        if nx and key in self.store:
            return False
        self.store[key] = float(value)
        return True

    async def async_increment(self, key: str, value: float):
        if self._warm_error is not None:
            raise self._warm_error
        self.store[key] = self.store.get(key, 0.0) + float(value)
        return self.store[key]


@pytest.mark.asyncio
async def test_coalesced_window_returns_none_when_warm_fails():
    """Greptile P1 (failed warm loses coordination): when the counter cannot be
    warmed in Redis, the DB snapshot never coordinated with concurrent
    reservations (they skipped the cold counter), so it must NOT be served as
    an admission value."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=4.5)
    redis = _FakeRedisCache(warm_error=RuntimeError("redis down"))
    cache = DualCache(redis_cache=redis)

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
        include_logs_floor=True,
    )

    assert result is None


@pytest.mark.asyncio
async def test_coalesced_window_adopts_winner_when_nx_loses():
    """Two concurrent reseeds of the same counter: the NX loser must return the
    winner's value (the one increments coordinate against), not its own
    un-coordinated snapshot."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=4.5)
    redis = _FakeRedisCache(race_value=9.0)  # winner seeds 9.0 while our DB read is in flight
    cache = DualCache(redis_cache=redis)

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 9.0
    assert cache.in_memory_cache.get_cache(key="spend:key:tok-1:window:30d") == 9.0


@pytest.mark.asyncio
async def test_coalesced_window_floors_row_at_aggregate():
    """Greptile P1 (lagging persisted spend): a row lagging increments queued on
    other pods must reseed the counter at the aggregate floor, not restore a
    stale-low counter."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=7.25)
    cache = DualCache()

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
        include_logs_floor=True,
    )

    assert result == 7.25
    assert cache.in_memory_cache.get_cache(key="spend:key:tok-1:window:30d") == 7.25


@pytest.mark.asyncio
async def test_end_user_from_db_reads_the_end_user_row_by_user_id():
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=0.0))

    result: Final = await SpendCounterReseed.end_user_from_db(
        prisma_client=prisma, counter_key="spend:end_user:customer-42"
    )

    assert result == 0.0
    assert prisma.db.litellm_endusertable.where_clauses == [{"user_id": "customer-42"}]


@pytest.mark.asyncio
async def test_end_user_from_db_returns_the_recorded_spend():
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=12.5))

    assert (
        await SpendCounterReseed.end_user_from_db(prisma_client=prisma, counter_key="spend:end_user:customer-42")
        == 12.5
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("counter_key", ["spend:key:hashed", "spend:team:t1", "spend:tag:t1"])
async def test_end_user_from_db_ignores_other_counter_kinds_without_touching_the_db(counter_key):
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="x", spend=5.0))

    assert await SpendCounterReseed.end_user_from_db(prisma_client=prisma, counter_key=counter_key) is None
    assert prisma.db.litellm_endusertable.where_clauses == []


@pytest.mark.asyncio
async def test_end_user_from_db_returns_none_without_a_row_a_client_or_on_db_error():
    assert (
        await SpendCounterReseed.end_user_from_db(prisma_client=None, counter_key="spend:end_user:customer-42")
        is None
    )
    assert (
        await SpendCounterReseed.end_user_from_db(
            prisma_client=_FakePrismaClient(end_user_row=None), counter_key="spend:end_user:customer-42"
        )
        is None
    )
    assert (
        await SpendCounterReseed.end_user_from_db(
            prisma_client=_FakePrismaClient(end_user_error=RuntimeError("db down")),
            counter_key="spend:end_user:customer-42",
        )
        is None
    )


@pytest.mark.asyncio
async def test_from_db_bounds_in_flight_prisma_requests_across_counter_keys():
    """Per-counter singleflight only collapses duplicates of one key. A cold-cache burst
    over many distinct keys must still not flood the prisma engine HTTP pool (LIT-6435)."""
    prisma: Final = _FakePrismaClient()
    burst: Final = PROXY_DB_LOOKUP_MAX_CONCURRENCY * 5

    results: Final = await asyncio.gather(
        *(SpendCounterReseed.from_db(prisma_client=prisma, counter_key=f"spend:key:hashed-{i}") for i in range(burst))
    )

    assert results == [1.0] * burst
    assert prisma.db.litellm_verificationtoken.max_in_flight == PROXY_DB_LOOKUP_MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_from_db_still_never_reads_the_end_user_row():
    """A cold end-user counter keeps seeding from the cached end-user object the auth
    path already loaded; the row is read only as the budget floor."""
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=5.0))

    assert await SpendCounterReseed.from_db(prisma_client=prisma, counter_key="spend:end_user:customer-42") is None
    assert prisma.db.litellm_endusertable.where_clauses == []


@pytest.mark.asyncio
async def test_coalesced_window_returns_none_when_db_has_no_value():
    """Codecov line: with no readable window value in the DB (row missing and
    the aggregate read failing), coalesced_window must return None so callers
    fall back to their conservative value - never a fabricated 0.0."""
    prisma = _FakePrismaClient(row=None, spend_logs_total=0.0)

    async def _broken_group_by(**kwargs):
        raise RuntimeError("db down")

    prisma.db.litellm_spendlogs.group_by = _broken_group_by
    redis = _FakeRedisCache()
    cache = DualCache(redis_cache=redis)

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result is None
    assert redis.store == {}


@pytest.mark.asyncio
async def test_coalesced_window_keeps_row_when_aggregate_read_fails():
    """The logs-floor reseed must still succeed when the aggregate read fails:
    the maintained row alone is the fallback (better than returning None and
    forcing every read onto the conservative fallback)."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=0.0)

    async def _broken_group_by(**kwargs):
        raise RuntimeError("db down")

    prisma.db.litellm_spendlogs.group_by = _broken_group_by
    redis = _FakeRedisCache()
    cache = DualCache(redis_cache=redis)

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
        include_logs_floor=True,
    )

    assert result == 4.5
    assert redis.store["spend:key:tok-1:window:30d"] == 4.5


@pytest.mark.asyncio
async def test_coalesced_window_seeds_via_increment_when_nx_lost_and_value_vanished():
    """Codecov line: NX lost the race but the re-read finds nothing (the
    winner's key expired between the two calls). The increment path is the
    last resort to still seed a counter instead of returning an
    un-coordinated snapshot."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=4.5)
    redis = _FakeRedisCache(nx_lost_vanished=True)
    cache = DualCache(redis_cache=redis)

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key="spend:key:tok-1:window:30d",
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert redis.store["spend:key:tok-1:window:30d"] == 4.5
    assert cache.in_memory_cache.get_cache(key="spend:key:tok-1:window:30d") == 4.5

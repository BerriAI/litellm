"""Direct behavior pins for ``SpendCounterReseed``.

This module sits at the center of two release-boundary fixes:
- #26459 reseed the enforcement read path from the DB on a counter miss
  (``from_db`` / ``coalesced``)
- #27854 seed the Redis counter via SET NX to stop cross-pod double-seed,
  the fix for phantom ``BudgetExceededError`` where the counter reported an
  integer multiple (N x) of real Postgres spend

It is exercised indirectly through ``proxy_server`` elsewhere, but the unit
contract of each branch was unpinned. These tests use a stateful shared-Redis
fake (faithful SET NX + INCRBYFLOAT semantics) and a fake prisma client so the
counter VALUE is observed, not just call flags.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from litellm.caching.caching import DualCache, InMemoryCache
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed


@pytest.fixture(autouse=True)
def _reset_reseed_locks():
    """Reseed singleflight locks are asyncio objects bound to an event loop.

    pytest-asyncio spins a fresh loop per test, so a lock carried over would be
    attached to a dead loop. Clear the registry on each side of the test.
    """
    SpendCounterReseed._locks.clear()
    SpendCounterReseed._registry_lock = None
    yield
    SpendCounterReseed._locks.clear()
    SpendCounterReseed._registry_lock = None


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _SharedRedisFake:
    """Shared Redis stand-in honoring SET NX and INCRBYFLOAT semantics."""

    def __init__(self):
        self.store: dict[str, float] = {}
        self.set_nx_seed_count = 0
        self.increment_count = 0

    async def async_get_cache(self, key, **kwargs):
        return self.store.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        if kwargs.get("nx", False) and key in self.store:
            return None
        self.store[key] = float(value)
        self.set_nx_seed_count += 1
        return True

    async def async_increment(self, key, value, **kwargs):
        self.increment_count += 1
        self.store[key] = float(self.store.get(key, 0.0)) + float(value)
        return self.store[key]

    async def async_delete_cache(self, key, **kwargs):
        self.store.pop(key, None)


class _DBRow:
    def __init__(self, spend):
        self.spend = spend


class _Table:
    def __init__(self, row="__missing__", raises=None):
        self._row = row
        self._raises = raises

    async def find_unique(self, where):
        if self._raises is not None:
            raise self._raises
        return self._row if self._row != "__missing__" else None


class _SpendLogsTable:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises

    async def group_by(self, by, where, sum):
        if self._raises is not None:
            raise self._raises
        return self._response


class _FakeDB:
    pass


class _FakePrisma:
    def __init__(self):
        self.db = _FakeDB()


def _prisma_with_row(table_attr, spend=None, row="__missing__", raises=None):
    prisma = _FakePrisma()
    if spend is not None:
        row = _DBRow(spend)
    setattr(prisma.db, table_attr, _Table(row=row, raises=raises))
    return prisma


def _make_dual_cache(redis_cache):
    return DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)


# ---------------------------------------------------------------------------
# from_db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_db_none_prisma_returns_none():
    assert await SpendCounterReseed.from_db(None, "spend:key:tok") is None


@pytest.mark.parametrize(
    "counter_key, table_attr",
    [
        ("spend:key:tok-1", "litellm_verificationtoken"),
        ("spend:team:team-1", "litellm_teamtable"),
        ("spend:user:user-1", "litellm_usertable"),
        ("spend:org:org-1", "litellm_organizationtable"),
        ("spend:team_member:user-1:team-1", "litellm_teammembership"),
    ],
)
@pytest.mark.asyncio
async def test_from_db_reads_row_spend_per_prefix(counter_key, table_attr):
    prisma = _prisma_with_row(table_attr, spend=42.5)
    assert await SpendCounterReseed.from_db(prisma, counter_key) == 42.5


@pytest.mark.asyncio
async def test_from_db_null_row_spend_coerces_to_zero():
    prisma = _prisma_with_row("litellm_verificationtoken", row=_DBRow(spend=None))
    assert await SpendCounterReseed.from_db(prisma, "spend:key:tok") == 0.0


@pytest.mark.parametrize(
    "counter_key",
    [
        "spend:tag:tenant-1",
        "spend:end_user:eu-1",
        "spend:unknown:whatever",
        "spend:key:tok:window:30d",
        "spend:team:team-1:window:1mo",
        "spend:team_member:no-colon-suffix",
    ],
)
@pytest.mark.asyncio
async def test_from_db_returns_none_for_non_reseedable_keys(counter_key):
    prisma = _prisma_with_row("litellm_verificationtoken", spend=999.0)
    prisma.db.litellm_teamtable = _Table(row=_DBRow(999.0))
    prisma.db.litellm_teammembership = _Table(row=_DBRow(999.0))
    assert await SpendCounterReseed.from_db(prisma, counter_key) is None


@pytest.mark.asyncio
async def test_from_db_missing_row_returns_none():
    prisma = _prisma_with_row("litellm_verificationtoken", row="__missing__")
    assert await SpendCounterReseed.from_db(prisma, "spend:key:tok") is None


@pytest.mark.asyncio
async def test_from_db_query_exception_returns_none_not_raise():
    prisma = _prisma_with_row(
        "litellm_verificationtoken", raises=RuntimeError("db down")
    )
    assert await SpendCounterReseed.from_db(prisma, "spend:key:tok") is None


# ---------------------------------------------------------------------------
# coalesced (SET NX cold-seed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coalesced_returns_cached_value_without_db_read():
    shared = _SharedRedisFake()
    counter_key = "spend:key:warm"
    shared.store[counter_key] = 73.0
    cache = _make_dual_cache(shared)
    prisma = _prisma_with_row("litellm_verificationtoken", spend=1000.0)

    result = await SpendCounterReseed.coalesced(
        prisma_client=prisma, spend_counter_cache=cache, counter_key=counter_key
    )

    assert result == 73.0
    assert shared.set_nx_seed_count == 0
    assert shared.increment_count == 0


@pytest.mark.asyncio
async def test_coalesced_cold_seed_sets_db_spend_via_set_nx():
    shared = _SharedRedisFake()
    counter_key = "spend:key:cold"
    cache = _make_dual_cache(shared)
    prisma = _prisma_with_row("litellm_verificationtoken", spend=1000.0)

    result = await SpendCounterReseed.coalesced(
        prisma_client=prisma, spend_counter_cache=cache, counter_key=counter_key
    )

    assert result == 1000.0
    assert shared.store[counter_key] == 1000.0
    assert shared.set_nx_seed_count == 1
    assert shared.increment_count == 0
    assert cache.in_memory_cache.get_cache(key=counter_key) == 1000.0


@pytest.mark.asyncio
async def test_coalesced_loser_reads_winner_value_never_re_adds_db_spend():
    """A pod that loses the SET NX race must adopt the winner's value, not add
    its own db_spend on top. INCRBYFLOAT here is the N x inflation bug."""
    winner_value = 1900.0
    db_spend = 1000.0

    class _LoserRedis:
        def __init__(self):
            self.get_calls = 0
            self.seed_attempt_value = None

        async def async_get_cache(self, key, **kwargs):
            self.get_calls += 1
            if self.get_calls == 1:
                return None
            return winner_value

        async def async_set_cache(self, key, value, **kwargs):
            self.seed_attempt_value = value
            return None

        async def async_increment(self, key, value, **kwargs):
            raise AssertionError("loser must not INCRBYFLOAT its own db_spend")

        async def async_delete_cache(self, key, **kwargs):
            pass

    loser = _LoserRedis()
    cache = _make_dual_cache(loser)
    prisma = _prisma_with_row("litellm_verificationtoken", spend=db_spend)

    result = await SpendCounterReseed.coalesced(
        prisma_client=prisma, spend_counter_cache=cache, counter_key="spend:key:race"
    )

    assert result == winner_value
    assert loser.seed_attempt_value == db_spend
    assert cache.in_memory_cache.get_cache(key="spend:key:race") == winner_value


@pytest.mark.asyncio
async def test_coalesced_db_unavailable_returns_none():
    shared = _SharedRedisFake()
    cache = _make_dual_cache(shared)

    result = await SpendCounterReseed.coalesced(
        prisma_client=None, spend_counter_cache=cache, counter_key="spend:key:nodb"
    )

    assert result is None
    assert shared.store == {}


@pytest.mark.asyncio
async def test_coalesced_no_redis_seeds_in_memory_via_increment():
    cache = _make_dual_cache(redis_cache=None)
    prisma = _prisma_with_row("litellm_verificationtoken", spend=250.0)
    counter_key = "spend:key:single-pod"

    result = await SpendCounterReseed.coalesced(
        prisma_client=prisma, spend_counter_cache=cache, counter_key=counter_key
    )

    assert result == 250.0
    assert cache.in_memory_cache.get_cache(key=counter_key) == 250.0


@pytest.mark.asyncio
async def test_coalesced_warm_failure_reraises_only_when_required():
    db_spend = 500.0

    class _SeedFailsRedis:
        async def async_get_cache(self, key, **kwargs):
            return None

        async def async_set_cache(self, key, value, **kwargs):
            raise RuntimeError("redis write failed")

        async def async_increment(self, key, value, **kwargs):
            raise RuntimeError("redis write failed")

        async def async_delete_cache(self, key, **kwargs):
            pass

    prisma = _prisma_with_row("litellm_verificationtoken", spend=db_spend)

    swallowed = await SpendCounterReseed.coalesced(
        prisma_client=prisma,
        spend_counter_cache=_make_dual_cache(_SeedFailsRedis()),
        counter_key="spend:key:soft",
        require_cache_warm=False,
    )
    assert swallowed == db_spend

    with pytest.raises(RuntimeError):
        await SpendCounterReseed.coalesced(
            prisma_client=prisma,
            spend_counter_cache=_make_dual_cache(_SeedFailsRedis()),
            counter_key="spend:key:strict",
            require_cache_warm=True,
        )


@pytest.mark.asyncio
async def test_coalesced_multi_call_seeds_once_across_pods():
    """Three cold pods sharing one Redis seed the counter exactly once."""
    db_spend = 1000.0
    shared = _SharedRedisFake()
    prisma = _prisma_with_row("litellm_verificationtoken", spend=db_spend)
    counter_key = "spend:key:multipod"

    for _ in range(3):
        await SpendCounterReseed.coalesced(
            prisma_client=prisma,
            spend_counter_cache=_make_dual_cache(shared),
            counter_key=counter_key,
        )

    assert shared.store[counter_key] == db_spend
    assert shared.set_nx_seed_count == 1


# ---------------------------------------------------------------------------
# window_from_spend_logs / coalesced_window
# ---------------------------------------------------------------------------


def _window_start():
    return datetime(2026, 5, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("entity_type", ["Key", "Team"])
@pytest.mark.asyncio
async def test_window_from_spend_logs_sums_spend(entity_type):
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(response=[{"_sum": {"spend": 123.5}}])

    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=prisma,
        entity_type=entity_type,
        entity_id="entity-1",
        window_start=_window_start(),
    )
    assert result == 123.5


@pytest.mark.asyncio
async def test_window_from_spend_logs_empty_result_is_zero():
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(response=[])

    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="entity-1",
        window_start=_window_start(),
    )
    assert result == 0.0


@pytest.mark.asyncio
async def test_window_from_spend_logs_unknown_entity_returns_none():
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(response=[{"_sum": {"spend": 5.0}}])

    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=prisma,
        entity_type="EndUser",
        entity_id="entity-1",
        window_start=_window_start(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_window_from_spend_logs_none_prisma_returns_none():
    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=None,
        entity_type="Key",
        entity_id="entity-1",
        window_start=_window_start(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_window_from_spend_logs_query_exception_returns_none():
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(raises=RuntimeError("group_by boom"))

    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=prisma,
        entity_type="Team",
        entity_id="entity-1",
        window_start=_window_start(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_coalesced_window_cold_seed_sets_window_spend_via_set_nx():
    shared = _SharedRedisFake()
    counter_key = "spend:key:tok:window:30d"
    cache = _make_dual_cache(shared)
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(response=[{"_sum": {"spend": 50.0}}])

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key=counter_key,
        entity_type="Key",
        entity_id="tok",
        window_start=_window_start(),
    )

    assert result == 50.0
    assert shared.store[counter_key] == 50.0
    assert shared.set_nx_seed_count == 1
    assert shared.increment_count == 0


@pytest.mark.asyncio
async def test_coalesced_window_multi_call_seeds_once_across_pods():
    shared = _SharedRedisFake()
    counter_key = "spend:team:team-1:window:1mo"
    prisma = _FakePrisma()
    prisma.db.litellm_spendlogs = _SpendLogsTable(response=[{"_sum": {"spend": 80.0}}])

    for _ in range(3):
        await SpendCounterReseed.coalesced_window(
            prisma_client=prisma,
            spend_counter_cache=_make_dual_cache(shared),
            counter_key=counter_key,
            entity_type="Team",
            entity_id="team-1",
            window_start=_window_start(),
        )

    assert shared.store[counter_key] == 80.0
    assert shared.set_nx_seed_count == 1

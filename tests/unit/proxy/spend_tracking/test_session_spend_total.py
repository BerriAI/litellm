from typing import Final

import pytest

from litellm.proxy.spend_tracking.session_spend_total import SessionSpendTotals


class _Redis:
    """The two RedisCache calls the session total makes, over a dict with Redis expiry on a test clock."""

    def __init__(self, failing: bool = False) -> None:
        self.failing = failing
        self.now = 0.0
        self.values: dict[str, tuple[float, float]] = {}  # mutable-ok: stands in for Redis state

    def _live(self, key: str) -> tuple[float, float] | None:
        entry: Final = self.values.get(key)
        return entry if entry is not None and entry[1] > self.now else None

    async def async_increment(
        self, key: str, value: float, *, ttl: int | None = None, refresh_ttl: bool = False
    ) -> float:
        if self.failing:
            raise ConnectionError("Redis dropped the write")
        live: Final = self._live(key)
        expires_at: Final = self.now + (ttl or 0) if live is None or refresh_ttl else live[1]
        total: Final = (live[0] if live is not None else 0.0) + value
        self.values[key] = (total, expires_at)
        return total

    async def async_get_cache(self, key: str) -> str | None:
        if self.failing:
            raise ConnectionError("Redis dropped the read")
        live: Final = self._live(key)
        return str(live[0]) if live is not None else None


def _totals(redis: _Redis) -> SessionSpendTotals:
    return SessionSpendTotals(lambda: redis)


@pytest.mark.asyncio
async def test_a_session_total_sums_its_own_keys_rows_only() -> None:
    totals: Final = _totals(_Redis())

    await totals.record("key-a", "session-1", 0.5)
    await totals.record("key-a", "session-1", 0.25)
    await totals.record("key-b", "session-1", 4.0)
    await totals.record("key-a", "session-2", 8.0)

    assert await totals.read("key-a", "session-1") == 0.75
    assert await totals.read("key-b", "session-1") == 4.0
    assert await totals.read("key-a", "session-3") is None


@pytest.mark.asyncio
async def test_a_redis_outage_never_fails_the_spend_write_and_reads_as_no_total() -> None:
    totals: Final = _totals(_Redis(failing=True))

    await totals.record("key", "session", 0.5)

    assert await totals.read("key", "session") is None


@pytest.mark.asyncio
async def test_without_a_shared_redis_there_is_no_total() -> None:
    totals: Final = SessionSpendTotals(lambda: None)

    await totals.record("key", "session", 0.5)

    assert await totals.read("key", "session") is None


@pytest.mark.asyncio
async def test_an_active_session_keeps_its_total_and_an_idle_one_restarts() -> None:
    redis: Final = _Redis()
    totals: Final = _totals(redis)

    await totals.record("key", "session", 0.5)
    redis.now = 3000.0
    await totals.record("key", "session", 0.25)
    redis.now = 6000.0
    active: Final = await totals.read("key", "session")
    redis.now = 6000.0 + 3601.0
    idle: Final = await totals.read("key", "session")
    await totals.record("key", "session", 0.125)

    assert active == 0.75
    assert idle is None
    assert await totals.read("key", "session") == 0.125

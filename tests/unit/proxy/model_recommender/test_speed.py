import asyncio
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.proxy.model_recommender.speed import SpeedCache, observed_speed


class SpeedDatabase:
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return ({"model_group": "example", "latency_seconds": 2.5, "output_tokens_per_second": 30},)


class UnavailableDatabase:
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        raise ConnectionError("unavailable")


@pytest.mark.asyncio
async def test_speed_returns_latency_and_throughput() -> None:
    result: Final = await observed_speed(SpeedDatabase(), ("example",))
    assert tuple((row.model_group, row.latency_seconds, row.output_tokens_per_second) for row in result) == (
        ("example", 2.5, 30),
    )


@pytest.mark.asyncio
async def test_no_database_no_candidates_and_database_error_are_unknown() -> None:
    assert await observed_speed(None, ("example",)) == ()
    assert await observed_speed(UnavailableDatabase(), ()) == ()
    assert await observed_speed(UnavailableDatabase(), ("example",)) == ()


@pytest.mark.asyncio
async def test_speed_cache_coalesces_misses_and_canonicalizes_groups() -> None:
    responses: Final = iter((2.5, 5.0))

    class Database:
        async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
            latency: Final = next(responses)
            await asyncio.sleep(0)
            return ({"model_group": "example", "latency_seconds": latency},)

    database: Final = Database()
    cache: Final = SpeedCache(clock=lambda: 10)
    first, second = await asyncio.gather(
        cache.get(database, ("example", "other")), cache.get(database, ("other", "example"))
    )
    assert first[0].latency_seconds == second[0].latency_seconds == 2.5
    assert (await cache.get(database, ("example", "other"))) == first
    assert (await cache.get(database, ("example",)))[0].latency_seconds == 5.0
    assert await cache.get(database, ("other", "example")) == first
    assert await cache.get(None, ("example",)) == ()


@pytest.mark.asyncio
async def test_speed_cache_expires_and_caches_empty_results() -> None:
    times: Final = iter((0.0, 10.0, 61.0))
    cache: Final = SpeedCache(clock=lambda: next(times))
    assert await cache.get(UnavailableDatabase(), ("example",)) == ()
    assert await cache.get(SpeedDatabase(), ("example",)) == ()
    assert (await cache.get(SpeedDatabase(), ("example",)))[0].latency_seconds == 2.5

from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.proxy.model_recommender.speed import observed_speed


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

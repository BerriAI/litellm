import io
import zipfile
from typing import Final

import pytest

from litellm.proxy.model_recommender.benchmarks import BenchmarkCache, intelligence_score, parse_archive


def archive() -> bytes:
    stream: Final = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr(
            "gpqa_diamond.csv", "Model version,mean_score\nexample_high,0.7\nexample_low,0.4\nexample-other,0.9\n"
        )
        zipped.writestr("terminalbench_external.csv", "Model version,Accuracy mean\nexample_unknown,0.6\nbroken,nan\n")
    return stream.getvalue()


def test_exact_normalized_match_takes_best_effort_and_external_accuracy_column() -> None:
    scores: Final = parse_archive(archive())
    assert intelligence_score(("openai/example",), frozenset({"general"}), scores) == 0.7
    assert intelligence_score(("example",), frozenset({"agentic"}), scores) == 0.6
    assert intelligence_score(("example-preview",), frozenset({"general"}), scores) is None
    assert intelligence_score(("example",), frozenset({"general", "coding"}), scores) is None
    assert intelligence_score(("example",), frozenset(), scores) is None


@pytest.mark.asyncio
async def test_cache_fetches_once_within_day() -> None:
    responses: Final = iter((archive(), b"invalid"))

    async def fetch() -> bytes:
        return next(responses)

    cache: Final = BenchmarkCache(fetcher=fetch, clock=lambda: 10)
    first: Final = await cache.get()
    assert len(first) == 4
    assert await cache.get() == first


@pytest.mark.asyncio
async def test_network_and_invalid_archives_fail_open() -> None:
    async def unavailable() -> bytes:
        raise OSError("offline")

    async def invalid() -> bytes:
        return b"invalid zip"

    assert await BenchmarkCache(fetcher=unavailable).get() == ()
    assert await BenchmarkCache(fetcher=invalid).get() == ()


@pytest.mark.asyncio
async def test_cache_expires_after_day_and_does_not_report_stale_data_on_failure() -> None:
    timestamps: Final = iter((0.0, 0.0, 86401.0, 86401.0))
    responses: Final = iter((archive(), b"invalid"))

    async def fetch() -> bytes:
        return next(responses)

    cache: Final = BenchmarkCache(fetcher=fetch, clock=lambda: next(timestamps))
    assert len(await cache.get()) == 4
    assert await cache.get() == ()
    assert cache.as_of is None

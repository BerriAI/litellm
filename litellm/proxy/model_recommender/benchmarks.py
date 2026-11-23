import asyncio
import csv
import io
import math
import re
import time
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final

import httpx

from .models import Focus, FrozenModel

EPOCH_URL: Final = "https://epoch.ai/data/benchmark_data.zip"
BENCHMARKS: Final[Mapping[Focus, tuple[str, str]]] = MappingProxyType(
    {
        "general": ("gpqa_diamond.csv", "GPQA Diamond"),
        "math": ("otis_mock_aime_2024_2025.csv", "AIME"),
        "coding": ("swe_bench_verified.csv", "SWE-bench Verified"),
        "agentic": ("terminalbench_external.csv", "Terminal-Bench"),
    }
)


class BenchmarkScore(FrozenModel):
    model: str
    focus: Focus
    score: float


def normalized_id(model: str) -> str:
    return re.sub(r"_(?:none|minimal|low|medium|high|xhigh|max|unknown)$", "", model.rsplit("/", 1)[-1].lower())


def read_score(row: Mapping[str, str], focus: Focus) -> BenchmarkScore | None:
    try:
        score: Final = float(row.get("mean_score", row.get("Accuracy mean", "")))
        model: Final = row.get("Model version", "")
        if not model or not math.isfinite(score) or not 0 <= score <= 1:
            return None
        return BenchmarkScore(model=normalized_id(model), focus=focus, score=score)
    except (TypeError, ValueError):
        return None


def read_csv(archive: zipfile.ZipFile, focus: Focus) -> tuple[BenchmarkScore, ...]:
    filename: Final = BENCHMARKS[focus][0]
    if filename not in archive.namelist():
        return ()
    member: Final = archive.getinfo(filename)
    if member.file_size > 10_000_000:
        return ()
    rows: Final = csv.DictReader(io.StringIO(archive.read(filename).decode("utf-8-sig")))
    return tuple(score for row in rows if (score := read_score(row, focus)) is not None)


def parse_archive(content: bytes) -> tuple[BenchmarkScore, ...]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return tuple(chain.from_iterable(read_csv(archive, focus) for focus in BENCHMARKS))


async def fetch_epoch(client: httpx.AsyncClient | None = None, max_bytes: int = 50_000_000) -> bytes:
    if client is None:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as owned_client:
            return await fetch_epoch(owned_client, max_bytes)
    async with client.stream("GET", EPOCH_URL) as response:
        response.raise_for_status()
        with io.BytesIO() as content:
            async for chunk in response.aiter_bytes():
                if content.tell() + len(chunk) > max_bytes:
                    return b""
                content.write(chunk)
            return content.getvalue()


class BenchmarkCache:
    def __init__(
        self,
        fetcher: Callable[[], Awaitable[bytes]] = fetch_epoch,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._clock = clock
        self._expires = 0.0
        self._scores: tuple[BenchmarkScore, ...] = ()
        self._as_of: str | None = None
        self._lock = asyncio.Lock()

    @property
    def as_of(self) -> str | None:
        return self._as_of

    async def get(self) -> tuple[BenchmarkScore, ...]:
        async with self._lock:
            if self._clock() < self._expires:
                return self._scores
            try:
                content: Final = await self._fetcher()
                scores: Final = await asyncio.to_thread(parse_archive, content)
                self._scores = scores
                self._as_of = datetime.now(timezone.utc).date().isoformat() if scores else None
                self._expires = self._clock() + 86400
            except Exception:  # noqa: BLE001  # Optional benchmark failures must not block recommendations.
                self._scores = ()
                self._as_of = None
                self._expires = self._clock() + 300
            return self._scores


def intelligence_score(
    models: tuple[str, ...], focus: frozenset[Focus], scores: tuple[BenchmarkScore, ...]
) -> float | None:
    ids: Final = frozenset(normalized_id(model) for model in models)
    best: Final = tuple(best_for_focus(ids, area, scores) for area in focus)
    if not best or any(score is None for score in best):
        return None
    return sum(score for score in best if score is not None) / len(best)


def best_for_focus(ids: frozenset[str], focus: Focus, scores: tuple[BenchmarkScore, ...]) -> float | None:
    return max((row.score for row in scores if row.focus == focus and row.model in ids), default=None)

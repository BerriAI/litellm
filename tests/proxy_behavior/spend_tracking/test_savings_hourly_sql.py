"""
The hourly savings query buckets on the caller's clock via an IANA timezone, so
Postgres resolves the UTC offset that was in effect at each request's own
instant. These tests run the real query against a live Postgres and prove the
offset tracks the selected date (DST), not today, and that boundary traffic
lands in the correct local day. A fixed offset, the bug this replaces, would
fail every one of them on the wrong side of a DST transition.
"""

import os

import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.proxy.spend_tracking.savings_endpoints import (
    _HOURLY_SAVINGS_ROWS,
    _HOURLY_SAVINGS_QUERY,
    build_hourly_buckets,
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="hourly savings SQL needs a live Postgres",
    ),
]

TZ = "America/Los_Angeles"
REQUEST_PREFIX = "savings-hourly-sql-test-"


@pytest_asyncio.fixture(scope="session")
async def db():
    client = Prisma()
    await client.connect()
    yield client
    await client.disconnect()


@pytest_asyncio.fixture
async def insert_cache_read(db):
    inserted: list[str] = []

    async def _insert(request_id_suffix: str, start_time_utc: str, cache_read_tokens: int) -> None:
        request_id = f"{REQUEST_PREFIX}{request_id_suffix}"
        inserted.append(request_id)
        await db.execute_raw(
            """
            INSERT INTO "LiteLLM_SpendLogs"
                (request_id, call_type, "startTime", "endTime", model, custom_llm_provider, metadata)
            VALUES ($1, 'acompletion', $2::timestamp, $2::timestamp, 'claude-sonnet-5', 'anthropic', $3::jsonb)
            """,
            request_id,
            start_time_utc,
            {"usage_object": {"cache_read_input_tokens": cache_read_tokens}},
        )

    yield _insert

    if inserted:
        await db.execute_raw(
            'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1::text[])',
            inserted,
        )


async def _buckets(db, start_date: str, end_date_exclusive: str):
    rows = await db.query_raw(_HOURLY_SAVINGS_QUERY, f"{start_date}T00:00:00", f"{end_date_exclusive}T00:00:00", TZ)
    return build_hourly_buckets(_HOURLY_SAVINGS_ROWS.validate_python(rows or []))


async def _caching_by_hour(db, start_date: str, end_date_exclusive: str) -> dict[str, float]:
    return {b.bucket_start: b.prompt_caching_savings_spend for b in await _buckets(db, start_date, end_date_exclusive)}


def _hours_that_grew(before: dict[str, float], after: dict[str, float]) -> list[str]:
    # Isolate the row this test added from whatever ambient traffic the shared
    # dev DB already holds on these dates.
    return sorted(h for h, value in after.items() if value - before.get(h, 0.0) > 1e-12)


async def test_offset_tracks_the_selected_date_not_today(db, insert_cache_read):
    # Same UTC wall-clock time, one instant in PDT summer, one in PST winter.
    # A fixed offset would file both under the same local hour; the correct
    # per-date offset files them one hour apart.
    before_summer = await _caching_by_hour(db, "2026-07-15", "2026-07-16")
    before_winter = await _caching_by_hour(db, "2026-01-15", "2026-01-16")
    await insert_cache_read("summer", "2026-07-15T20:30:00", 9624)
    await insert_cache_read("winter", "2026-01-15T20:30:00", 9624)
    after_summer = await _caching_by_hour(db, "2026-07-15", "2026-07-16")
    after_winter = await _caching_by_hour(db, "2026-01-15", "2026-01-16")

    assert _hours_that_grew(before_summer, after_summer) == ["2026-07-15T13:00"]  # PDT, UTC-7
    assert _hours_that_grew(before_winter, after_winter) == ["2026-01-15T12:00"]  # PST, UTC-8


async def test_boundary_traffic_lands_in_the_local_day_not_the_utc_day(db, insert_cache_read):
    # 2026-07-16 05:30 UTC is still 2026-07-15 22:30 in PDT.
    before_local = await _caching_by_hour(db, "2026-07-15", "2026-07-16")
    before_utc = await _caching_by_hour(db, "2026-07-16", "2026-07-17")
    await insert_cache_read("boundary", "2026-07-16T05:30:00", 9624)
    after_local = await _caching_by_hour(db, "2026-07-15", "2026-07-16")
    after_utc = await _caching_by_hour(db, "2026-07-16", "2026-07-17")

    assert _hours_that_grew(before_local, after_local) == ["2026-07-15T22:00"]
    assert _hours_that_grew(before_utc, after_utc) == []


async def test_a_plain_day_is_twenty_four_hours(db):
    buckets = await _buckets(db, "2026-07-15", "2026-07-16")
    assert [b.bucket_start for b in buckets] == [f"2026-07-15T{h:02d}:00" for h in range(24)]


async def test_spring_forward_day_is_twenty_three_hours(db):
    # 2026-03-08: clocks jump 02:00 -> 03:00, so 02:00 never occurs locally.
    buckets = await _buckets(db, "2026-03-08", "2026-03-09")
    labels = [b.bucket_start for b in buckets]
    assert len(labels) == 23
    assert "2026-03-08T02:00" not in labels
    assert "2026-03-08T03:00" in labels


async def test_fall_back_repeated_hour_merges_into_one_wall_clock_bucket(db, insert_cache_read):
    # 2026-11-01: clocks fall 02:00 -> 01:00, so the local 01:00 hour happens
    # twice. By design the chart buckets on the wall clock, so both elapsed
    # hours share the single "01:00" label and their savings sum into it. The
    # day therefore has 24 labels, not 25, and no label repeats.
    buckets = await _buckets(db, "2026-11-01", "2026-11-02")
    labels = [b.bucket_start for b in buckets]
    assert len(labels) == 24
    assert len(set(labels)) == 24

    # Traffic in each of the two distinct 01:00 hours lands on the same bucket.
    # 08:30 UTC is 01:30 PDT (first pass); 09:30 UTC is 01:30 PST (second pass).
    before = await _caching_by_hour(db, "2026-11-01", "2026-11-02")
    await insert_cache_read("fallback-pdt", "2026-11-01T08:30:00", 5000)
    await insert_cache_read("fallback-pst", "2026-11-01T09:30:00", 4000)
    after = await _caching_by_hour(db, "2026-11-01", "2026-11-02")

    assert _hours_that_grew(before, after) == ["2026-11-01T01:00"]
    assert after["2026-11-01T01:00"] - before.get("2026-11-01T01:00", 0.0) > 0

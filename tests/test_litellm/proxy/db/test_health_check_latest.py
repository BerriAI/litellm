from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.health_check_latest import (
    LATEST_HEALTH_CHECKS_FOR_MODELS_SQL,
    LATEST_HEALTH_CHECKS_SQL,
    fetch_latest_health_checks,
    fetch_latest_health_checks_for_models,
)


def _prisma(rows):
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=rows)
    return prisma


def _raw_row(**overrides):
    row = {
        "health_check_id": "hc-1",
        "model_name": "gpt-4",
        "model_id": "deployment-abc",
        "status": "healthy",
        "healthy_count": 1,
        "unhealthy_count": 0,
        "error_message": None,
        "response_time_ms": 12.5,
        "details": None,
        "checked_by": "pod-1",
        "checked_at": "2026-08-25T00:00:00+00:00",
        "created_at": "2026-08-25T00:00:00+00:00",
        "updated_at": "2026-08-25T00:00:00+00:00",
    }
    return {**row, **overrides}


@pytest.mark.asyncio
async def test_fetch_all_runs_one_distinct_on_query_with_no_parameters():
    """The dedup must be in the SQL: prisma find_many(distinct=...) streams the whole history table."""
    prisma = _prisma([])
    assert await fetch_latest_health_checks(prisma) == ()
    assert prisma.db.query_raw.await_args.args == (LATEST_HEALTH_CHECKS_SQL,)
    assert 'DISTINCT ON ("model_id", "model_name")' in LATEST_HEALTH_CHECKS_SQL
    assert '"checked_at" DESC' in LATEST_HEALTH_CHECKS_SQL


@pytest.mark.asyncio
async def test_raw_datetimes_come_back_tz_aware_with_or_without_an_offset():
    """The save path subtracts checked_at from datetime.now(timezone.utc); a naive value would TypeError."""
    naive = _raw_row(health_check_id="hc-naive", model_id=None, checked_at="2026-08-25T00:00:00")
    aware = _raw_row(health_check_id="hc-aware", checked_at="2026-08-25T01:00:00+02:00")
    rows = await fetch_latest_health_checks(_prisma([naive, aware]))
    assert {row.health_check_id: (row.model_id, row.checked_at) for row in rows} == {
        "hc-naive": (None, datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc)),
        "hc-aware": ("deployment-abc", datetime(2026, 8, 24, 23, 0, tzinfo=timezone.utc)),
    }


@pytest.mark.asyncio
async def test_json_details_decode_from_text_and_pass_through_as_dict():
    rows = await fetch_latest_health_checks(
        _prisma(
            [
                _raw_row(health_check_id="text", details='{"region": "eu"}'),
                _raw_row(health_check_id="dict", details={"region": "us"}),
                _raw_row(health_check_id="none", details=None),
            ]
        )
    )
    assert {row.health_check_id: row.details for row in rows} == {
        "text": {"region": "eu"},
        "dict": {"region": "us"},
        "none": None,
    }


@pytest.mark.asyncio
async def test_fetch_all_degrades_to_no_rows_when_the_query_fails():
    prisma = _prisma([])
    prisma.db.query_raw.side_effect = RuntimeError("db down")
    assert await fetch_latest_health_checks(prisma) == ()


@pytest.mark.asyncio
async def test_fetch_all_degrades_to_no_rows_for_a_malformed_row():
    assert await fetch_latest_health_checks(_prisma([{"unexpected": "shape"}])) == ()


@pytest.mark.asyncio
async def test_fetch_for_models_binds_the_page_as_the_only_parameter():
    prisma = _prisma([_raw_row()])
    rows = await fetch_latest_health_checks_for_models(prisma, ("gpt-4", "claude-opus"))
    assert prisma.db.query_raw.await_args.args == (LATEST_HEALTH_CHECKS_FOR_MODELS_SQL, ["gpt-4", "claude-opus"])
    assert [row.model_name for row in rows] == ["gpt-4"]
    assert 'WHERE "model_name" = ANY($1)' in LATEST_HEALTH_CHECKS_FOR_MODELS_SQL


@pytest.mark.asyncio
async def test_fetch_for_models_skips_the_database_for_an_empty_page():
    prisma = _prisma([])
    assert await fetch_latest_health_checks_for_models(prisma, ()) == ()
    prisma.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_for_models_degrades_to_no_rows_when_the_query_fails():
    prisma = _prisma([])
    prisma.db.query_raw.side_effect = RuntimeError("db down")
    assert await fetch_latest_health_checks_for_models(prisma, ("gpt-4",)) == ()

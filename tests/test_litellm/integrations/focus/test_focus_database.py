"""Tests for FocusLiteLLMDatabase query construction."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.focus.database import FocusLiteLLMDatabase


def _setup_db(monkeypatch: pytest.MonkeyPatch, query_return):
    """Create a database instance with a stubbed prisma client."""
    query_mock = AsyncMock(return_value=query_return)
    mock_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_mock))
    db = FocusLiteLLMDatabase()
    monkeypatch.setattr(db, "_ensure_prisma_client", lambda: mock_client)
    return db, query_mock


@pytest.mark.asyncio
async def test_should_parameterize_filters_and_limit(monkeypatch: pytest.MonkeyPatch):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data(limit=25, start_time_utc=start, end_time_utc=end)

    query_text, *params = query_mock.await_args.args
    assert "dus.updated_at >= $1::timestamptz" in query_text
    assert "dus.updated_at <= $2::timestamptz" in query_text
    assert "LIMIT $3" in query_text
    assert params == [start, end, 25]


@pytest.mark.asyncio
async def test_should_execute_without_filters(monkeypatch: pytest.MonkeyPatch):
    row = {
        "id": 1,
        "user_id": "user",
        "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    db, query_mock = _setup_db(monkeypatch, [row])

    result = await db.get_usage_data()

    query_text, *params = query_mock.await_args.args
    assert "WHERE" not in query_text
    assert "LIMIT $" not in query_text
    assert params == []
    assert result.height == 1
    assert result["id"][0] == 1


@pytest.mark.asyncio
async def test_should_accept_string_timestamps(monkeypatch: pytest.MonkeyPatch):
    db, query_mock = _setup_db(monkeypatch, [])

    start = "2024-02-01T00:00:00+00:00"
    end = "2024-02-02T00:00:00+00:00"
    await db.get_usage_data(start_time_utc=start, end_time_utc=end)

    _, *params = query_mock.await_args.args
    assert params == [start, end]


@pytest.mark.asyncio
async def test_should_reject_invalid_limit(monkeypatch: pytest.MonkeyPatch):
    db, query_mock = _setup_db(monkeypatch, [])

    with pytest.raises(ValueError):
        await db.get_usage_data(limit="invalid")

    assert query_mock.await_count == 0

"""Tests for LiteLLM CloudZero database helper."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.cloudzero.database import LiteLLMDatabase


def _setup_db(monkeypatch: pytest.MonkeyPatch, query_return):
    """Return a database instance with prisma client mocked out."""
    query_mock = AsyncMock(return_value=query_return)
    mock_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_mock))
    db = LiteLLMDatabase()
    monkeypatch.setattr(db, "_ensure_prisma_client", lambda: mock_client)
    return db, query_mock


@pytest.mark.asyncio
async def test_get_usage_data_parameterized(monkeypatch: pytest.MonkeyPatch):
    """Start/end filters and limit should be parameterized via placeholders."""
    start = datetime(2024, 5, 1, tzinfo=timezone.utc)
    end = datetime(2024, 5, 2, tzinfo=timezone.utc)
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data(limit=10, start_time_utc=start, end_time_utc=end)

    query_text, *params = query_mock.await_args.args
    assert "dus.updated_at >= $1::timestamptz" in query_text
    assert "dus.updated_at <= $2::timestamptz" in query_text
    assert "LIMIT $3" in query_text
    assert params == [start, end, 10]


@pytest.mark.asyncio
async def test_get_usage_data_handles_missing_filters(monkeypatch: pytest.MonkeyPatch):
    """When no filters provided the params should be None placeholders."""
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *params = query_mock.await_args.args
    assert "LIMIT $3" not in query_text
    assert params == [None, None]


@pytest.mark.asyncio
async def test_get_usage_data_rejects_invalid_limit(monkeypatch: pytest.MonkeyPatch):
    """limit must coerce to int or raise ValueError before hitting the DB."""
    db, query_mock = _setup_db(monkeypatch, [])

    with pytest.raises(ValueError):
        await db.get_usage_data(limit="invalid")

    assert query_mock.await_count == 0

"""Tests for FocusSpendLogsDatabase query construction."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.focus.spend_logs_database import FocusSpendLogsDatabase


def _setup_db(monkeypatch: pytest.MonkeyPatch, query_return):
    """Create a database instance with a stubbed prisma client."""
    query_mock = AsyncMock(return_value=query_return)
    mock_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_mock))
    db = FocusSpendLogsDatabase()
    monkeypatch.setattr(db, "_ensure_prisma_client", lambda: mock_client)
    return db, query_mock


@pytest.mark.asyncio
async def test_should_execute_without_filters(monkeypatch):
    row = {
        "request_id": "req-123",
        "api_key": "sk-key",
        "spend": 0.001,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "model": "gpt-4o",
        "model_group": "openai",
        "custom_llm_provider": "openai",
        "team_id": "team-1",
        "user_id": "user-1",
        "date": "2024-03-15",
        "api_key_alias": None,
        "team_alias": None,
        "user_email": None,
    }
    db, query_mock = _setup_db(monkeypatch, [row])

    result = await db.get_usage_data()

    query_text, *params = query_mock.await_args.args
    assert "WHERE" not in query_text
    assert "LIMIT" not in query_text
    assert params == []
    assert result.height == 1
    assert result["request_id"][0] == "req-123"


@pytest.mark.asyncio
async def test_should_parameterize_start_time_filter(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    await db.get_usage_data(start_time_utc=start)

    query_text, *params = query_mock.await_args.args
    assert '"startTime" >= $1::timestamptz' in query_text
    assert params[0] == start


@pytest.mark.asyncio
async def test_should_parameterize_end_time_filter(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    await db.get_usage_data(end_time_utc=end)

    query_text, *params = query_mock.await_args.args
    assert '"startTime" <= $1::timestamptz' in query_text
    assert params[0] == end


@pytest.mark.asyncio
async def test_should_parameterize_both_time_filters(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    await db.get_usage_data(start_time_utc=start, end_time_utc=end)

    query_text, *params = query_mock.await_args.args
    assert "$1::timestamptz" in query_text
    assert "$2::timestamptz" in query_text
    assert params == [start, end]


@pytest.mark.asyncio
async def test_should_parameterize_limit(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data(limit=100)

    query_text, *params = query_mock.await_args.args
    assert "LIMIT $1" in query_text
    assert params == [100]


@pytest.mark.asyncio
async def test_should_parameterize_limit_after_time_filters(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    await db.get_usage_data(start_time_utc=start, end_time_utc=end, limit=50)

    query_text, *params = query_mock.await_args.args
    assert "LIMIT $3" in query_text
    assert params == [start, end, 50]


@pytest.mark.asyncio
async def test_should_reject_invalid_limit(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    with pytest.raises(ValueError):
        await db.get_usage_data(limit="bad")

    assert query_mock.await_count == 0


@pytest.mark.asyncio
async def test_should_reject_negative_limit(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    with pytest.raises(ValueError):
        await db.get_usage_data(limit=-1)

    assert query_mock.await_count == 0


@pytest.mark.asyncio
async def test_query_extracts_cache_creation_tokens_from_metadata_json(monkeypatch):
    """The SQL must extract cache_creation_input_tokens from metadata JSON."""
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "cache_creation_input_tokens" in query_text
    assert "additional_usage_values" in query_text


@pytest.mark.asyncio
async def test_query_extracts_cache_read_tokens_from_metadata_json(monkeypatch):
    """The SQL must extract cache_read_input_tokens from metadata JSON."""
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "cache_read_input_tokens" in query_text
    assert "additional_usage_values" in query_text


@pytest.mark.asyncio
async def test_query_selects_from_spend_logs_table(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "LiteLLM_SpendLogs" in query_text


@pytest.mark.asyncio
async def test_query_includes_request_id(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "request_id" in query_text


@pytest.mark.asyncio
async def test_query_joins_verification_token_for_api_key_alias(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "LiteLLM_VerificationToken" in query_text
    assert "api_key_alias" in query_text


@pytest.mark.asyncio
async def test_query_joins_team_table_for_team_alias(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "LiteLLM_TeamTable" in query_text
    assert "team_alias" in query_text


@pytest.mark.asyncio
async def test_query_joins_user_table_for_email(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert "LiteLLM_UserTable" in query_text
    assert "user_email" in query_text


@pytest.mark.asyncio
async def test_returns_dataframe_with_correct_columns(monkeypatch):
    row = {
        "request_id": "req-abc",
        "api_key": "sk-test",
        "spend": 0.005,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 10,
        "model": "claude-3-5-sonnet",
        "model_group": "anthropic",
        "custom_llm_provider": "anthropic",
        "team_id": "team-abc",
        "user_id": "user-abc",
        "date": "2024-06-01",
        "api_key_alias": "prod",
        "team_alias": "my-team",
        "user_email": "user@example.com",
    }
    db, _ = _setup_db(monkeypatch, [row])

    result = await db.get_usage_data()

    assert result.height == 1
    expected_cols = {
        "request_id", "api_key", "spend", "prompt_tokens", "completion_tokens",
        "total_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        "model", "model_group", "custom_llm_provider", "team_id", "user_id",
        "date", "api_key_alias", "team_alias", "user_email",
    }
    assert expected_cols.issubset(set(result.columns))
    assert result["request_id"][0] == "req-abc"
    assert result["cache_creation_input_tokens"][0] == 20
    assert result["cache_read_input_tokens"][0] == 10


@pytest.mark.asyncio
async def test_returns_empty_dataframe_when_no_rows(monkeypatch):
    db, _ = _setup_db(monkeypatch, [])

    result = await db.get_usage_data()

    assert result.is_empty()

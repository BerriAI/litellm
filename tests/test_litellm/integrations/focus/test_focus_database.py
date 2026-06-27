"""Tests for FocusLiteLLMDatabase query construction."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.transformer import FocusTransformer


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
    assert "dus.updated_at >=" not in query_text
    assert "dus.updated_at <=" not in query_text
    assert "LIMIT $" not in query_text
    # no stray WHERE clause anywhere when no window is given; the only legitimate
    # WHERE is the ARRAY_AGG FILTER, so strip that before asserting
    assert "WHERE" not in query_text.replace("FILTER (WHERE", "")
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


@pytest.mark.asyncio
async def test_should_join_organization_table(monkeypatch: pytest.MonkeyPatch):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert (
        "COALESCE(vt.organization_id, tt.organization_id) as organization_id"
        in query_text
    )
    assert "ot.organization_alias as organization_alias" in query_text
    assert 'LEFT JOIN "LiteLLM_OrganizationTable" ot' in query_text


@pytest.mark.asyncio
async def test_should_join_spend_logs_for_per_user_request_tags(
    monkeypatch: pytest.MonkeyPatch,
):
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data()

    query_text, *_ = query_mock.await_args.args
    assert 'FROM "LiteLLM_SpendLogs" sl' in query_text
    assert "jsonb_array_elements_text" in query_text
    assert "tag_rollup.request_tags as request_tags" in query_text
    # exact per-user attribution: tags join on user_id so a shared api_key
    # never leaks one user's tags onto another user's export row
    assert "COALESCE(tag_rollup.user_id, '') = COALESCE(dus.user_id, '')" in query_text


@pytest.mark.asyncio
async def test_should_scope_tag_subquery_to_time_window(
    monkeypatch: pytest.MonkeyPatch,
):
    """The SpendLogs tag aggregation must be bounded to the requested time
    window (using the startTime index) instead of scanning all history."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    db, query_mock = _setup_db(monkeypatch, [])

    await db.get_usage_data(start_time_utc=start, end_time_utc=end)

    query_text, *params = query_mock.await_args.args
    # window filter converts the bound param to the column's naive-UTC frame so
    # it is independent of the Postgres session TimeZone
    assert "sl.\"startTime\" >= ($1::timestamptz AT TIME ZONE 'UTC')" in query_text
    assert "sl.\"startTime\" <= ($2::timestamptz AT TIME ZONE 'UTC')" in query_text
    # bucketing must NOT apply AT TIME ZONE to the column (startTime is already
    # naive-UTC; rendering it in the session TZ would shift the date and break
    # the join with dus.date on non-UTC sessions)
    assert "to_char(sl.\"startTime\", 'YYYY-MM-DD')" in query_text
    assert "AT TIME ZONE 'UTC', 'YYYY-MM-DD'" not in query_text
    assert params == [start, end]


@pytest.mark.asyncio
async def test_request_tags_flow_from_db_array_into_focus_tags(
    monkeypatch: pytest.MonkeyPatch,
):
    """Postgres returns the aggregated tags as an array; it must survive the
    DataFrame round-trip and land inside the transformed FOCUS Tags column."""
    row = {
        "date": "2024-01-01",
        "user_id": "user",
        "api_key": "sk-test",
        "api_key_alias": "prod-key",
        "model": "gpt-4o",
        "model_group": "gpt-4o",
        "custom_llm_provider": "openai",
        "spend": 0.05,
        "api_requests": 1,
        "team_id": "team-1",
        "team_alias": "Platform",
        "request_tags": ["prod", "checkout"],
    }
    db, _ = _setup_db(monkeypatch, [row])

    frame = await db.get_usage_data()
    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert json.loads(tags["request_tags"]) == ["prod", "checkout"]

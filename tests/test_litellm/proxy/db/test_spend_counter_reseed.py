"""
Regression tests for SpendCounterReseed.window_from_spend_logs.

Covers the bug where entity_type="User" was not handled, causing
user budget windows to never seed from the DB.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed


def _mock_prisma_client():
    return MagicMock()


@pytest.mark.asyncio
async def test_window_from_spend_logs_user_entity_queries_by_user_field():
    prisma = _mock_prisma_client()
    window_start = datetime(2026, 7, 1, 0, 0, 0)
    fake_response = [{"user": "user-abc", "_sum": {"spend": 42.5}}]

    with patch(
        "litellm.proxy.db.spend_counter_reseed.SpendLogsRepository"
    ) as MockRepo:
        mock_table = MagicMock()
        mock_table.group_by = AsyncMock(return_value=fake_response)
        MockRepo.return_value.table = mock_table

        result = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma,
            entity_type="User",
            entity_id="user-abc",
            window_start=window_start,
        )

    assert result == 42.5
    mock_table.group_by.assert_called_once()
    call_kwargs = mock_table.group_by.call_args
    assert call_kwargs.kwargs["by"] == ["user"]
    assert call_kwargs.kwargs["where"]["user"] == "user-abc"
    assert call_kwargs.kwargs["where"]["startTime"] == {"gte": window_start}


@pytest.mark.asyncio
async def test_window_from_spend_logs_user_empty_response_returns_zero():
    prisma = _mock_prisma_client()

    with patch(
        "litellm.proxy.db.spend_counter_reseed.SpendLogsRepository"
    ) as MockRepo:
        mock_table = MagicMock()
        mock_table.group_by = AsyncMock(return_value=[])
        MockRepo.return_value.table = mock_table

        result = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma,
            entity_type="User",
            entity_id="user-xyz",
            window_start=datetime(2026, 7, 1),
        )

    assert result == 0.0


@pytest.mark.asyncio
async def test_window_from_spend_logs_unknown_entity_returns_none():
    prisma = _mock_prisma_client()

    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=prisma,
        entity_type="Organization",
        entity_id="org-1",
        window_start=datetime(2026, 7, 1),
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_spend_logs_none_prisma_returns_none():
    result = await SpendCounterReseed.window_from_spend_logs(
        prisma_client=None,
        entity_type="User",
        entity_id="user-abc",
        window_start=datetime(2026, 7, 1),
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_spend_logs_key_entity_queries_by_api_key():
    prisma = _mock_prisma_client()
    fake_response = [{"api_key": "sk-test", "_sum": {"spend": 10.0}}]

    with patch(
        "litellm.proxy.db.spend_counter_reseed.SpendLogsRepository"
    ) as MockRepo:
        mock_table = MagicMock()
        mock_table.group_by = AsyncMock(return_value=fake_response)
        MockRepo.return_value.table = mock_table

        result = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma,
            entity_type="Key",
            entity_id="sk-test",
            window_start=datetime(2026, 7, 1),
        )

    assert result == 10.0
    call_kwargs = mock_table.group_by.call_args
    assert call_kwargs.kwargs["by"] == ["api_key"]
    assert call_kwargs.kwargs["where"]["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_window_from_spend_logs_team_entity_queries_by_team_id():
    prisma = _mock_prisma_client()
    fake_response = [{"team_id": "team-1", "_sum": {"spend": 25.0}}]

    with patch(
        "litellm.proxy.db.spend_counter_reseed.SpendLogsRepository"
    ) as MockRepo:
        mock_table = MagicMock()
        mock_table.group_by = AsyncMock(return_value=fake_response)
        MockRepo.return_value.table = mock_table

        result = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma,
            entity_type="Team",
            entity_id="team-1",
            window_start=datetime(2026, 7, 1),
        )

    assert result == 25.0
    call_kwargs = mock_table.group_by.call_args
    assert call_kwargs.kwargs["by"] == ["team_id"]
    assert call_kwargs.kwargs["where"]["team_id"] == "team-1"


@pytest.mark.asyncio
async def test_window_from_spend_logs_db_exception_returns_none():
    prisma = _mock_prisma_client()

    with patch(
        "litellm.proxy.db.spend_counter_reseed.SpendLogsRepository"
    ) as MockRepo:
        mock_table = MagicMock()
        mock_table.group_by = AsyncMock(side_effect=RuntimeError("db down"))
        MockRepo.return_value.table = mock_table

        result = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma,
            entity_type="User",
            entity_id="user-abc",
            window_start=datetime(2026, 7, 1),
        )

    assert result is None

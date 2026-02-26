"""
Tests that spend log queries include `take` limits to prevent unbounded result sets
that can cause Prisma query engine OOM (28GB+ RSS observed in production).

See: https://github.com/prisma/prisma/issues/21471 (JSONB RSS retention)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_spend_log_find_many_with_key_val_has_take_limit():
    """
    When get_data is called for spend table with key_val and find_all,
    the query should include a take limit.
    """
    from litellm.proxy.utils import PrismaClient

    mock_prisma = MagicMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma.litellm_spendlogs.find_many = mock_find_many

    prisma_client = PrismaClient.__new__(PrismaClient)
    prisma_client.db = mock_prisma
    prisma_client._spend_log_transactions_lock = asyncio.Lock()
    prisma_client.spend_log_transactions = []
    prisma_client.proxy_logging_obj = MagicMock()

    await prisma_client.get_data(
        table_name="spend",
        key_val={"key": "api_key", "value": "test-key"},
        query_type="find_all",
    )

    mock_find_many.assert_called_once()
    call_kwargs = mock_find_many.call_args[1]
    assert "take" in call_kwargs, "find_many on spend logs must include 'take' to prevent OOM"
    assert call_kwargs["take"] == 1000


@pytest.mark.asyncio
async def test_spend_log_find_many_without_key_val_has_take_limit():
    """
    When get_data is called for spend table without key_val (all logs),
    the query should include a take limit.
    """
    from litellm.proxy.utils import PrismaClient

    mock_prisma = MagicMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma.litellm_spendlogs.find_many = mock_find_many

    prisma_client = PrismaClient.__new__(PrismaClient)
    prisma_client.db = mock_prisma
    prisma_client._spend_log_transactions_lock = asyncio.Lock()
    prisma_client.spend_log_transactions = []
    prisma_client.proxy_logging_obj = MagicMock()

    await prisma_client.get_data(
        table_name="spend",
        query_type="find_all",
    )

    mock_find_many.assert_called_once()
    call_kwargs = mock_find_many.call_args[1]
    assert "take" in call_kwargs, "find_many on spend logs must include 'take' to prevent OOM"
    assert call_kwargs["take"] == 1000


@pytest.mark.asyncio
async def test_spend_log_find_many_respects_explicit_limit():
    """
    When get_data is called with an explicit limit parameter,
    it should use that limit instead of the default.
    """
    from litellm.proxy.utils import PrismaClient

    mock_prisma = MagicMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma.litellm_spendlogs.find_many = mock_find_many

    prisma_client = PrismaClient.__new__(PrismaClient)
    prisma_client.db = mock_prisma
    prisma_client._spend_log_transactions_lock = asyncio.Lock()
    prisma_client.spend_log_transactions = []
    prisma_client.proxy_logging_obj = MagicMock()

    await prisma_client.get_data(
        table_name="spend",
        query_type="find_all",
        limit=500,
    )

    mock_find_many.assert_called_once()
    call_kwargs = mock_find_many.call_args[1]
    assert call_kwargs["take"] == 500


@pytest.mark.asyncio
async def test_spend_log_find_many_with_key_val_respects_explicit_limit():
    """
    When get_data is called with key_val and an explicit limit,
    it should use that limit.
    """
    from litellm.proxy.utils import PrismaClient

    mock_prisma = MagicMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma.litellm_spendlogs.find_many = mock_find_many

    prisma_client = PrismaClient.__new__(PrismaClient)
    prisma_client.db = mock_prisma
    prisma_client._spend_log_transactions_lock = asyncio.Lock()
    prisma_client.spend_log_transactions = []
    prisma_client.proxy_logging_obj = MagicMock()

    await prisma_client.get_data(
        table_name="spend",
        key_val={"key": "user", "value": "test-user"},
        query_type="find_all",
        limit=200,
    )

    mock_find_many.assert_called_once()
    call_kwargs = mock_find_many.call_args[1]
    assert call_kwargs["take"] == 200

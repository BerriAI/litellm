"""
Regression tests for the spend_log_transactions queue cap.

Validates that:
1. The queue does not grow beyond MAX_SPEND_LOG_TRANSACTIONS
2. Oldest entries are dropped when the cap is reached
3. A warning is logged when entries are dropped
4. Normal operation (below cap) is unaffected
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_prisma_client():
    """Create a mock PrismaClient with spend_log_transactions list and lock."""
    client = MagicMock()
    client.spend_log_transactions = []
    client._spend_log_transactions_lock = asyncio.Lock()
    return client


@pytest.fixture
def db_writer():
    """Create a DBSpendUpdateWriter instance."""
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    return DBSpendUpdateWriter()


@pytest.mark.asyncio
async def test_spend_log_queue_normal_append(mock_prisma_client, db_writer):
    """should append spend log entries normally when below cap"""
    payload = {"request_id": "req-1", "spend": 0.001}
    await db_writer._insert_spend_log_to_db(
        payload=payload, prisma_client=mock_prisma_client
    )
    assert len(mock_prisma_client.spend_log_transactions) == 1
    assert mock_prisma_client.spend_log_transactions[0]["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_spend_log_queue_cap_drops_oldest(mock_prisma_client, db_writer):
    """should drop oldest entries when queue reaches MAX_SPEND_LOG_TRANSACTIONS"""
    cap = 100  # Use a small cap for testing

    with patch("litellm.constants.MAX_SPEND_LOG_TRANSACTIONS", cap):
        # Fill the queue to capacity
        for i in range(cap):
            mock_prisma_client.spend_log_transactions.append(
                {"request_id": f"req-{i}", "spend": 0.001}
            )

        assert len(mock_prisma_client.spend_log_transactions) == cap

        # Now insert one more — should trigger the drop
        payload = {"request_id": "req-overflow", "spend": 0.001}
        await db_writer._insert_spend_log_to_db(
            payload=payload, prisma_client=mock_prisma_client
        )

        # Queue should be smaller than cap + 1 (oldest were dropped)
        # Drop count is max(1, cap // 10) = 10 for cap=100
        drop_count = max(1, cap // 10)
        expected_len = cap - drop_count + 1  # dropped 10, added 1
        assert len(mock_prisma_client.spend_log_transactions) == expected_len

        # The newest entry should be the one we just added
        assert mock_prisma_client.spend_log_transactions[-1]["request_id"] == "req-overflow"

        # The oldest entries (req-0 through req-9) should have been dropped
        remaining_ids = [
            e["request_id"] for e in mock_prisma_client.spend_log_transactions
        ]
        for i in range(drop_count):
            assert f"req-{i}" not in remaining_ids


@pytest.mark.asyncio
async def test_spend_log_queue_cap_logs_warning(mock_prisma_client, db_writer):
    """should log a warning when queue cap is reached"""
    cap = 50

    with patch("litellm.constants.MAX_SPEND_LOG_TRANSACTIONS", cap), patch(
        "litellm.proxy.db.db_spend_update_writer.verbose_proxy_logger"
    ) as mock_logger:
        # Fill to capacity
        for i in range(cap):
            mock_prisma_client.spend_log_transactions.append(
                {"request_id": f"req-{i}", "spend": 0.001}
            )

        # Insert one more
        await db_writer._insert_spend_log_to_db(
            payload={"request_id": "req-overflow", "spend": 0.001},
            prisma_client=mock_prisma_client,
        )

        # Should have logged a warning
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "spend_log_transactions queue at capacity" in warning_msg


@pytest.mark.asyncio
async def test_spend_log_queue_no_append_without_prisma(db_writer):
    """should skip appending when prisma_client is None"""
    payload = {"request_id": "req-1", "spend": 0.001}
    result = await db_writer._insert_spend_log_to_db(
        payload=payload, prisma_client=None
    )
    assert result is None


@pytest.mark.asyncio
async def test_spend_log_queue_repeated_overflow(mock_prisma_client, db_writer):
    """should handle repeated overflows without growing unboundedly"""
    cap = 20

    with patch("litellm.constants.MAX_SPEND_LOG_TRANSACTIONS", cap):
        # Insert 3x the cap
        for i in range(cap * 3):
            await db_writer._insert_spend_log_to_db(
                payload={"request_id": f"req-{i}", "spend": 0.001},
                prisma_client=mock_prisma_client,
            )

        # Queue should never exceed cap + 1 (one new entry after drop)
        assert len(mock_prisma_client.spend_log_transactions) <= cap + 1
        # The most recent entry should be the last one inserted
        assert (
            mock_prisma_client.spend_log_transactions[-1]["request_id"]
            == f"req-{cap * 3 - 1}"
        )

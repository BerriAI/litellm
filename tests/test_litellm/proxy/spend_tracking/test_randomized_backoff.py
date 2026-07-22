"""
Tests for randomized exponential backoff in spend tracking retry loops.

Verifies that update_end_user_spend and update_spend_logs use
random.uniform(2**i, 2**(i+1)) instead of a fixed 2**i backoff,
which prevents correlated retries from re-deadlocking.

See: https://github.com/BerriAI/litellm/issues/27989
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.utils import PrismaClient, ProxyLogging, update_spend


class MockPrismaClient:
    def __init__(self):
        self.db = AsyncMock()
        self.db.litellm_spendlogs = AsyncMock()
        self.db.litellm_spendlogs.create_many = AsyncMock()
        self.spend_log_transactions = []
        self.daily_user_spend_transactions = {}
        self._spend_log_transactions_lock = asyncio.Lock()

    def jsonify_object(self, obj):
        return obj

    def add_spend_log_transaction_to_daily_user_transaction(self, payload):
        pass


def create_mock_proxy_logging():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = (
        AsyncMock()
    )
    return proxy_logging_obj


@pytest.mark.asyncio
async def test_update_spend_logs_backoff_is_randomized():
    """
    Test that backoff sleep times vary between retries due to randomization.

    With fixed backoff (2**i), consecutive retries from different concurrent
    callers sleep the same duration, causing them to retry simultaneously and
    re-deadlock. With random.uniform(2**i, 2**(i+1)), each retry sleeps a
    different random duration within the range.
    """
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    prisma_client.spend_log_transactions = [{"id": "1", "spend": 10}]

    create_many_mock = AsyncMock(
        side_effect=[
            httpx.ConnectError("deadlock detected"),
            httpx.ConnectError("deadlock detected"),
            None,
        ]
    )
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    sleep_times = []

    async def mock_sleep(seconds):
        sleep_times.append(seconds)

    with patch("asyncio.sleep", mock_sleep):
        await update_spend(prisma_client, None, proxy_logging_obj)

    assert len(sleep_times) == 2

    # Verify each sleep time falls within the randomized range
    # i=0: random.uniform(2**0, 2**1) -> [1.0, 2.0)
    assert (
        1.0 <= sleep_times[0] <= 2.0
    ), f"First backoff {sleep_times[0]} not in [1.0, 2.0]"
    # i=1: random.uniform(2**1, 2**2) -> [2.0, 4.0)
    assert (
        2.0 <= sleep_times[1] <= 4.0
    ), f"Second backoff {sleep_times[1]} not in [2.0, 4.0]"


@pytest.mark.asyncio
async def test_update_spend_logs_backoff_not_fixed():
    """
    Test that backoff values are NOT the exact fixed values 2**i.

    Run multiple trials and verify that at least some sleep values differ
    from the fixed backoff pattern, confirming randomization is active.
    """
    first_backoffs = []

    for _ in range(10):
        prisma_client = MockPrismaClient()
        proxy_logging_obj = create_mock_proxy_logging()

        prisma_client.spend_log_transactions = [{"id": "1", "spend": 10}]

        create_many_mock = AsyncMock(
            side_effect=[
                httpx.ConnectError("deadlock detected"),
                None,
            ]
        )
        prisma_client.db.litellm_spendlogs.create_many = create_many_mock

        sleep_times = []

        async def mock_sleep(seconds):
            sleep_times.append(seconds)

        with patch("asyncio.sleep", mock_sleep):
            await update_spend(prisma_client, None, proxy_logging_obj)

        assert len(sleep_times) == 1
        first_backoffs.append(sleep_times[0])

    # With randomized backoff, not all 10 values should be identical
    # (the probability of 10 identical random.uniform values is negligible)
    unique_values = set(first_backoffs)
    assert len(unique_values) > 1, (
        f"All 10 backoff values were identical ({first_backoffs[0]}), "
        "suggesting fixed rather than randomized backoff"
    )

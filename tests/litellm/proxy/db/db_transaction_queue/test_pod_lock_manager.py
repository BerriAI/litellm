import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager


# Mock Prisma client class
class MockPrismaClient:
    def __init__(self):
        self.db = MagicMock()
        self.db.litellm_cronjob = AsyncMock()


@pytest.fixture
def mock_prisma(monkeypatch):
    mock_client = MockPrismaClient()

    # Mock the prisma_client import in proxy_server
    def mock_get_prisma():
        return mock_client

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_client)
    return mock_client


@pytest.fixture
def pod_lock_manager():
    return PodLockManager(cronjob_id="test_job")


@pytest.mark.asyncio
async def test_acquire_lock_success(pod_lock_manager, mock_prisma):
    """
    Test that the lock is acquired successfully when no existing lock exists
    """
    # Mock find_unique to return None (no existing lock)
    mock_prisma.db.litellm_cronjob.find_unique.return_value = None

    # Mock successful creation of new lock
    mock_response = AsyncMock()
    mock_response.status = "ACTIVE"
    mock_response.pod_id = pod_lock_manager.pod_id
    mock_prisma.db.litellm_cronjob.create.return_value = mock_response

    result = await pod_lock_manager.acquire_lock()
    assert result == True

    # Verify find_unique was called
    mock_prisma.db.litellm_cronjob.find_unique.assert_called_once()
    # Verify create was called with correct parameters
    mock_prisma.db.litellm_cronjob.create.assert_called_once()
    call_args = mock_prisma.db.litellm_cronjob.create.call_args[1]
    assert call_args["data"]["cronjob_id"] == "test_job"
    assert call_args["data"]["pod_id"] == pod_lock_manager.pod_id
    assert call_args["data"]["status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_acquire_lock_existing_active(pod_lock_manager, mock_prisma):
    """
    Test that the lock is not acquired if there's an active lock by different pod
    """
    # Mock existing active lock
    mock_existing = AsyncMock()
    mock_existing.status = "ACTIVE"
    mock_existing.pod_id = "different_pod_id"
    mock_existing.ttl = datetime.now(timezone.utc) + timedelta(seconds=30)  # Future TTL
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    result = await pod_lock_manager.acquire_lock()
    assert result == False

    # Verify find_unique was called but update/create were not
    mock_prisma.db.litellm_cronjob.find_unique.assert_called_once()
    mock_prisma.db.litellm_cronjob.update.assert_not_called()
    mock_prisma.db.litellm_cronjob.create.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_lock_expired(pod_lock_manager, mock_prisma):
    """
    Test that the lock can be acquired if existing lock is expired
    """
    # Mock existing expired lock
    mock_existing = AsyncMock()
    mock_existing.status = "ACTIVE"
    mock_existing.pod_id = "different_pod_id"
    mock_existing.ttl = datetime.now(timezone.utc) - timedelta(seconds=30)  # Past TTL
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    # Mock successful update
    mock_updated = AsyncMock()
    mock_updated.pod_id = pod_lock_manager.pod_id
    mock_prisma.db.litellm_cronjob.update.return_value = mock_updated

    result = await pod_lock_manager.acquire_lock()
    assert result == True

    # Verify both find_unique and update were called
    mock_prisma.db.litellm_cronjob.find_unique.assert_called_once()
    mock_prisma.db.litellm_cronjob.update.assert_called_once()


@pytest.mark.asyncio
async def test_renew_lock(pod_lock_manager, mock_prisma):
    """
    Test that the renew lock calls the DB update method with the correct parameters
    """
    mock_prisma.db.litellm_cronjob.update.return_value = AsyncMock()

    await pod_lock_manager.renew_lock()

    # Verify update was called with correct parameters
    mock_prisma.db.litellm_cronjob.update.assert_called_once()
    call_args = mock_prisma.db.litellm_cronjob.update.call_args[1]
    assert call_args["where"]["cronjob_id"] == "test_job"
    assert call_args["where"]["pod_id"] == pod_lock_manager.pod_id
    assert "ttl" in call_args["data"]
    assert "last_updated" in call_args["data"]


@pytest.mark.asyncio
async def test_release_lock(pod_lock_manager, mock_prisma):
    """
    Test that the release lock calls the DB update method with the correct parameters

    specifically, the status should be set to INACTIVE
    """
    mock_prisma.db.litellm_cronjob.update.return_value = AsyncMock()

    await pod_lock_manager.release_lock()

    # Verify update was called with correct parameters
    mock_prisma.db.litellm_cronjob.update.assert_called_once()
    call_args = mock_prisma.db.litellm_cronjob.update.call_args[1]
    assert call_args["where"]["cronjob_id"] == "test_job"
    assert call_args["where"]["pod_id"] == pod_lock_manager.pod_id
    assert call_args["data"]["status"] == "INACTIVE"


@pytest.mark.asyncio
async def test_prisma_client_none(pod_lock_manager, monkeypatch):
    # Mock prisma_client as None
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    # Test all methods with None client
    assert await pod_lock_manager.acquire_lock() == False
    assert await pod_lock_manager.renew_lock() == False
    assert await pod_lock_manager.release_lock() == False


@pytest.mark.asyncio
async def test_database_error_handling(pod_lock_manager, mock_prisma):
    # Mock database errors
    mock_prisma.db.litellm_cronjob.upsert.side_effect = Exception("Database error")
    mock_prisma.db.litellm_cronjob.update.side_effect = Exception("Database error")

    # Test error handling in all methods
    assert await pod_lock_manager.acquire_lock() == False
    await pod_lock_manager.renew_lock()  # Should not raise exception
    await pod_lock_manager.release_lock()  # Should not raise exception


@pytest.mark.asyncio
async def test_acquire_lock_inactive_status(pod_lock_manager, mock_prisma):
    """
    Test that the lock can be acquired if existing lock is INACTIVE
    """
    # Mock existing inactive lock
    mock_existing = AsyncMock()
    mock_existing.status = "INACTIVE"
    mock_existing.pod_id = "different_pod_id"
    mock_existing.ttl = datetime.now(timezone.utc) + timedelta(seconds=30)
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    # Mock successful update
    mock_updated = AsyncMock()
    mock_updated.pod_id = pod_lock_manager.pod_id
    mock_prisma.db.litellm_cronjob.update.return_value = mock_updated

    result = await pod_lock_manager.acquire_lock()
    assert result == True

    mock_prisma.db.litellm_cronjob.update.assert_called_once()


@pytest.mark.asyncio
async def test_acquire_lock_same_pod(pod_lock_manager, mock_prisma):
    """
    Test that the lock returns True if the same pod already holds the lock
    """
    # Mock existing active lock held by same pod
    mock_existing = AsyncMock()
    mock_existing.status = "ACTIVE"
    mock_existing.pod_id = pod_lock_manager.pod_id
    mock_existing.ttl = datetime.now(timezone.utc) + timedelta(seconds=30)
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    result = await pod_lock_manager.acquire_lock()
    assert result == True

    # Verify no update was needed
    mock_prisma.db.litellm_cronjob.update.assert_not_called()
    mock_prisma.db.litellm_cronjob.create.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_lock_race_condition(pod_lock_manager, mock_prisma):
    """
    Test handling of potential race conditions during lock acquisition
    """
    # First find_unique returns None
    mock_prisma.db.litellm_cronjob.find_unique.return_value = None

    # But create raises unique constraint violation
    mock_prisma.db.litellm_cronjob.create.side_effect = Exception(
        "Unique constraint violation"
    )

    result = await pod_lock_manager.acquire_lock()
    assert result == False


@pytest.mark.asyncio
async def test_ttl_calculation(pod_lock_manager, mock_prisma):
    """
    Test that TTL is calculated correctly when acquiring lock
    """
    mock_prisma.db.litellm_cronjob.find_unique.return_value = None
    mock_prisma.db.litellm_cronjob.create.return_value = AsyncMock()

    await pod_lock_manager.acquire_lock()

    call_args = mock_prisma.db.litellm_cronjob.create.call_args[1]
    ttl = call_args["data"]["ttl"]

    # Verify TTL is in the future by DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
    expected_ttl = datetime.now(timezone.utc) + timedelta(
        seconds=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
    )
    assert abs((ttl - expected_ttl).total_seconds()) < 1  # Allow 1 second difference


@pytest.mark.asyncio
async def test_concurrent_lock_acquisition_simulation(mock_prisma):
    """
    Simulate multiple pods trying to acquire the lock simultaneously
    """
    pod1 = PodLockManager(cronjob_id="test_job")
    pod2 = PodLockManager(cronjob_id="test_job")
    pod3 = PodLockManager(cronjob_id="test_job")

    # Simulate first pod getting the lock
    mock_prisma.db.litellm_cronjob.find_unique.return_value = None
    mock_response = AsyncMock()
    mock_response.pod_id = pod1.pod_id
    mock_response.status = "ACTIVE"
    mock_prisma.db.litellm_cronjob.create.return_value = mock_response

    # First pod should get the lock
    result1 = await pod1.acquire_lock()
    assert result1 == True

    # Simulate other pods trying to acquire same lock immediately after
    mock_existing = AsyncMock()
    mock_existing.status = "ACTIVE"
    mock_existing.pod_id = pod1.pod_id
    mock_existing.ttl = datetime.now(timezone.utc) + timedelta(seconds=30)
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    # Other pods should fail to acquire
    result2 = await pod2.acquire_lock()
    result3 = await pod3.acquire_lock()
    assert result2 == False
    assert result3 == False


@pytest.mark.asyncio
async def test_lock_takeover_race_condition(mock_prisma):
    """
    Test scenario where multiple pods try to take over an expired lock
    """
    pod1 = PodLockManager(cronjob_id="test_job")
    pod2 = PodLockManager(cronjob_id="test_job")

    # Simulate expired lock
    mock_existing = AsyncMock()
    mock_existing.status = "ACTIVE"
    mock_existing.pod_id = "old_pod"
    mock_existing.ttl = datetime.now(timezone.utc) - timedelta(seconds=30)
    mock_prisma.db.litellm_cronjob.find_unique.return_value = mock_existing

    # Simulate pod1's update succeeding
    mock_update1 = AsyncMock()
    mock_update1.pod_id = pod1.pod_id
    mock_prisma.db.litellm_cronjob.update.return_value = mock_update1

    # First pod should successfully take over
    result1 = await pod1.acquire_lock()
    assert result1 == True

    # Simulate pod2's update failing due to race condition
    mock_prisma.db.litellm_cronjob.update.side_effect = Exception(
        "Row was updated by another transaction"
    )

    # Second pod should fail to take over
    result2 = await pod2.acquire_lock()
    assert result2 == False

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
from litellm.proxy.db.pod_lock_manager import PodLockManager


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
    # Mock successful lock acquisition
    mock_response = AsyncMock()
    mock_response.status = "ACTIVE"
    mock_response.pod_id = pod_lock_manager.pod_id
    mock_prisma.db.litellm_cronjob.upsert.return_value = mock_response

    result = await pod_lock_manager.acquire_lock()
    assert result == True

    # Verify upsert was called with correct parameters
    mock_prisma.db.litellm_cronjob.upsert.assert_called_once()
    call_args = mock_prisma.db.litellm_cronjob.upsert.call_args[1]
    assert call_args["where"]["cronjob_id"] == "test_job"
    assert "create" in call_args["data"]
    assert "update" in call_args["data"]


@pytest.mark.asyncio
async def test_acquire_lock_failure(pod_lock_manager, mock_prisma):
    """
    Test that the lock is not acquired if the lock is held by a different pod
    """
    # Mock failed lock acquisition (different pod holds the lock)
    mock_response = AsyncMock()
    mock_response.status = "ACTIVE"
    mock_response.pod_id = "different_pod_id"
    mock_prisma.db.litellm_cronjob.upsert.return_value = mock_response

    result = await pod_lock_manager.acquire_lock()
    assert result == False


@pytest.mark.asyncio
async def test_renew_lock(pod_lock_manager, mock_prisma):
    # Mock successful lock renewal
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
    # Mock successful lock release
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

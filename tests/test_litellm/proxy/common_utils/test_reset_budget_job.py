import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.utils import ProxyLogging


# Mock classes for testing
class MockPrismaClient:
    def __init__(self):
        self.data = {"key": [], "user": [], "team": []}
        self.updated_data = {"key": [], "user": [], "team": []}

    async def get_data(self, table_name, query_type, **kwargs):
        return self.data.get(table_name, [])

    async def update_data(self, query_type, data_list, table_name):
        self.updated_data[table_name] = data_list
        return data_list


class MockProxyLogging:
    class MockServiceLogging:
        async def async_service_success_hook(self, **kwargs):
            pass

        async def async_service_failure_hook(self, **kwargs):
            pass

    def __init__(self):
        self.service_logging_obj = self.MockServiceLogging()


# Test fixtures
@pytest.fixture
def mock_prisma_client():
    return MockPrismaClient()


@pytest.fixture
def mock_proxy_logging():
    return MockProxyLogging()


@pytest.fixture
def reset_budget_job(mock_prisma_client, mock_proxy_logging):
    return ResetBudgetJob(
        proxy_logging_obj=mock_proxy_logging, prisma_client=mock_prisma_client
    )


# Helper function to run async tests
async def run_async_test(coro):
    return await coro


# Tests
def test_reset_budget_for_key(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_key = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": "30d",
            "budget_reset_at": now,
            "id": "test-key-1",
        },
    )

    mock_prisma_client.data["key"] = [test_key]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    # Verify results
    assert len(mock_prisma_client.updated_data["key"]) == 1
    updated_key = mock_prisma_client.updated_data["key"][0]
    assert updated_key.spend == 0.0
    assert updated_key.budget_reset_at > now


def test_reset_budget_for_user(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_user = type(
        "LiteLLM_UserTable",
        (),
        {
            "spend": 200.0,
            "budget_duration": "7d",
            "budget_reset_at": now,
            "id": "test-user-1",
        },
    )

    mock_prisma_client.data["user"] = [test_user]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_users())

    # Verify results
    assert len(mock_prisma_client.updated_data["user"]) == 1
    updated_user = mock_prisma_client.updated_data["user"][0]
    assert updated_user.spend == 0.0
    assert updated_user.budget_reset_at > now


def test_reset_budget_for_team(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_team = type(
        "LiteLLM_TeamTable",
        (),
        {
            "spend": 500.0,
            "budget_duration": "1mo",
            "budget_reset_at": now,
            "id": "test-team-1",
        },
    )

    mock_prisma_client.data["team"] = [test_team]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_teams())

    # Verify results
    assert len(mock_prisma_client.updated_data["team"]) == 1
    updated_team = mock_prisma_client.updated_data["team"][0]
    assert updated_team.spend == 0.0
    assert updated_team.budget_reset_at > now


def test_reset_budget_all(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)

    # Create test objects for all three types
    test_key = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": "30d",
            "budget_reset_at": now,
            "id": "test-key-1",
        },
    )

    test_user = type(
        "LiteLLM_UserTable",
        (),
        {
            "spend": 200.0,
            "budget_duration": "7d",
            "budget_reset_at": now,
            "id": "test-user-1",
        },
    )

    test_team = type(
        "LiteLLM_TeamTable",
        (),
        {
            "spend": 500.0,
            "budget_duration": "1mo",
            "budget_reset_at": now,
            "id": "test-team-1",
        },
    )

    mock_prisma_client.data["key"] = [test_key]
    mock_prisma_client.data["user"] = [test_user]
    mock_prisma_client.data["team"] = [test_team]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget())

    # Verify results
    assert len(mock_prisma_client.updated_data["key"]) == 1
    assert len(mock_prisma_client.updated_data["user"]) == 1
    assert len(mock_prisma_client.updated_data["team"]) == 1

    # Check that all spends were reset to 0
    assert mock_prisma_client.updated_data["key"][0].spend == 0.0
    assert mock_prisma_client.updated_data["user"][0].spend == 0.0
    assert mock_prisma_client.updated_data["team"][0].spend == 0.0

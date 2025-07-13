import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.utils import ProxyLogging


# Mock classes for testing
class MockLiteLLMTeamMembership:
    async def update_many(
        self, where: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Mock the update_many method for litellm_teammembership
        return {"count": 1}


class MockDB:
    def __init__(self):
        self.litellm_teammembership = MockLiteLLMTeamMembership()


class MockPrismaClient:
    def __init__(self):
        self.data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
        }
        self.updated_data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
        }
        self.db = MockDB()

    async def get_data(self, table_name, query_type, **kwargs):
        data = self.data.get(table_name, [])

        # Handle specific filtering for budget table queries
        if table_name == "budget" and query_type == "find_all" and "reset_at" in kwargs:
            # Return budgets that need to be reset (simulate expired budgets)
            return [item for item in data if hasattr(item, "budget_reset_at")]

        # Handle specific filtering for enduser table queries
        if (
            table_name == "enduser"
            and query_type == "find_all"
            and "budget_id_list" in kwargs
        ):
            budget_id_list = kwargs["budget_id_list"]
            # Return endusers that match the budget IDs
            return [
                item
                for item in data
                if hasattr(item, "litellm_budget_table")
                and hasattr(item.litellm_budget_table, "budget_id")
                and item.litellm_budget_table.budget_id in budget_id_list
            ]

        # Handle key queries with expires and reset_at
        if (
            table_name == "key"
            and query_type == "find_all"
            and ("expires" in kwargs or "reset_at" in kwargs)
        ):
            return [item for item in data if hasattr(item, "budget_reset_at")]

        return data

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


def test_reset_budget_for_enduser(reset_budget_job, mock_prisma_client):
    # Setup test data
    now = datetime.now(timezone.utc)
    test_budget = type(
        "LiteLLM_BudgetTable",
        (),
        {
            "max_budget": 500.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "budget_id": "test-budget-1",
        },
    )

    test_enduser = type(
        "LiteLLM_EndUserTable",
        (),
        {
            "spend": 20.0,
            "litellm_budget_table": test_budget,
            "user_id": "test-enduser-1",
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Verify results
    assert len(mock_prisma_client.updated_data["enduser"]) == 1
    assert len(mock_prisma_client.updated_data["budget"]) == 1
    updated_enduser = mock_prisma_client.updated_data["enduser"][0]
    updated_budget = mock_prisma_client.updated_data["budget"][0]
    assert updated_enduser.spend == 0.0
    assert updated_budget.budget_reset_at > now


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

    test_budget = type(
        "LiteLLM_BudgetTable",
        (),
        {
            "max_budget": 500.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "budget_id": "test-budget-1",
        },
    )

    test_enduser = type(
        "LiteLLM_EndUserTable",
        (),
        {
            "spend": 20.0,
            "litellm_budget_table": test_budget,
            "user_id": "test-enduser-1",
        },
    )

    mock_prisma_client.data["key"] = [test_key]
    mock_prisma_client.data["user"] = [test_user]
    mock_prisma_client.data["team"] = [test_team]
    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget())

    # Verify results
    assert len(mock_prisma_client.updated_data["key"]) == 1
    assert len(mock_prisma_client.updated_data["user"]) == 1
    assert len(mock_prisma_client.updated_data["team"]) == 1
    assert len(mock_prisma_client.updated_data["enduser"]) == 1
    assert len(mock_prisma_client.updated_data["budget"]) == 1

    # Check that all spends were reset to 0
    assert mock_prisma_client.updated_data["key"][0].spend == 0.0
    assert mock_prisma_client.updated_data["user"][0].spend == 0.0
    assert mock_prisma_client.updated_data["team"][0].spend == 0.0
    assert mock_prisma_client.updated_data["enduser"][0].spend == 0.0

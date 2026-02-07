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


class MockLiteLLMOrganizationTable:
    def __init__(self, organizations_data: List[Any]):
        self.organizations_data = organizations_data

    async def find_many(self, where: Dict[str, Any]) -> List[Any]:
        # Mock find_many for organizations - filter by budget_id
        if "budget_id" in where and "in" in where["budget_id"]:
            budget_id_list = where["budget_id"]["in"]
            return [
                org
                for org in self.organizations_data
                if hasattr(org, "budget_id") and org.budget_id in budget_id_list
            ]
        return []

    async def update_many(
        self, where: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Mock the update_many method for organizations
        # Update spend in the organizations_data
        if "budget_id" in where and "in" in where["budget_id"]:
            budget_id_list = where["budget_id"]["in"]
            count = 0
            for org in self.organizations_data:
                if hasattr(org, "budget_id") and org.budget_id in budget_id_list:
                    if "spend" in data:
                        org.spend = data["spend"]
                    count += 1
            return {"count": count}
        return {"count": 0}


class MockDB:
    def __init__(self, organizations_data: List[Any] = None):
        self.litellm_teammembership = MockLiteLLMTeamMembership()
        self.litellm_organizationtable = MockLiteLLMOrganizationTable(
            organizations_data or []
        )


class MockPrismaClient:
    def __init__(self):
        self.data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
            "organization": [],
        }
        self.updated_data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
            "organization": [],
        }
        self.db = MockDB(organizations_data=[])
        
    def _update_db_organizations(self):
        """Update the MockDB with current organization data"""
        self.db.litellm_organizationtable = MockLiteLLMOrganizationTable(
            self.data.get("organization", [])
        )

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

    test_organization = type(
        "LiteLLM_OrganizationTable",
        (),
        {
            "organization_id": "test-org-1",
            "organization_alias": "Test Org",
            "budget_id": "test-budget-1",
            "spend": 150.0,
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]
    mock_prisma_client.data["organization"] = [test_organization]
    mock_prisma_client._update_db_organizations()

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Verify results
    assert len(mock_prisma_client.updated_data["enduser"]) == 1
    assert len(mock_prisma_client.updated_data["budget"]) == 1
    updated_enduser = mock_prisma_client.updated_data["enduser"][0]
    updated_budget = mock_prisma_client.updated_data["budget"][0]
    assert updated_enduser.spend == 0.0
    assert updated_budget.budget_reset_at > now
    # Verify organization spend was reset
    assert test_organization.spend == 0.0


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

    test_organization = type(
        "LiteLLM_OrganizationTable",
        (),
        {
            "organization_id": "test-org-1",
            "organization_alias": "Test Org",
            "budget_id": "test-budget-1",
            "spend": 150.0,
        },
    )

    mock_prisma_client.data["key"] = [test_key]
    mock_prisma_client.data["user"] = [test_user]
    mock_prisma_client.data["team"] = [test_team]
    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]
    mock_prisma_client.data["organization"] = [test_organization]
    mock_prisma_client._update_db_organizations()

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
    # Verify organization spend was reset
    assert test_organization.spend == 0.0


def test_reset_budget_for_organization(reset_budget_job, mock_prisma_client):
    """
    Test that organizations with expired budgets have their spend reset to 0.
    This tests the new organization budget reset logic.
    """
    # Setup test data
    now = datetime.now(timezone.utc)
    test_budget = type(
        "LiteLLM_BudgetTable",
        (),
        {
            "max_budget": 1000.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "budget_id": "test-budget-org-1",
            "created_at": now - timedelta(days=2),
        },
    )

    test_organization_1 = type(
        "LiteLLM_OrganizationTable",
        (),
        {
            "organization_id": "test-org-1",
            "organization_alias": "Test Org 1",
            "budget_id": "test-budget-org-1",
            "spend": 250.0,
        },
    )

    test_organization_2 = type(
        "LiteLLM_OrganizationTable",
        (),
        {
            "organization_id": "test-org-2",
            "organization_alias": "Test Org 2",
            "budget_id": "test-budget-org-1",
            "spend": 350.0,
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["organization"] = [
        test_organization_1,
        test_organization_2,
    ]
    mock_prisma_client._update_db_organizations()

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Verify results
    assert len(mock_prisma_client.updated_data["budget"]) == 1
    updated_budget = mock_prisma_client.updated_data["budget"][0]
    assert updated_budget.budget_reset_at > now

    # Verify both organizations had their spend reset to 0
    assert test_organization_1.spend == 0.0
    assert test_organization_2.spend == 0.0


def test_reset_budget_for_organization_failure(reset_budget_job, mock_prisma_client):
    """
    Test that organization reset failures are properly handled and tracked.
    This tests the error handling logic for organization budget resets.
    The function catches exceptions and logs them, but doesn't re-raise them.
    """
    # Setup test data
    now = datetime.now(timezone.utc)
    test_budget = type(
        "LiteLLM_BudgetTable",
        (),
        {
            "max_budget": 1000.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "budget_id": "test-budget-org-fail",
            "created_at": now - timedelta(days=2),
        },
    )

    test_organization = type(
        "LiteLLM_OrganizationTable",
        (),
        {
            "organization_id": "test-org-fail",
            "organization_alias": "Test Org Fail",
            "budget_id": "test-budget-org-fail",
            "spend": 500.0,
        },
    )

    # Create a mock that will raise an exception on update_many
    class FailingMockLiteLLMOrganizationTable(MockLiteLLMOrganizationTable):
        async def update_many(
            self, where: Dict[str, Any], data: Dict[str, Any]
        ) -> Dict[str, Any]:
            raise Exception("Database connection failed")

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["organization"] = [test_organization]
    mock_prisma_client.db.litellm_organizationtable = (
        FailingMockLiteLLMOrganizationTable([test_organization])
    )

    # Run the test - the function catches exceptions internally and logs them
    # It doesn't re-raise them, so the function completes normally
    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Verify that the organization spend was NOT reset due to the failure
    # The spend should still be 500.0 because the update failed
    assert test_organization.spend == 500.0
    
    # Verify that the budget was still processed (it should be updated)
    assert len(mock_prisma_client.updated_data["budget"]) == 1

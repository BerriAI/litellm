"""
Unit tests for customer budget operations.

Tests customer update functionality related to budget management:
- Linking customers to existing budgets via budget_id
- Creating new budgets for customers with proper field validation
- Budget creation with required metadata fields
- Proper database relationship handling
- Budget initialization on customer creation
"""

from datetime import datetime, timedelta

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    NewCustomerRequest,
    UpdateCustomerRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.customer_endpoints import (
    new_budget_request,
    update_end_user,
)


@pytest.fixture
def mock_user_api_key_dict():
    """Mock user API key auth object."""
    mock_auth = MagicMock(spec=UserAPIKeyAuth)
    mock_auth.user_id = "test-admin-user"
    return mock_auth


@pytest.fixture
def mock_existing_customer():
    """Mock existing customer data."""
    return MagicMock(spec=LiteLLM_EndUserTable)


@pytest.fixture
def mock_budget_table():
    """Mock budget table data."""
    return MagicMock(spec=LiteLLM_BudgetTable)


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_with_budget_id(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test updating a customer to link them to an existing budget using budget_id.

    When only budget_id is provided (no budget creation fields), the customer
    should be linked to the existing budget without creating a new one.
    """
    # Arrange
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_existing_customer
    )

    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {
        "user_id": "test-user",
        "budget_id": "existing-budget-123",
        "blocked": False,
    }

    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=mock_updated_user
    )

    # Create update request with only budget_id (no other budget fields)
    update_request = UpdateCustomerRequest(
        user_id="test-user", budget_id="existing-budget-123"
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert
    # Verify that update was called on end user table with budget_id
    mock_prisma_client.db.litellm_endusertable.update.assert_called_once()
    call_args = mock_prisma_client.db.litellm_endusertable.update.call_args

    # Check that budget_id is in the update data for end user table
    update_data = call_args[1]["data"]  # kwargs['data']
    assert "budget_id" in update_data
    assert update_data["budget_id"] == "existing-budget-123"

    # Verify that NO budget creation was attempted
    assert not mock_prisma_client.db.litellm_budgettable.create.called


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_creates_budget_with_proper_relations(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test that creating a new budget for a customer uses proper database relations.

    When budget creation fields are provided, the system should create a budget
    with correct database relationship includes.
    """
    # Arrange
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_existing_customer
    )

    # Mock budget creation
    mock_created_budget = MagicMock()
    mock_created_budget.budget_id = "new-budget-456"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_created_budget
    )

    # Mock end user update
    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=mock_updated_user
    )

    # Create update request with budget creation fields (not just budget_id)
    update_request = UpdateCustomerRequest(
        user_id="test-user",
        max_budget=100.0,  # This triggers budget creation
        rpm_limit=200,  # Use valid budget field
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert
    # Verify budget creation was called with correct include field
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    call_args = mock_prisma_client.db.litellm_budgettable.create.call_args

    # Check that include uses correct relation name "end_users"
    include_param = call_args[1]["include"]  # kwargs['include']
    assert "end_users" in include_param
    assert include_param["end_users"] is True


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_creates_budget_with_required_fields(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test that creating a budget for a customer includes all required metadata fields.

    Budget creation should include created_by and updated_by fields for proper
    audit trail and data integrity.
    """
    # Arrange
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_existing_customer
    )

    # Mock budget creation
    mock_created_budget = MagicMock()
    mock_created_budget.budget_id = "new-budget-789"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_created_budget
    )

    # Mock end user update
    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=mock_updated_user
    )

    # Create update request with budget creation fields
    update_request = UpdateCustomerRequest(user_id="test-user", max_budget=200.0)

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert
    # Verify budget creation was called with required fields
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    call_args = mock_prisma_client.db.litellm_budgettable.create.call_args

    # Check that created_by and updated_by are present in creation data
    creation_data = call_args[1]["data"]  # kwargs['data']
    assert "created_by" in creation_data
    assert "updated_by" in creation_data

    # Verify the values are set correctly
    assert creation_data["created_by"] == "test-admin-user"
    assert creation_data["updated_by"] == "test-admin-user"

    # Verify budget fields are also included
    assert "max_budget" in creation_data
    assert creation_data["max_budget"] == 200.0


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_budget_creation_with_fallback_admin(
    mock_prisma_client, mock_existing_customer
):
    """
    Test budget creation falls back to admin name when user_id is not available.

    When the requesting user's ID is None, the system should use the configured
    proxy admin name for created_by and updated_by fields.
    """
    # Arrange - user with None user_id
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key_dict.user_id = None

    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_existing_customer
    )

    # Mock budget creation
    mock_created_budget = MagicMock()
    mock_created_budget.budget_id = "new-budget-fallback"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_created_budget
    )

    # Mock end user update
    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=mock_updated_user
    )

    # Create update request with budget creation fields
    update_request = UpdateCustomerRequest(
        user_id="test-user",
        max_budget=150.0,
        tpm_limit=1000,  # Add another budget field
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert
    # Verify budget creation was called with fallback admin name
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    call_args = mock_prisma_client.db.litellm_budgettable.create.call_args

    creation_data = call_args[1]["data"]  # kwargs['data']
    assert creation_data["created_by"] == "admin"  # litellm_proxy_admin_name
    assert creation_data["updated_by"] == "admin"


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_with_budget_id_and_creation_fields(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test customer update when both budget_id and budget creation fields are provided.

    When both linking (budget_id) and creation fields are provided, the system
    should prioritize creating a new budget and assign its ID to the customer.
    """
    # Arrange
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_existing_customer
    )

    # Mock budget creation
    mock_created_budget = MagicMock()
    mock_created_budget.budget_id = "new-budget-combo"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_created_budget
    )

    # Mock end user update
    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=mock_updated_user
    )

    # Create update request with both budget_id and budget creation fields
    update_request = UpdateCustomerRequest(
        user_id="test-user",
        budget_id="existing-budget-link",  # For linking to existing budget
        max_budget=300.0,  # This should trigger new budget creation
        rpm_limit=500,  # Use valid budget field
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert
    # Verify budget creation occurred (because max_budget was provided)
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()

    # Verify end user update was called
    mock_prisma_client.db.litellm_endusertable.update.assert_called_once()
    call_args = mock_prisma_client.db.litellm_endusertable.update.call_args

    # The update data should contain budget_id from the created budget, not the original budget_id
    update_data = call_args[1]["data"]
    assert update_data["budget_id"] == "new-budget-combo"  # From created budget


def test_new_budget_request_sets_budget_reset_at_when_duration_provided():
    """
    Test that new_budget_request auto-populates budget_reset_at when
    budget_duration is provided but budget_reset_at is not.

    Without this fix, budgets created via /customer/new with a budget_duration
    but no budget_reset_at would have budget_reset_at=NULL in the DB, causing
    the ResetBudgetJob to immediately pick them up and zero out enduser spend.
    """
    data = NewCustomerRequest(
        user_id="test-user",
        max_budget=10.0,
        budget_duration="30d",
    )

    before = datetime.utcnow()
    result = new_budget_request(data)
    after = datetime.utcnow()

    assert result is not None
    assert result.budget_reset_at is not None
    assert result.budget_duration == "30d"

    expected_min = before + timedelta(days=30)
    expected_max = after + timedelta(days=30)
    assert expected_min <= result.budget_reset_at <= expected_max


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_new_budget_persists_budget_duration_and_reset_at(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test that /customer/update persists budget_duration and computes
    budget_reset_at when creating a new budget for an end-user without one.

    Regression test for https://github.com/BerriAI/litellm/issues/33941 —
    budget_duration was silently dropped at request parsing (missing from
    UpdateCustomerRequest) and budget_reset_at was never computed, so the
    created budget never reset and the customer had a one-time cap.
    """
    # Arrange - end user exists but has no budget
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_existing_customer)

    mock_created_budget = MagicMock()
    mock_created_budget.budget_id = "new-budget-duration"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(return_value=mock_created_budget)

    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=mock_updated_user)

    update_request = UpdateCustomerRequest(
        user_id="test-user",
        max_budget=50.0,
        budget_duration="30d",
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert - budget_duration reached the create call and reset_at was computed
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    creation_data = mock_prisma_client.db.litellm_budgettable.create.call_args[1]["data"]

    assert creation_data["max_budget"] == 50.0
    assert creation_data["budget_duration"] == "30d"

    # budget_reset_at must match what get_budget_reset_time computes
    # (it snaps to calendar boundaries, so compare against the helper itself)
    reset_at = creation_data.get("budget_reset_at")
    assert reset_at is not None
    assert reset_at == get_budget_reset_time(budget_duration="30d")


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_existing_budget_updates_duration_and_reset_at(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test that updating budget_duration on an end-user with an existing budget
    updates the budget row and recomputes budget_reset_at (mirrors /budget/update).
    """
    # Arrange - end user already has a budget
    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "budget_id": "existing-budget-123",
        "litellm_budget_table": {
            "budget_id": "existing-budget-123",
            "max_budget": 10.0,
        },
    }
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_existing_customer)

    mock_prisma_client.db.litellm_budgettable.update = AsyncMock(return_value=MagicMock())

    mock_updated_user = MagicMock()
    mock_updated_user.model_dump.return_value = {"user_id": "test-user", "blocked": False}
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=mock_updated_user)

    update_request = UpdateCustomerRequest(
        user_id="test-user",
        budget_duration="1d",
    )

    # Act
    await update_end_user(update_request, mock_user_api_key_dict)

    # Assert - existing budget row updated with duration + computed reset_at
    mock_prisma_client.db.litellm_budgettable.update.assert_called_once()
    update_call = mock_prisma_client.db.litellm_budgettable.update.call_args
    assert update_call[1]["where"] == {"budget_id": "existing-budget-123"}
    update_data = update_call[1]["data"]
    assert update_data["budget_duration"] == "1d"
    assert update_data.get("budget_reset_at") is not None

    # No new budget row should be created
    assert not mock_prisma_client.db.litellm_budgettable.create.called


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
@patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
async def test_update_customer_invalid_budget_duration_returns_400(
    mock_prisma_client, mock_user_api_key_dict, mock_existing_customer
):
    """
    Test that an unsupported budget_duration (e.g. "fortnightly") is rejected
    with a 400 instead of being silently persisted with a next-midnight
    fallback reset (which would schedule an unintended daily reset).
    """
    from litellm.proxy._types import ProxyException

    mock_existing_customer.model_dump.return_value = {
        "user_id": "test-user",
        "blocked": False,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_existing_customer)
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock()

    update_request = UpdateCustomerRequest(
        user_id="test-user",
        max_budget=50.0,
        budget_duration="fortnightly",
    )

    with pytest.raises(ProxyException) as exc_info:
        await update_end_user(update_request, mock_user_api_key_dict)

    assert str(exc_info.value.code) in ("400", "HTTP_400_BAD_REQUEST")
    assert "budget_duration" in str(exc_info.value.message)

    # No budget row should have been written
    assert not mock_prisma_client.db.litellm_budgettable.create.called

import json
import os
import sys
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LiteLLM_UserTableFiltered,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    LiteLLM_UserTableWithKeyCount,
    _update_internal_user_params,
    get_user_key_counts,
    get_users,
    ui_view_users,
)
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_ui_view_users_with_null_email(mocker, caplog):
    """
    Test that /user/filter/ui endpoint returns users even when they have null email fields
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Create mock user data with null email
    mock_user = mocker.MagicMock()
    mock_user.model_dump.return_value = {
        "user_id": "test-user-null-email",
        "user_email": None,
        "user_role": "proxy_admin",
        "created_at": "2024-01-01T00:00:00Z",
    }

    # Setup the mock find_many response
    # Setup the mock find_many response as an async function
    async def mock_find_many(*args, **kwargs):
        return [mock_user]

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call ui_view_users function directly
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="test_user"),
        user_id="test_user",
        user_email=None,
        page=1,
        page_size=50,
    )

    assert response == [
        LiteLLM_UserTableFiltered(user_id="test-user-null-email", user_email=None)
    ]


def test_user_daily_activity_types():
    """
    Assert all fiels in SpendMetrics are reported in DailySpendMetadata as "total_"
    """
    from litellm.proxy.management_endpoints.common_daily_activity import (
        DailySpendMetadata,
        SpendMetrics,
    )

    # Create a SpendMetrics instance
    spend_metrics = SpendMetrics()

    # Create a DailySpendMetadata instance
    daily_spend_metadata = DailySpendMetadata()

    # Assert all fields in SpendMetrics are reported in DailySpendMetadata as "total_"
    for field in spend_metrics.__dict__:
        if field.startswith("total_"):
            assert hasattr(
                daily_spend_metadata, field
            ), f"Field {field} is not reported in DailySpendMetadata"
        else:
            assert not hasattr(
                daily_spend_metadata, field
            ), f"Field {field} is reported in DailySpendMetadata"


@pytest.mark.asyncio
async def test_get_users_includes_timestamps(mocker):
    """
    Test that /user/list endpoint returns users with created_at and updated_at fields.
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Create mock user data with timestamps
    mock_user_data = {
        "user_id": "test-user-timestamps",
        "user_email": "timestamps@example.com",
        "user_role": "internal_user",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = mock_user_data

    # Setup the mock find_many response as an async function
    async def mock_find_many(*args, **kwargs):
        return [mock_user_row]

    # Setup the mock count response as an async function
    async def mock_count(*args, **kwargs):
        return 1

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many
    mock_prisma_client.db.litellm_usertable.count = mock_count

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock the helper function get_user_key_counts
    async def mock_get_user_key_counts(*args, **kwargs):
        return {"test-user-timestamps": 0}

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_key_counts",
        mock_get_user_key_counts,
    )

    # Call get_users function directly
    response = await get_users(page=1, page_size=1)

    print("user /list response: ", response)

    # Assertions
    assert response is not None
    assert "users" in response
    assert "total" in response
    assert response["total"] == 1
    assert len(response["users"]) == 1

    user_response = response["users"][0]
    assert user_response.user_id == "test-user-timestamps"
    assert user_response.created_at is not None
    assert isinstance(user_response.created_at, datetime)
    assert user_response.updated_at is not None
    assert isinstance(user_response.updated_at, datetime)
    assert user_response.created_at == mock_user_data["created_at"]
    assert user_response.updated_at == mock_user_data["updated_at"]
    assert user_response.key_count == 0


def test_validate_sort_params():
    """
    Test that validate_sort_params returns None if sort_by is None
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _validate_sort_params,
    )

    assert _validate_sort_params(None, "asc") is None
    assert _validate_sort_params(None, "desc") is None
    assert _validate_sort_params("user_id", "asc") == {"user_id": "asc"}
    assert _validate_sort_params("user_id", "desc") == {"user_id": "desc"}
    with pytest.raises(Exception):
        _validate_sort_params("user_id", "invalid")


def test_update_user_request_pydantic_object():
    """
    Test that _update_internal_user_params correctly processes an email-only update
    """
    data = UpdateUserRequest(user_email="test@example.com")

    data_json = data.model_dump(exclude_unset=True)

    assert data_json == {"user_email": "test@example.com"}


def test_update_internal_user_params_email():
    """
    Test that _update_internal_user_params correctly processes an email-only update
    """
    from litellm.proxy._types import UpdateUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_user_params,
    )

    # Create test data with only email update
    data_json = {"user_email": "test@example.com"}
    data = UpdateUserRequest(user_email="test@example.com")

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert len(non_default_values) == 1  # Should only contain email
    assert "user_email" in non_default_values
    assert non_default_values["user_email"] == "test@example.com"
    assert "user_id" not in non_default_values  # Should not add user_id if not provided
    assert "max_budget" not in non_default_values  # Should not add default values
    assert "budget_duration" not in non_default_values  # Should not add default values


def test_update_internal_user_params_reset_spend_and_max_budget():
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/10495
    """
    from litellm.proxy._types import UpdateUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_user_params,
    )

    # Create test data with only email update
    data = UpdateUserRequest(spend=0, max_budget=0, user_id="test_user_id")
    data_json = data.model_dump(exclude_unset=True)

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert len(non_default_values) == 3  # Should only contain email
    assert "spend" in non_default_values
    assert non_default_values["spend"] == 0
    assert "max_budget" in non_default_values
    assert non_default_values["max_budget"] == 0
    assert "user_id" in non_default_values  # Should not add user_id if not provided
    assert non_default_values["user_id"] == "test_user_id"
    assert "budget_duration" not in non_default_values  # Should not add default values

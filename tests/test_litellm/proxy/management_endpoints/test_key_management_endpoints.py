import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.management_endpoints.key_management_endpoints import (
    _list_key_helper,
    _handle_soft_budget_update,
)
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_list_keys():
    mock_prisma_client = AsyncMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many
    args = {
        "prisma_client": mock_prisma_client,
        "page": 1,
        "size": 50,
        "user_id": "cda88cb4-cc2c-4e8c-b871-dc71ca111b00",
        "team_id": None,
        "organization_id": None,
        "key_alias": None,
        "key_hash": None,
        "exclude_team_id": None,
        "return_full_object": True,
        "admin_team_ids": ["28bd3181-02c5-48f2-b408-ce790fb3d5ba"],
    }
    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    mock_find_many.assert_called_once()

    where_condition = mock_find_many.call_args.kwargs["where"]
    print(f"where_condition: {where_condition}")
    assert json.dumps({"team_id": {"not": "litellm-dashboard"}}) in json.dumps(
        where_condition
    )


def test_generate_key_fn():
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import GenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )
    from litellm.proxy.utils import (
        hash_token,
    )  # Import the function you want to mock

    # Setup mocks
    mock_prisma_client = AsyncMock()
    mock_user_api_key_auth = MagicMock()
    data = GenerateKeyRequest(models=["model1", "model2"], duration="7d")

    # Use patch to mock hash_token
    with patch("litellm.proxy.utils.hash_token", return_value="mocked_hash"):
        # Call the function (you'll need to mock other dependencies as needed)
        # This is a simplified example - you'll need to properly mock all dependencies
        pass


def test_get_new_token_with_valid_existing_key():
    """Test get_new_token function when provided with a valid existing key"""
    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_new_token,
    )

    # Test with valid existing_key
    data = RegenerateKeyRequest(existing_key="sk-oldkey123456789")

    result = get_new_token(data)

    # Verify the result
    assert result is not None
    assert result.startswith("sk-")
    assert len(result) > 10  # Ensure it's a reasonable length


def test_get_new_token_with_valid_new_key():
    """Test get_new_token function when provided with a valid new key"""
    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_new_token,
    )

    # Test with valid new_key
    data = RegenerateKeyRequest(new_key="sk-test123456789")

    result = get_new_token(data)

    # Verify the result
    assert result == "sk-test123456789"


def test_get_new_token_with_invalid_key():
    """Test get_new_token function when provided with an invalid key that doesn't start with 'sk-'"""
    from fastapi import HTTPException

    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_new_token,
    )

    # Test with invalid new_key (doesn't start with 'sk-')
    data = RegenerateKeyRequest(new_key="invalid-key-123")

    with pytest.raises(HTTPException) as exc_info:
        get_new_token(data)

    assert exc_info.value.status_code == 400
    assert "New key must start with 'sk-'" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_handle_soft_budget_update_with_existing_budget():
    """Test _handle_soft_budget_update when key has existing budget"""
    mock_prisma_client = AsyncMock()
    mock_existing_key = MagicMock(budget_id="existing-budget-123")
    mock_user_api_key_dict = MagicMock(user_id="user-123")
    
    # Test updating existing budget
    result = await _handle_soft_budget_update(
        soft_budget=100.0,
        existing_key_row=mock_existing_key,
        prisma_client=mock_prisma_client,
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by="admin-user",
    )
    
    # Verify update was called
    mock_prisma_client.db.litellm_budgettable.update.assert_called_once_with(
        where={"budget_id": "existing-budget-123"},
        data={"soft_budget": 100.0},
    )
    
    # Should return None when updating existing budget
    assert result is None


@pytest.mark.asyncio
async def test_handle_soft_budget_update_create_new_budget():
    """Test _handle_soft_budget_update when creating new budget"""
    mock_prisma_client = AsyncMock()
    mock_existing_key = MagicMock(budget_id=None)
    mock_user_api_key_dict = MagicMock(user_id="user-123")
    
    # Mock the budget creation
    mock_new_budget = MagicMock(budget_id="new-budget-456")
    mock_prisma_client.db.litellm_budgettable.create.return_value = mock_new_budget
    
    # Mock jsonify_object
    mock_prisma_client.jsonify_object = MagicMock(return_value={
        "soft_budget": 50.0,
        "model_max_budget": {},
    })
    
    # Test creating new budget
    result = await _handle_soft_budget_update(
        soft_budget=50.0,
        existing_key_row=mock_existing_key,
        prisma_client=mock_prisma_client,
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by=None,
    )
    
    # Verify create was called with correct data
    create_call_args = mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    assert create_call_args["soft_budget"] == 50.0
    assert create_call_args["created_by"] == "user-123"
    assert create_call_args["updated_by"] == "user-123"
    
    # Should return new budget_id
    assert result == "new-budget-456"


@pytest.mark.asyncio
async def test_handle_soft_budget_update_with_none_budget():
    """Test _handle_soft_budget_update when soft_budget is None"""
    mock_prisma_client = AsyncMock()
    mock_existing_key = MagicMock(budget_id="existing-budget-123")
    mock_user_api_key_dict = MagicMock(user_id="user-123")
    
    # Test with None soft_budget
    result = await _handle_soft_budget_update(
        soft_budget=None,
        existing_key_row=mock_existing_key,
        prisma_client=mock_prisma_client,
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by="admin-user",
    )
    
    # Should not call any database operations
    mock_prisma_client.db.litellm_budgettable.update.assert_not_called()
    mock_prisma_client.db.litellm_budgettable.create.assert_not_called()
    
    # Should return None
    assert result is None


@pytest.mark.asyncio
async def test_handle_soft_budget_update_fallback_user_ids():
    """Test _handle_soft_budget_update with different user_id fallbacks"""
    mock_prisma_client = AsyncMock()
    mock_existing_key = MagicMock(budget_id=None)
    mock_user_api_key_dict = MagicMock(user_id=None)
    
    # Mock the budget creation
    mock_new_budget = MagicMock(budget_id="new-budget-789")
    mock_prisma_client.db.litellm_budgettable.create.return_value = mock_new_budget
    
    # Mock jsonify_object
    mock_prisma_client.jsonify_object = MagicMock(return_value={
        "soft_budget": 25.0,
        "model_max_budget": {},
    })
    
    # Test with litellm_changed_by fallback
    result = await _handle_soft_budget_update(
        soft_budget=25.0,
        existing_key_row=mock_existing_key,
        prisma_client=mock_prisma_client,
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by="changed-by-user",
    )
    
    # Verify fallback to litellm_changed_by
    create_call_args = mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    assert create_call_args["created_by"] == "changed-by-user"
    assert create_call_args["updated_by"] == "changed-by-user"
    
    # Reset mocks
    mock_prisma_client.reset_mock()
    mock_prisma_client.db.litellm_budgettable.create.return_value = mock_new_budget
    mock_prisma_client.jsonify_object.return_value = {
        "soft_budget": 25.0,
        "model_max_budget": {},
    }
    
    # Test with fallback to "admin"
    result = await _handle_soft_budget_update(
        soft_budget=25.0,
        existing_key_row=mock_existing_key,
        prisma_client=mock_prisma_client,
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by=None,
    )
    
    # Verify fallback to "admin"
    create_call_args = mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    assert create_call_args["created_by"] == "admin"
    assert create_call_args["updated_by"] == "admin"
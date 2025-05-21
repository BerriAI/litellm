import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from datetime import datetime, timedelta

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    SSOUserDefinedValues,
)
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken, get_user_object
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.utils import get_utc_datetime


@pytest.fixture(autouse=True)
def set_salt_key(monkeypatch):
    """Automatically set LITELLM_SALT_KEY for all tests"""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")


@pytest.fixture
def valid_sso_user_defined_values():
    return LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        models=["gpt-3.5-turbo"],
        max_budget=100.0,
    )


@pytest.fixture
def invalid_sso_user_defined_values():
    return LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=None,  # Missing user role
        models=["gpt-3.5-turbo"],
        max_budget=100.0,
    )


def test_get_experimental_ui_login_jwt_auth_token_valid(valid_sso_user_defined_values):
    """Test generating JWT token with valid user role"""
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )

    # Decrypt and verify token contents
    decrypted_token = decrypt_value_helper(token, exception_type="debug")
    # Check that decrypted_token is not None before using json.loads
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)

    assert token_data["user_id"] == "test_user"
    assert token_data["user_role"] == LitellmUserRoles.PROXY_ADMIN.value
    assert token_data["models"] == ["gpt-3.5-turbo"]
    assert token_data["max_budget"] == litellm.max_ui_session_budget

    # Verify expiration time is set and valid
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    assert expires > get_utc_datetime()
    assert expires <= get_utc_datetime() + timedelta(minutes=10)


def test_get_experimental_ui_login_jwt_auth_token_invalid(
    invalid_sso_user_defined_values,
):
    """Test generating JWT token with missing user role"""
    with pytest.raises(Exception) as exc_info:
        ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
            invalid_sso_user_defined_values
        )

    assert str(exc_info.value) == "User role is required for experimental UI login"


def test_get_key_object_from_ui_hash_key_valid(
    valid_sso_user_defined_values, monkeypatch
):
    """Test getting key object from valid UI hash key"""
    monkeypatch.setenv("EXPERIMENTAL_UI_LOGIN", "True")
    # Generate a valid token
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )

    # Get key object
    key_object = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(token)

    assert key_object is not None
    assert key_object.user_id == "test_user"
    assert key_object.user_role == LitellmUserRoles.PROXY_ADMIN
    assert key_object.models == ["gpt-3.5-turbo"]
    assert key_object.max_budget == litellm.max_ui_session_budget


def test_get_key_object_from_ui_hash_key_invalid():
    """Test getting key object from invalid UI hash key"""
    # Test with invalid token
    key_object = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key("invalid_token")
    assert key_object is None


@pytest.mark.asyncio
async def test_get_object_from_cache_redis():
    """Test that _get_object_from_cache retrieves from Redis cache when available"""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_VerificationToken
    from litellm.proxy.auth.auth_checks import _get_object_from_cache

    # Create mock objects
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.internal_usage_cache.dual_cache = AsyncMock()
    mock_user_api_key_cache = DualCache()

    # Create test data
    test_key = "test_key"
    test_data = {
        "token": "test_token",
        "key_name": "test_key_name",
        "spend": 0.0,
        "models": ["gpt-3.5-turbo"],
    }

    # Mock Redis cache response
    mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache.return_value = (
        test_data
    )

    # Call the function
    result = await _get_object_from_cache(
        key=test_key,
        proxy_logging_obj=mock_proxy_logging,
        user_api_key_cache=mock_user_api_key_cache,
        parent_otel_span=None,
        base_model=LiteLLM_VerificationToken,
    )

    # Verify Redis cache was checked
    mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache.assert_called_once_with(
        key=test_key, parent_otel_span=None
    )

    # Verify result is correct
    assert isinstance(result, LiteLLM_VerificationToken)
    assert result.token == "test_token"
    assert result.key_name == "test_key_name"
    assert result.spend == 0.0
    assert result.models == ["gpt-3.5-turbo"]

async def test_default_internal_user_params_with_get_user_object(monkeypatch):
    """Test that default_internal_user_params is used when creating a new user via get_user_object"""
    # Set up default_internal_user_params
    default_params = {
        "models": ["gpt-4", "claude-3-opus"],
        "max_budget": 200.0,
        "user_role": "internal_user",
    }
    monkeypatch.setattr(litellm, "default_internal_user_params", default_params)

    # Mock the necessary dependencies
    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db

    # Set up the user creation mock - create a complete user model that can be converted to a dict
    mock_user = MagicMock()
    mock_user.user_id = "new_test_user"
    mock_user.models = ["gpt-4", "claude-3-opus"]
    mock_user.max_budget = 200.0
    mock_user.user_role = "internal_user"
    mock_user.organization_memberships = []

    # Make the mock model_dump or dict method return appropriate data
    mock_user.dict = lambda: {
        "user_id": "new_test_user",
        "models": ["gpt-4", "claude-3-opus"],
        "max_budget": 200.0,
        "user_role": "internal_user",
        "organization_memberships": [],
    }

    # Setup the mock returns
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.create = AsyncMock(return_value=mock_user)

    # Create a mock cache - use AsyncMock for async methods
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    # Call get_user_object with user_id_upsert=True to trigger user creation
    try:
        user_obj = await get_user_object(
            user_id="new_test_user",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            user_id_upsert=True,
            proxy_logging_obj=None,
        )
    except Exception as e:
        # this fails since the mock object is a MagicMock and not a LiteLLM_UserTable
        print(e)

    # Verify the user was created with the default params
    mock_prisma_client.db.litellm_usertable.create.assert_called_once()
    creation_args = mock_prisma_client.db.litellm_usertable.create.call_args[1]["data"]

    # Verify defaults were applied to the creation args
    assert "models" in creation_args
    assert creation_args["models"] == ["gpt-4", "claude-3-opus"]
    assert creation_args["max_budget"] == 200.0
    assert creation_args["user_role"] == "internal_user"

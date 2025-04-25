import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from datetime import UTC, datetime, timedelta

import pytest

import litellm
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    SSOUserDefinedValues,
)
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper


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
    token_data = json.loads(decrypted_token)

    assert token_data["user_id"] == "test_user"
    assert token_data["user_role"] == LitellmUserRoles.PROXY_ADMIN.value
    assert token_data["models"] == ["gpt-3.5-turbo"]
    assert token_data["max_budget"] == litellm.max_ui_session_budget

    # Verify expiration time is set and valid
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    assert expires > datetime.now(UTC)
    assert expires <= datetime.now(UTC) + timedelta(minutes=10)


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

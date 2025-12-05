"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    hash_token,
)
from litellm.proxy.auth.login_utils import (
    LoginResult,
    authenticate_user,
    get_ui_credentials,
)


def test_get_ui_credentials_prefers_explicit_password():
    """The configured UI password should be returned when available."""
    with patch.dict(
        os.environ,
        {"UI_USERNAME": "test-admin", "UI_PASSWORD": "secure-pass"},
        clear=True,
    ):
        username, password = get_ui_credentials(master_key="sk-123")

    assert username == "test-admin"
    assert password == "secure-pass"


def test_get_ui_credentials_can_use_master_key():
    """Master key should be used as password when UI_PASSWORD is missing."""
    with patch.dict(os.environ, {"UI_USERNAME": "fallback-admin"}, clear=True):
        username, password = get_ui_credentials(master_key="fallback-key")

    assert username == "fallback-admin"
    assert password == "fallback-key"


def test_get_ui_credentials_requires_password():
    """Missing UI password and master key results in error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ProxyException) as exc_info:
            get_ui_credentials(master_key=None)

    assert exc_info.value.type == ProxyErrorTypes.auth_error
    assert exc_info.value.code == "500"


@pytest.mark.asyncio
async def test_authenticate_user_admin_login_with_ui_credentials():
    """Test admin login using UI_USERNAME and UI_PASSWORD"""
    master_key = "sk-1234"
    ui_username = "admin"
    ui_password = "sk-1234"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    with patch.dict(
        os.environ,
        {
            "UI_USERNAME": ui_username,
            "UI_PASSWORD": ui_password,
            "DATABASE_URL": "postgresql://test:test@localhost/test",
        },
    ):
        with patch(
            "litellm.proxy.auth.login_utils.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_generate_key:
            mock_generate_key.return_value = {
                "token": "test-token-123",
                "user_id": LITELLM_PROXY_ADMIN_NAME,
            }

            with patch(
                "litellm.proxy.auth.login_utils.user_update",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_user_update:
                with patch(
                    "litellm.proxy.auth.login_utils.get_secret_bool",
                    return_value=False,
                ):
                    result = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                    )

                    assert isinstance(result, LoginResult)
                    assert result.user_id == LITELLM_PROXY_ADMIN_NAME
                    assert result.key == "test-token-123"
                    assert result.user_email is None
                    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
                    assert result.login_method == "username_password"


@pytest.mark.asyncio
async def test_authenticate_user_admin_login_with_master_key_as_password():
    """Test admin login when UI_PASSWORD is not set, should use master_key"""
    master_key = "sk-1234"
    ui_username = "admin"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    env_vars = {"UI_USERNAME": ui_username, "DATABASE_URL": "postgresql://test:test@localhost/test"}
    # Remove UI_PASSWORD to test fallback to master_key
    if "UI_PASSWORD" in os.environ:
        # Keep other env vars but don't set UI_PASSWORD
        pass
    else:
        # Ensure UI_PASSWORD is not in the patched env
        pass

    with patch.dict(os.environ, env_vars, clear=False):
        # Explicitly remove UI_PASSWORD if it exists
        original_ui_password = os.environ.pop("UI_PASSWORD", None)
        try:
            with patch(
                "litellm.proxy.auth.login_utils.generate_key_helper_fn",
                new_callable=AsyncMock,
            ) as mock_generate_key:
                mock_generate_key.return_value = {
                    "token": "test-token-123",
                    "user_id": LITELLM_PROXY_ADMIN_NAME,
                }

                with patch(
                    "litellm.proxy.auth.login_utils.user_update",
                    new_callable=AsyncMock,
                    return_value=None,
                ) as mock_user_update:
                    with patch(
                        "litellm.proxy.auth.login_utils.get_secret_bool",
                        return_value=False,
                    ):
                        result = await authenticate_user(
                            username=ui_username,
                            password=master_key,
                            master_key=master_key,
                            prisma_client=mock_prisma_client,
                        )

                        assert isinstance(result, LoginResult)
                        assert result.user_id == LITELLM_PROXY_ADMIN_NAME
                        assert result.user_role == LitellmUserRoles.PROXY_ADMIN
        finally:
            if original_ui_password:
                os.environ["UI_PASSWORD"] = original_ui_password

@pytest.mark.asyncio
async def test_authenticate_user_invalid_credentials():
    """Test authentication failure with invalid credentials"""
    master_key = "sk-1234"
    ui_username = "admin"
    wrong_password = "wrong-password"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    with patch.dict(os.environ, {"UI_USERNAME": ui_username, "UI_PASSWORD": "correct-password"}):
        with pytest.raises(ProxyException) as exc_info:
            await authenticate_user(
                username=ui_username,
                password=wrong_password,
                master_key=master_key,
                prisma_client=mock_prisma_client,
            )

        assert exc_info.value.type == ProxyErrorTypes.auth_error
        assert exc_info.value.code == "401"
        assert "Invalid credentials" in exc_info.value.message


@pytest.mark.asyncio
async def test_authenticate_user_missing_master_key():
    """Test authentication failure when master_key is None"""
    mock_prisma_client = MagicMock()

    with pytest.raises(ProxyException) as exc_info:
        await authenticate_user(
            username="admin",
            password="password",
            master_key=None,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.type == ProxyErrorTypes.auth_error
    assert exc_info.value.code == "500"
    assert "Master Key not set" in exc_info.value.message


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password():
    """Test authentication failure with wrong password for database user"""
    master_key = "sk-1234"
    user_email = "test@example.com"
    correct_password = "correct-password"
    wrong_password = "wrong-password"
    hashed_password = hash_token(token=correct_password)

    mock_user = LiteLLM_UserTable(
        user_id="test-user-123",
        user_email=user_email,
        password=hashed_password,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(
        return_value=mock_user
    )

    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "admin-password",
        },
    ):
        with pytest.raises(ProxyException) as exc_info:
            await authenticate_user(
                username=user_email,
                password=wrong_password,
                master_key=master_key,
                prisma_client=mock_prisma_client,
            )

        assert exc_info.value.type == ProxyErrorTypes.auth_error
        assert exc_info.value.code == "401"
        assert "Invalid credentials" in exc_info.value.message


@pytest.mark.asyncio
async def test_authenticate_user_database_required_for_admin():
    """Test that database is required for admin login"""
    master_key = "sk-1234"
    ui_username = "admin"
    ui_password = "sk-1234"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    with patch.dict(os.environ, {"UI_USERNAME": ui_username, "UI_PASSWORD": ui_password}):
        with patch(
            "litellm.proxy.auth.login_utils.user_update",
            new_callable=AsyncMock,
            return_value=None,
        ):
            # Remove DATABASE_URL to simulate no database
            original_db_url = os.environ.get("DATABASE_URL")
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]

            try:
                with pytest.raises(ProxyException) as exc_info:
                    await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                    )

                assert exc_info.value.type == ProxyErrorTypes.auth_error
                assert exc_info.value.code == "500"
                assert "No Database connected" in exc_info.value.message
            finally:
                if original_db_url:
                    os.environ["DATABASE_URL"] = original_db_url

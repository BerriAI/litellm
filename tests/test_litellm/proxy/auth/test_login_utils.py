"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import os
from datetime import datetime, timezone, timedelta
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
    expire_previous_ui_session_tokens,
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
async def test_authenticate_user_email_case_insensitive_login():
    """Test that email lookup is case-insensitive during login"""
    master_key = "sk-1234"
    stored_email = "testemail@test.com"
    login_email_mixed_case = "testEmail@test.com"
    correct_password = "correct-password"
    hashed_password = hash_token(token=correct_password)

    # `LiteLLM_UserTable` does not define a `password` field, but `authenticate_user()`
    # expects `user_row.password` to exist (invite-link login). Use a simple object.
    mock_user = MagicMock()
    mock_user.user_id = "test-user-123"
    mock_user.user_email = stored_email
    mock_user.password = hashed_password
    mock_user.user_role = LitellmUserRoles.INTERNAL_USER

    def mock_find_first(**kwargs):
        where = kwargs.get("where", {})
        user_email = where.get("user_email", {})
        if user_email.get("mode") != "insensitive":
            return None
        if str(user_email.get("equals", "")).lower() == stored_email.lower():
            return mock_user
        return None

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(
        side_effect=mock_find_first
    )

    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "admin-password",
        },
    ):
        with patch(
            "litellm.proxy.auth.login_utils.expire_previous_ui_session_tokens",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "litellm.proxy.auth.login_utils.generate_key_helper_fn",
                new_callable=AsyncMock,
            ) as mock_generate_key:
                mock_generate_key.side_effect = [
                    {"token": "token-1"},
                    {"token": "token-2"},
                ]

                result_mixed = await authenticate_user(
                    username=login_email_mixed_case,
                    password=correct_password,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                )
                result_lower = await authenticate_user(
                    username=stored_email,
                    password=correct_password,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                )

    assert result_mixed.user_id == result_lower.user_id == "test-user-123"
    assert result_mixed.user_email == result_lower.user_email == stored_email

    calls = mock_prisma_client.db.litellm_usertable.find_first.await_args_list
    assert len(calls) == 2
    for call, expected_username in zip(calls, [login_email_mixed_case, stored_email]):
        where = call.kwargs["where"]
        assert where["user_email"]["equals"] == expected_username
        assert where["user_email"]["mode"] == "insensitive"


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


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_none_prisma_client():
    """Test that function returns early when prisma_client is None"""
    await expire_previous_ui_session_tokens("test-user", None)
    # Should not raise any exception


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_only_litellm_dashboard_team():
    """Test that only tokens with team_id='litellm-dashboard' are expired"""
    user_id = "test-user"
    current_time = datetime.now(timezone.utc)

    # Create mock tokens with proper attributes
    token1 = MagicMock()
    token1.token = "token1"
    token1.user_id = user_id
    token1.team_id = "litellm-dashboard"
    token1.blocked = None
    token1.expires = current_time + timedelta(hours=1)

    token2 = MagicMock()
    token2.token = "token2"
    token2.user_id = user_id
    token2.team_id = "other-team"
    token2.blocked = None
    token2.expires = current_time + timedelta(hours=1)

    def mock_find_many(**kwargs):
        """Mock find_many that filters tokens based on query criteria"""
        where_clause = kwargs.get("where", {})
        filtered_tokens = []

        for token in [token1, token2]:
            # Check user_id match
            if token.user_id != where_clause.get("user_id"):
                continue
            # Check team_id match
            if token.team_id != where_clause.get("team_id"):
                continue
            # Check blocked condition (None or False)
            if token.blocked is not None and token.blocked is not False:
                continue
            # Check expires > current_time
            if token.expires <= where_clause.get("expires", {}).get("gt"):
                continue
            filtered_tokens.append(token)

        return filtered_tokens

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=mock_find_many)
    mock_prisma_client.db.litellm_verificationtoken.update_many = AsyncMock()

    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

    # Should only call update_many with the litellm-dashboard token
    mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once_with(
        where={"token": {"in": ["token1"]}},
        data={"blocked": True}
    )


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_blocks_null_and_false():
    """Test that tokens with blocked=None and blocked=False are both processed"""
    user_id = "test-user"
    current_time = datetime.now(timezone.utc)

    # Create mock tokens with proper attributes
    token1 = MagicMock()
    token1.token = "token1"
    token1.user_id = user_id
    token1.team_id = "litellm-dashboard"
    token1.blocked = None
    token1.expires = current_time + timedelta(hours=1)

    token2 = MagicMock()
    token2.token = "token2"
    token2.user_id = user_id
    token2.team_id = "litellm-dashboard"
    token2.blocked = False
    token2.expires = current_time + timedelta(hours=1)

    token3 = MagicMock()
    token3.token = "token3"
    token3.user_id = user_id
    token3.team_id = "litellm-dashboard"
    token3.blocked = True  # This should be ignored
    token3.expires = current_time + timedelta(hours=1)

    def mock_find_many(**kwargs):
        """Mock find_many that filters tokens based on query criteria"""
        where_clause = kwargs.get("where", {})
        filtered_tokens = []

        for token in [token1, token2, token3]:
            # Check user_id match
            if token.user_id != where_clause.get("user_id"):
                continue
            # Check team_id match
            if token.team_id != where_clause.get("team_id"):
                continue
            # Check blocked condition (None or False)
            if token.blocked is not None and token.blocked is not False:
                continue
            # Check expires > current_time
            if token.expires <= where_clause.get("expires", {}).get("gt"):
                continue
            filtered_tokens.append(token)

        return filtered_tokens

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=mock_find_many)
    mock_prisma_client.db.litellm_verificationtoken.update_many = AsyncMock()

    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

    # Should only block token1 and token2 (not token3 which is already blocked)
    mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once_with(
        where={"token": {"in": ["token1", "token2"]}},
        data={"blocked": True}
    )


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_only_non_expired():
    """Test that only non-expired tokens are processed"""
    user_id = "test-user"
    current_time = datetime.now(timezone.utc)

    # Create mock tokens with proper attributes
    token1 = MagicMock()
    token1.token = "token1"
    token1.user_id = user_id
    token1.team_id = "litellm-dashboard"
    token1.blocked = None
    token1.expires = current_time + timedelta(hours=1)  # Not expired

    token2 = MagicMock()
    token2.token = "token2"
    token2.user_id = user_id
    token2.team_id = "litellm-dashboard"
    token2.blocked = None
    token2.expires = current_time - timedelta(hours=1)  # Already expired

    def mock_find_many(**kwargs):
        """Mock find_many that filters tokens based on query criteria"""
        where_clause = kwargs.get("where", {})
        filtered_tokens = []

        for token in [token1, token2]:
            # Check user_id match
            if token.user_id != where_clause.get("user_id"):
                continue
            # Check team_id match
            if token.team_id != where_clause.get("team_id"):
                continue
            # Check blocked condition (None or False)
            if token.blocked is not None and token.blocked is not False:
                continue
            # Check expires > current_time
            if token.expires <= where_clause.get("expires", {}).get("gt"):
                continue
            filtered_tokens.append(token)

        return filtered_tokens

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=mock_find_many)
    mock_prisma_client.db.litellm_verificationtoken.update_many = AsyncMock()

    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

    # Should only block the non-expired token
    mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once_with(
        where={"token": {"in": ["token1"]}},
        data={"blocked": True}
    )


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_no_tokens_found():
    """Test behavior when no valid tokens are found"""
    user_id = "test-user"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_verificationtoken.update_many = AsyncMock()

    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

    # Should not call update_many when no tokens found
    mock_prisma_client.db.litellm_verificationtoken.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_filters_none_token():
    """Test that tokens with None token value are filtered out"""
    user_id = "test-user"
    current_time = datetime.now(timezone.utc)

    # Create mock tokens with proper attributes
    token1 = MagicMock()
    token1.token = "token1"
    token1.user_id = user_id
    token1.team_id = "litellm-dashboard"
    token1.blocked = None
    token1.expires = current_time + timedelta(hours=1)

    token2 = MagicMock()
    token2.token = None  # This should be filtered out in the token collection step
    token2.user_id = user_id
    token2.team_id = "litellm-dashboard"
    token2.blocked = None
    token2.expires = current_time + timedelta(hours=1)

    def mock_find_many(**kwargs):
        """Mock find_many that filters tokens based on query criteria"""
        where_clause = kwargs.get("where", {})
        filtered_tokens = []

        for token in [token1, token2]:
            # Check user_id match
            if token.user_id != where_clause.get("user_id"):
                continue
            # Check team_id match
            if token.team_id != where_clause.get("team_id"):
                continue
            # Check blocked condition (None or False)
            if token.blocked is not None and token.blocked is not False:
                continue
            # Check expires > current_time
            if token.expires <= where_clause.get("expires", {}).get("gt"):
                continue
            filtered_tokens.append(token)

        return filtered_tokens

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=mock_find_many)
    mock_prisma_client.db.litellm_verificationtoken.update_many = AsyncMock()

    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

    # Should only block token1 (token with None value should be filtered out)
    mock_prisma_client.db.litellm_verificationtoken.update_many.assert_called_once_with(
        where={"token": {"in": ["token1"]}},
        data={"blocked": True}
    )


@pytest.mark.asyncio
async def test_expire_previous_ui_session_tokens_exception_handling():
    """Test that exceptions during token expiry are silently handled"""
    user_id = "test-user"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(side_effect=Exception("Database error"))

    # Should not raise exception despite database error
    await expire_previous_ui_session_tokens(user_id, mock_prisma_client)

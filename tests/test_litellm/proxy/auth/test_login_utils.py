"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import hashlib
import os
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
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
    is_env_credential_login_enabled,
    screen_login_password_for_breach,
)

# Successful DB-user logins schedule the background HIBP screen; disable it so
# no test ever does live network I/O to haveibeenpwned.com from CI.
_POLICY_NO_BREACH_CHECK = {"password_policy_check_breached_passwords": False}


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
async def test_authenticate_user_admin_login_with_master_key_as_password(monkeypatch):
    """Test admin login when UI_PASSWORD is not set, should use master_key"""
    master_key = "sk-1234"
    ui_username = "admin"

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    env_vars = {
        "UI_USERNAME": ui_username,
        "DATABASE_URL": "postgresql://test:test@localhost/test",
    }
    # Remove UI_PASSWORD to test fallback to master_key
    if "UI_PASSWORD" in os.environ:
        # Keep other env vars but don't set UI_PASSWORD
        pass
    else:
        # Ensure UI_PASSWORD is not in the patched env
        pass

    with patch.dict(os.environ, env_vars, clear=False):
        # Explicitly remove UI_PASSWORD if it exists
        monkeypatch.delenv("UI_PASSWORD", raising=False)
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
        assert "UI_USERNAME" in exc_info.value.message


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
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=mock_user)

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
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(side_effect=mock_find_first)

    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "admin-password",
        },
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
                general_settings=_POLICY_NO_BREACH_CHECK,
            )
            result_lower = await authenticate_user(
                username=stored_email,
                password=correct_password,
                master_key=master_key,
                prisma_client=mock_prisma_client,
                general_settings=_POLICY_NO_BREACH_CHECK,
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
async def test_authenticate_user_database_required_for_admin(monkeypatch):
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
                    monkeypatch.setenv("DATABASE_URL", original_db_url)


@pytest.mark.asyncio
async def test_authenticate_user_admin_login_with_non_ascii_characters():
    """Test admin login with non-ASCII characters in password (issue #19559)"""
    master_key = "sk-1234"
    ui_username = "admin£test"
    ui_password = "sk-1234£pass"

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
                    assert result.user_role == LitellmUserRoles.PROXY_ADMIN


def test_authenticate_user_non_ascii_direct_comparison():
    """Test that non-ASCII characters can be compared directly (unit test for fix)"""
    import secrets

    # This test verifies the fix handles non-ASCII by encoding to bytes
    username = "admin£test"
    password = "pass£word"

    # This would fail without encoding:
    # secrets.compare_digest(username, username)  # TypeError!

    # But works with the fix:
    result = secrets.compare_digest(username.encode("utf-8"), username.encode("utf-8"))
    assert result is True

    # And correctly returns False for different passwords
    result = secrets.compare_digest(password.encode("utf-8"), "different£pass".encode("utf-8"))
    assert result is False


@pytest.mark.asyncio
async def test_authenticate_user_multiple_logins_generate_unique_tokens():
    """Test that multiple logins for the same user each generate unique tokens.

    This test verifies that users can have multiple concurrent UI sessions.
    Previous UI session tokens should NOT be expired/blocked when a new session is created.
    """
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
            # Each login should generate a unique token
            mock_generate_key.side_effect = [
                {"token": "session-token-1", "user_id": LITELLM_PROXY_ADMIN_NAME},
                {"token": "session-token-2", "user_id": LITELLM_PROXY_ADMIN_NAME},
                {"token": "session-token-3", "user_id": LITELLM_PROXY_ADMIN_NAME},
            ]

            with patch(
                "litellm.proxy.auth.login_utils.user_update",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with patch(
                    "litellm.proxy.auth.login_utils.get_secret_bool",
                    return_value=False,
                ):
                    # Simulate multiple logins from the same user
                    result1 = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                    )
                    result2 = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                    )
                    result3 = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                    )

                    # Each login should return a unique token
                    assert result1.key == "session-token-1"
                    assert result2.key == "session-token-2"
                    assert result3.key == "session-token-3"

                    # All tokens should be different (concurrent sessions allowed)
                    assert len({result1.key, result2.key, result3.key}) == 3

                    # generate_key_helper_fn should be called 3 times (once per login)
                    assert mock_generate_key.call_count == 3


@pytest.mark.asyncio
async def test_authenticate_user_database_login_with_non_ascii_password():
    """Test database user login with non-ASCII characters in password (issue #19559)"""
    master_key = "sk-1234"
    user_email = "test@example.com"
    password_with_special_char = "correct£password"
    hashed_password = hash_token(token=password_with_special_char)

    mock_user = MagicMock()
    mock_user.user_id = "test-user-123"
    mock_user.user_email = user_email
    mock_user.password = hashed_password
    mock_user.user_role = LitellmUserRoles.INTERNAL_USER

    def mock_find_first(**kwargs):
        where = kwargs.get("where", {})
        user_email_filter = where.get("user_email", {})
        if str(user_email_filter.get("equals", "")).lower() == user_email.lower():
            return mock_user
        return None

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(side_effect=mock_find_first)

    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "admin-password",
        },
    ):
        with patch(
            "litellm.proxy.auth.login_utils.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_generate_key:
            mock_generate_key.return_value = {"token": "token-123"}

            result = await authenticate_user(
                username=user_email,
                password=password_with_special_char,
                master_key=master_key,
                prisma_client=mock_prisma_client,
                general_settings=_POLICY_NO_BREACH_CHECK,
            )

            assert isinstance(result, LoginResult)
            assert result.user_id == "test-user-123"
            assert result.user_email == user_email


class TestEncodeUiSessionJwt:
    """The UI session cookie must carry a bounded exp so it does not stay
    signature-valid until the master key rotates, and so the session-cookie readers
    that require a bounded lifetime (the MCP interactive sign-in) accept it."""

    def _decode(self, token: str) -> dict:
        import jwt

        return jwt.decode(token, "sk-master-for-tests", algorithms=["HS256"])

    def test_encoded_cookie_carries_bounded_exp(self):
        import time

        from litellm.proxy.auth.login_utils import encode_ui_session_jwt

        token_object = {"user_id": "u1", "key": "sk-abc", "login_method": "username_password"}
        with patch("litellm.proxy.auth.login_utils.LITELLM_UI_SESSION_DURATION", "24h"):
            token = encode_ui_session_jwt(token_object, "sk-master-for-tests")
        claims = self._decode(token)
        assert claims["user_id"] == "u1"
        assert claims["login_method"] == "username_password"
        remaining = claims["exp"] - int(time.time())
        assert 23 * 3600 < remaining <= 24 * 3600

    def test_duration_is_honored_from_env(self):
        import time

        from litellm.proxy.auth.login_utils import encode_ui_session_jwt

        with patch("litellm.proxy.auth.login_utils.LITELLM_UI_SESSION_DURATION", "1h"):
            token = encode_ui_session_jwt({"user_id": "u1"}, "sk-master-for-tests")
        remaining = self._decode(token)["exp"] - int(time.time())
        assert 0 < remaining <= 3600

    def test_cookie_is_accepted_by_the_exp_requiring_session_reader(self):
        """The regression this change exists for: before it, the UI cookie carried no
        exp and _user_id_from_session_cookie (require=["exp"]) rejected every real login,
        so the MCP interactive sign-in could never capture identity. A cookie minted by
        this helper must now be accepted."""
        from unittest.mock import MagicMock

        from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
            _user_id_from_session_cookie,
        )
        from litellm.proxy.auth.login_utils import encode_ui_session_jwt

        token_object = {"user_id": "cornell-user", "key": "sk-abc", "login_method": "sso"}
        with patch("litellm.proxy.auth.login_utils.LITELLM_UI_SESSION_DURATION", "24h"):
            token = encode_ui_session_jwt(token_object, "sk-master-for-tests")
        request = MagicMock()
        request.cookies = {"token": token}
        with patch("litellm.proxy.proxy_server.master_key", "sk-master-for-tests"):
            assert _user_id_from_session_cookie(request) == "cornell-user"


def _patch_sso_configured(stack: ExitStack, *, configured: bool) -> None:
    stack.enter_context(
        patch(  # test-quality-ok: no HTTP boundary here; same internal the pre-existing tests above already mock
            "litellm.proxy.auth.login_utils.is_sso_provider_fully_configured", return_value=configured
        )
    )


def _patch_successful_admin_login_deps(stack: ExitStack) -> None:
    """The collaborators a real admin login exercises past the SSO gate:
    generating the UI session key, syncing the admin role, and reading the
    experimental-login flag. Shared so the two "still allowed" tests below
    don't each repeat the same three-mock wiring."""
    stack.enter_context(
        patch(  # test-quality-ok: internal orchestration, no HTTP boundary; matches pre-existing tests
            "litellm.proxy.auth.login_utils.generate_key_helper_fn",
            new_callable=AsyncMock,
            return_value={"token": "test-token", "user_id": LITELLM_PROXY_ADMIN_NAME},
        )
    )
    stack.enter_context(
        patch(  # test-quality-ok: internal orchestration, no HTTP boundary; matches pre-existing tests
            "litellm.proxy.auth.login_utils.user_update",
            new_callable=AsyncMock,
            return_value=None,
        )
    )
    stack.enter_context(
        patch(  # test-quality-ok: internal orchestration, no HTTP boundary; matches pre-existing tests
            "litellm.proxy.auth.login_utils.get_secret_bool",
            return_value=False,
        )
    )


class TestDisablePasswordLoginWhenSSOEnabled:
    """`disable_password_login_when_sso_enabled` must reject every
    username/password login attempt (including the UI_USERNAME/UI_PASSWORD
    admin fallback) once SSO is configured, so SSO becomes the only way to
    reach the Admin UI. It must not affect logins when SSO is unconfigured,
    so admins can never lock themselves out with no SSO to fall back to."""

    @pytest.mark.asyncio
    async def test_rejects_correct_admin_credentials_when_sso_configured(self):
        master_key = "sk-1234"
        ui_username = "admin"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"UI_USERNAME": ui_username, "UI_PASSWORD": master_key}):
            with ExitStack() as stack:
                _patch_sso_configured(stack, configured=True)
                with pytest.raises(ProxyException) as exc_info:
                    await authenticate_user(
                        username=ui_username,
                        password=master_key,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                        general_settings={"disable_password_login_when_sso_enabled": True},
                    )

        assert exc_info.value.type == ProxyErrorTypes.auth_error
        assert exc_info.value.code == "403"
        # The credential comparison must never even run.
        mock_prisma_client.db.litellm_usertable.find_first.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_correct_db_user_credentials_when_sso_configured(self):
        master_key = "sk-1234"
        user_email = "test@example.com"
        password = "correct-password"

        mock_user = LiteLLM_UserTable(
            user_id="test-user-123",
            user_email=user_email,
            password=hash_token(token=password),
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=mock_user)

        with patch.dict(os.environ, {"UI_USERNAME": "admin", "UI_PASSWORD": "unrelated"}):
            with ExitStack() as stack:
                _patch_sso_configured(stack, configured=True)
                with pytest.raises(ProxyException) as exc_info:
                    await authenticate_user(
                        username=user_email,
                        password=password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                        general_settings={"disable_password_login_when_sso_enabled": True},
                    )

        assert exc_info.value.code == "403"
        mock_prisma_client.db.litellm_usertable.find_first.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_password_login_when_setting_enabled_but_sso_not_configured(self):
        """The setting alone must not lock out an admin who has not actually
        configured SSO — there would be no fallback left."""
        master_key = "sk-1234"
        ui_username = "admin"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {
                "UI_USERNAME": ui_username,
                "UI_PASSWORD": master_key,
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
            clear=True,
        ):
            with ExitStack() as stack:
                _patch_sso_configured(stack, configured=False)
                _patch_successful_admin_login_deps(stack)
                result = await authenticate_user(
                    username=ui_username,
                    password=master_key,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={"disable_password_login_when_sso_enabled": True},
                )

        assert isinstance(result, LoginResult)
        assert result.user_id == LITELLM_PROXY_ADMIN_NAME

    @pytest.mark.asyncio
    async def test_allows_password_login_when_sso_env_is_incomplete(self):
        """Regression: a lone MICROSOFT_CLIENT_ID with no client secret or
        tenant makes has_user_setup_sso() True, but a real SSO sign-in would
        fail. The gate must read the real env (no is_sso_provider_fully_configured
        mock here) and still let password login through, or an admin who set
        one env var by mistake is locked out with no way in."""
        master_key = "sk-1234"
        ui_username = "admin"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {
                "UI_USERNAME": ui_username,
                "UI_PASSWORD": master_key,
                "DATABASE_URL": "postgresql://test:test@localhost/test",
                "MICROSOFT_CLIENT_ID": "ms-client-id-only",
            },
            clear=True,
        ):
            with ExitStack() as stack:
                _patch_successful_admin_login_deps(stack)
                result = await authenticate_user(
                    username=ui_username,
                    password=master_key,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={"disable_password_login_when_sso_enabled": True},
                )

        assert isinstance(result, LoginResult)
        assert result.user_id == LITELLM_PROXY_ADMIN_NAME

    @pytest.mark.asyncio
    async def test_allows_password_login_when_sso_configured_but_setting_not_enabled(self):
        """SSO being configured must not, by itself, disable the password
        fallback: the setting is opt-in."""
        master_key = "sk-1234"
        ui_username = "admin"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {
                "UI_USERNAME": ui_username,
                "UI_PASSWORD": master_key,
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
            clear=True,
        ):
            with ExitStack() as stack:
                _patch_sso_configured(stack, configured=True)
                _patch_successful_admin_login_deps(stack)
                result = await authenticate_user(
                    username=ui_username,
                    password=master_key,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={},
                )

        assert isinstance(result, LoginResult)
        assert result.user_id == LITELLM_PROXY_ADMIN_NAME


class TestDisableEnvCredentialLogin:
    """`disable_env_credential_login` must reject a login with the env
    credentials (UI_USERNAME/UI_PASSWORD, or the master-key fallback when
    UI_PASSWORD is unset) while leaving database-user password logins
    untouched, so admins with real accounts keep a way in."""

    @pytest.mark.asyncio
    async def test_rejects_correct_env_credentials_when_disabled(self):
        master_key = "sk-1234"
        ui_username = "admin"
        ui_password = "env-only-password"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"UI_USERNAME": ui_username, "UI_PASSWORD": ui_password}):
            with pytest.raises(ProxyException) as exc_info:
                await authenticate_user(
                    username=ui_username,
                    password=ui_password,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={"disable_env_credential_login": True},
                )

        assert exc_info.value.type == ProxyErrorTypes.auth_error
        assert exc_info.value.code == "401"
        assert "UI_USERNAME" not in exc_info.value.message
        assert "UI_PASSWORD" not in exc_info.value.message

    @pytest.mark.asyncio
    async def test_rejects_master_key_fallback_when_disabled(self):
        """With UI_PASSWORD unset, the master key IS the env password, so the
        setting must reject it too or it protects nothing by default."""
        master_key = "sk-1234"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"UI_USERNAME": "admin"}, clear=True):
            with pytest.raises(ProxyException) as exc_info:
                await authenticate_user(
                    username="admin",
                    password=master_key,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={"disable_env_credential_login": True},
                )

        assert exc_info.value.code == "401"

    @pytest.mark.asyncio
    async def test_db_user_login_still_works_when_disabled(self):
        master_key = "sk-1234"
        user_email = "admin@example.com"
        password = "Str0ng!Passw0rd"

        mock_user = LiteLLM_UserTable(
            user_id="db-admin-1",
            user_email=user_email,
            password=hash_token(token=password),
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=mock_user)

        with patch.dict(
            os.environ,
            {
                "UI_USERNAME": "admin",
                "UI_PASSWORD": "env-password",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
            clear=True,
        ):
            with ExitStack() as stack:
                stack.enter_context(
                    patch(  # test-quality-ok: internal orchestration, no HTTP boundary; matches pre-existing tests
                        "litellm.proxy.auth.login_utils.generate_key_helper_fn",
                        new_callable=AsyncMock,
                        return_value={"token": "db-user-token"},
                    )
                )
                result = await authenticate_user(
                    username=user_email,
                    password=password,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={"disable_env_credential_login": True},
                )

        assert isinstance(result, LoginResult)
        assert result.user_id == "db-admin-1"
        assert result.user_role == LitellmUserRoles.PROXY_ADMIN

    @pytest.mark.asyncio
    async def test_env_login_still_works_when_setting_absent(self):
        """Env-credential login is the bootstrap path on a fresh install and
        must stay on by default."""
        master_key = "sk-1234"
        ui_username = "admin"

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {
                "UI_USERNAME": ui_username,
                "UI_PASSWORD": master_key,
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
            clear=True,
        ):
            with ExitStack() as stack:
                _patch_successful_admin_login_deps(stack)
                result = await authenticate_user(
                    username=ui_username,
                    password=master_key,
                    master_key=master_key,
                    prisma_client=mock_prisma_client,
                    general_settings={},
                )

        assert isinstance(result, LoginResult)
        assert result.user_id == LITELLM_PROXY_ADMIN_NAME


class TestIsEnvCredentialLoginEnabled:
    """Drives the Admin UI warning banner: it must be True exactly when a
    login with the env credentials could actually succeed."""

    def test_enabled_by_default(self):
        assert is_env_credential_login_enabled({}) is True

    def test_disabled_by_dedicated_setting(self):
        assert is_env_credential_login_enabled({"disable_env_credential_login": True}) is False

    def test_explicit_false_keeps_it_enabled(self):
        assert is_env_credential_login_enabled({"disable_env_credential_login": False}) is True

    def test_disabled_when_sso_gate_blocks_all_password_logins(self):
        """`disable_password_login_when_sso_enabled` with SSO configured
        rejects every username/password login before the env comparison runs,
        so the banner must not nag about an already-unreachable path."""
        with ExitStack() as stack:
            _patch_sso_configured(stack, configured=True)
            assert is_env_credential_login_enabled({"disable_password_login_when_sso_enabled": True}) is False

    def test_enabled_when_sso_gate_is_set_but_sso_not_configured(self):
        with ExitStack() as stack:
            _patch_sso_configured(stack, configured=False)
            assert is_env_credential_login_enabled({"disable_password_login_when_sso_enabled": True}) is True


def _db_user_row(*, password: str, password_reset_required: bool | None = None, last_breach_check_at=None):
    hashed = hash_token(token=password)
    row = MagicMock()
    row.user_id = "reset-user-1"
    row.user_email = "reset@example.com"
    row.password = hashed
    row.user_role = LitellmUserRoles.INTERNAL_USER
    row.password_reset_required = password_reset_required
    row.last_breach_check_at = last_breach_check_at
    return row


def _prisma_with_user(row) -> MagicMock:
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=row)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=row)
    return mock_prisma_client


_DB_LOGIN_ENV = {
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "UI_USERNAME": "admin",
    "UI_PASSWORD": "admin-password",
}


class TestPasswordResetRequiredSessionMinting:
    """A user flagged `password_reset_required` must receive a UI session key
    restricted to the change-password endpoint (server-side enforcement, so a
    script driving the management API with the session key is blocked too);
    an unflagged user must keep getting an unrestricted key."""

    async def _login(self, mock_prisma_client) -> tuple[LoginResult, dict]:
        with patch.dict(os.environ, _DB_LOGIN_ENV):
            with patch(  # test-quality-ok: asserting the minted key's restriction requires seeing its kwargs
                "litellm.proxy.auth.login_utils.generate_key_helper_fn",
                new_callable=AsyncMock,
                return_value={"token": "session-token"},
            ) as mock_generate_key:
                result = await authenticate_user(
                    username="reset@example.com",
                    password="Str0ng!Passw0rd",
                    master_key="sk-1234",
                    prisma_client=mock_prisma_client,
                    general_settings=_POLICY_NO_BREACH_CHECK,
                )
        return result, mock_generate_key.call_args.kwargs

    @pytest.mark.asyncio
    async def test_flagged_user_gets_key_restricted_to_change_password(self):
        row = _db_user_row(password="Str0ng!Passw0rd", password_reset_required=True)
        result, key_kwargs = await self._login(_prisma_with_user(row))

        assert key_kwargs["allowed_routes"] == ["/user/password/change"]
        assert key_kwargs["metadata"] == {"password_reset_required": True}
        assert result.password_reset_required is True

    @pytest.mark.asyncio
    async def test_unflagged_user_gets_unrestricted_key(self):
        row = _db_user_row(password="Str0ng!Passw0rd", password_reset_required=None)
        result, key_kwargs = await self._login(_prisma_with_user(row))

        assert key_kwargs["allowed_routes"] is None
        assert not key_kwargs["metadata"]
        assert result.password_reset_required is False

    async def _login_with_screen_result(self, mock_prisma_client, breached: bool) -> tuple[LoginResult, dict, dict]:
        with patch.dict(os.environ, _DB_LOGIN_ENV):
            with patch(  # test-quality-ok: asserting the minted key's restriction requires seeing its kwargs
                "litellm.proxy.auth.login_utils.generate_key_helper_fn",
                new_callable=AsyncMock,
                return_value={"token": "session-token"},
            ) as mock_generate_key:
                with (
                    patch(  # test-quality-ok: authenticate_user has no HIBP client seam; the screen itself is tested against MockTransport below
                        "litellm.proxy.auth.login_utils.screen_login_password_for_breach",
                        new_callable=AsyncMock,
                        return_value=breached,
                    ) as mock_screen
                ):
                    result = await authenticate_user(
                        username="reset@example.com",
                        password="Str0ng!Passw0rd",
                        master_key="sk-1234",
                        prisma_client=mock_prisma_client,
                        general_settings=_POLICY_NO_BREACH_CHECK,
                    )
        return result, mock_generate_key.call_args.kwargs, mock_screen.call_args.kwargs

    @pytest.mark.asyncio
    async def test_login_screens_with_row_state_before_minting(self):
        """The login must hand the screen the row's recheck timestamp, or the
        24h throttle can never work."""
        checked_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _db_user_row(password="Str0ng!Passw0rd", last_breach_check_at=checked_at)
        mock_prisma_client = _prisma_with_user(row)

        _, _, screen_kwargs = await self._login_with_screen_result(mock_prisma_client, breached=False)

        assert screen_kwargs["user_id"] == "reset-user-1"
        assert screen_kwargs["password"] == "Str0ng!Passw0rd"
        assert screen_kwargs["last_breach_check_at"] == checked_at
        assert screen_kwargs["prisma_client"] is mock_prisma_client

    @pytest.mark.asyncio
    async def test_fresh_breach_hit_restricts_the_current_session(self):
        """A breach found during THIS login must restrict THIS session, not
        just the next one."""
        row = _db_user_row(password="Str0ng!Passw0rd", password_reset_required=None)
        mock_prisma_client = _prisma_with_user(row)

        result, key_kwargs, _ = await self._login_with_screen_result(mock_prisma_client, breached=True)

        assert key_kwargs["allowed_routes"] == ["/user/password/change"]
        assert key_kwargs["metadata"] == {"password_reset_required": True}
        assert result.password_reset_required is True


def _sha1_upper(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()


def _client_with_transport(handler) -> AsyncHTTPHandler:
    http_handler = AsyncHTTPHandler()
    http_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return http_handler


def _client_returning_breach_hit(password: str) -> AsyncHTTPHandler:
    body = f"{_sha1_upper(password)[5:]}:42"
    return _client_with_transport(lambda request: httpx.Response(200, text=body))


def _client_returning_no_hit() -> AsyncHTTPHandler:
    return _client_with_transport(lambda request: httpx.Response(200, text="0000000000000000000000000000000000A:3"))


def _client_never_called() -> AsyncHTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call to {request.url}")

    return _client_with_transport(handler)


class TestScreenLoginPasswordForBreach:
    """The awaited login-time screen: flags a breached password for a forced
    reset, stamps the recheck timestamp, rechecks at most every 24h, returns
    the breach verdict so the login can restrict the session it is minting,
    and never raises into the login."""

    @pytest.mark.asyncio
    async def test_breached_password_sets_reset_flag_and_timestamp(self):
        password = "Password123!"
        mock_prisma_client = _prisma_with_user(None)

        breached = await screen_login_password_for_breach(
            user_id="reset-user-1",
            password=password,
            last_breach_check_at=None,
            general_settings={},
            prisma_client=mock_prisma_client,
            client=_client_returning_breach_hit(password),
        )

        assert breached is True
        update_kwargs = mock_prisma_client.db.litellm_usertable.update.call_args.kwargs
        assert update_kwargs["where"] == {"user_id": "reset-user-1"}
        assert update_kwargs["data"]["password_reset_required"] is True
        assert isinstance(update_kwargs["data"]["last_breach_check_at"], datetime)

    @pytest.mark.asyncio
    async def test_clean_password_stamps_timestamp_without_flag(self):
        mock_prisma_client = _prisma_with_user(None)

        breached = await screen_login_password_for_breach(
            user_id="reset-user-1",
            password="Str0ng!Passw0rd",
            last_breach_check_at=None,
            general_settings={},
            prisma_client=mock_prisma_client,
            client=_client_returning_no_hit(),
        )

        assert breached is False
        update_kwargs = mock_prisma_client.db.litellm_usertable.update.call_args.kwargs
        assert "password_reset_required" not in update_kwargs["data"]
        assert isinstance(update_kwargs["data"]["last_breach_check_at"], datetime)

    @pytest.mark.asyncio
    async def test_skips_hibp_when_checked_within_24_hours(self):
        mock_prisma_client = _prisma_with_user(None)

        breached = await screen_login_password_for_breach(
            user_id="reset-user-1",
            password="Password123!",
            last_breach_check_at=datetime.now(timezone.utc) - timedelta(hours=23),
            general_settings={},
            prisma_client=mock_prisma_client,
            client=_client_never_called(),
        )

        assert breached is False
        mock_prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_rechecks_when_last_check_is_older_than_24_hours(self):
        password = "Password123!"
        mock_prisma_client = _prisma_with_user(None)

        breached = await screen_login_password_for_breach(
            user_id="reset-user-1",
            password=password,
            last_breach_check_at=datetime.now(timezone.utc) - timedelta(hours=25),
            general_settings={},
            prisma_client=mock_prisma_client,
            client=_client_returning_breach_hit(password),
        )

        assert breached is True
        assert (
            mock_prisma_client.db.litellm_usertable.update.call_args.kwargs["data"]["password_reset_required"] is True
        )

    @pytest.mark.asyncio
    async def test_skips_hibp_when_check_disabled(self):
        mock_prisma_client = _prisma_with_user(None)

        breached = await screen_login_password_for_breach(
            user_id="reset-user-1",
            password="Password123!",
            last_breach_check_at=None,
            general_settings=_POLICY_NO_BREACH_CHECK,
            prisma_client=mock_prisma_client,
            client=_client_never_called(),
        )

        assert breached is False
        mock_prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_failure_never_raises_but_still_reports_the_breach(self):
        """A failed flag write must not fail the login, but the breach verdict
        still has to restrict the session being minted right now."""
        password = "Password123!"
        mock_prisma_client = _prisma_with_user(None)
        mock_prisma_client.db.litellm_usertable.update = AsyncMock(side_effect=RuntimeError("db down"))

        assert (
            await screen_login_password_for_breach(
                user_id="reset-user-1",
                password=password,
                last_breach_check_at=None,
                general_settings={},
                prisma_client=mock_prisma_client,
                client=_client_returning_breach_hit(password),
            )
            is True
        )

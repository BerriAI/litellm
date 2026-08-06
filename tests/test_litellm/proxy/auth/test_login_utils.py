"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _unlimited_throttle():
    """A throttle wired to a real in-memory store with a limit no test can reach."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    return LoginThrottle(client_ip="1.2.3.4", max_attempts=10_000, window_seconds=900, cache=DualCache())



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
                        throttle=_unlimited_throttle(),
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
                        throttle=_unlimited_throttle(),
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
                throttle=_unlimited_throttle(),
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
            throttle=_unlimited_throttle(),
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
                throttle=_unlimited_throttle(),
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
                throttle=_unlimited_throttle(),
            )
            result_lower = await authenticate_user(
                username=stored_email,
                password=correct_password,
                master_key=master_key,
                prisma_client=mock_prisma_client,
                throttle=_unlimited_throttle(),
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
                        throttle=_unlimited_throttle(),
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
                        throttle=_unlimited_throttle(),
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
                        throttle=_unlimited_throttle(),
                    )
                    result2 = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                        throttle=_unlimited_throttle(),
                    )
                    result3 = await authenticate_user(
                        username=ui_username,
                        password=ui_password,
                        master_key=master_key,
                        prisma_client=mock_prisma_client,
                        throttle=_unlimited_throttle(),
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
                throttle=_unlimited_throttle(),
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


# ---------------------------------------------------------------------------
# Failed-login accounting (LIT-5285)
# ---------------------------------------------------------------------------


def _throttle(max_attempts: int = 3, window_seconds: int = 900, client_ip: str = "1.2.3.4", cache=None, redis_cache=None):
    """A throttle over a real in-memory store, so the tests exercise the true counters."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    return LoginThrottle(
        client_ip=client_ip,
        max_attempts=max_attempts,
        window_seconds=window_seconds,
        cache=cache if cache is not None else DualCache(),
        redis_cache=redis_cache,
    )


async def _guess(throttle, username: str = "admin", password: str = "wrong"):
    from litellm.proxy.auth.login_utils import authenticate_user

    return await authenticate_user(
        username=username,
        password=password,
        master_key="sk-master",
        prisma_client=None,
        throttle=throttle,
    )


@pytest.mark.asyncio
async def test_attempts_are_refused_once_the_limit_is_reached(monkeypatch):
    """The limit denies further attempts for the window, and the denial carries Retry-After."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=3, window_seconds=77)

    for _ in range(3):
        with pytest.raises(ProxyException) as first:
            await _guess(throttle)
        assert first.value.code == "401"

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429"
    assert blocked.value.headers.get("Retry-After") == "77"


@pytest.mark.asyncio
async def test_a_correct_password_is_refused_while_blocked(monkeypatch):
    """The check precedes the credential comparison, so being over the limit wins."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle, password="right")
    assert blocked.value.code == "429"


@pytest.mark.asyncio
async def test_a_blocked_attempt_does_not_extend_the_window(monkeypatch):
    """Hammering while blocked must not push the counter or refresh its TTL."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2)
    key = throttle._key("admin")

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)
    counted_at_limit = await throttle._failures(key)

    for _ in range(5):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    assert await throttle._failures(key) == counted_at_limit == 2


@pytest.mark.asyncio
async def test_a_successful_sign_in_clears_the_bucket(monkeypatch):
    """Success resets the budget rather than leaving the operator near the limit."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(max_attempts=3)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        await _guess(throttle, password="right")

    assert await throttle._failures(throttle._key("admin")) == 0


@pytest.mark.asyncio
async def test_a_configuration_error_never_counts(monkeypatch):
    """A 500 from an unset master key is not a guess and must not consume the budget."""
    from litellm.proxy._types import ProxyException

    throttle = _throttle(max_attempts=2)
    for _ in range(5):
        with pytest.raises(ProxyException) as exc:
            await authenticate_user(
                username="admin", password="x", master_key=None, prisma_client=None, throttle=throttle
            )
        assert exc.value.code == "500"

    assert await throttle._failures(throttle._key("admin")) == 0


@pytest.mark.asyncio
async def test_the_username_is_case_folded_into_one_bucket(monkeypatch):
    """The DB lookup is case-insensitive, so casing must not multiply the budget."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=4)

    for name in ("admin@corp.com", "ADMIN@corp.com", "Admin@corp.com", "aDmIn@corp.com"):
        with pytest.raises(ProxyException) as exc:
            await _guess(throttle, username=name)
        assert exc.value.code == "401"

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle, username="admin@CORP.com")
    assert blocked.value.code == "429"


@pytest.mark.asyncio
async def test_a_different_username_from_the_same_source_is_unaffected(monkeypatch):
    """The bucket is the pair, so one username's failures do not block another."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2)

    for _ in range(3):
        with pytest.raises(ProxyException):
            await _guess(throttle, username="admin")

    with pytest.raises(ProxyException) as other:
        await _guess(throttle, username="someone-else@example.com")
    assert other.value.code == "401", "a second username must still reach the credential check"


@pytest.mark.asyncio
async def test_both_credential_rejections_are_indistinguishable(monkeypatch):
    """One message for the known and the unknown username, so responses do not enumerate."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")

    with pytest.raises(ProxyException) as unknown:
        await _guess(_throttle(max_attempts=99), username="nobody@example.com")

    fake_user = MagicMock()
    fake_user.user_id = "u-1"
    fake_user.user_email = "known@example.com"
    fake_user.user_role = "internal_user"
    fake_user.password = "scrypt:fake"
    repo = MagicMock()
    repo.return_value.table.find_first = AsyncMock(return_value=fake_user)
    with patch("litellm.proxy.auth.login_utils.UserRepository", repo), patch(
        "litellm.proxy.auth.login_utils.verify_password", return_value=False
    ):
        with pytest.raises(ProxyException) as known:
            await authenticate_user(
                username="known@example.com",
                password="wrong",
                master_key="sk-master",
                prisma_client=MagicMock(),
                throttle=_throttle(max_attempts=99),
            )

    assert unknown.value.message == known.value.message
    assert "known@example.com" not in unknown.value.message + known.value.message


@pytest.mark.asyncio
async def test_a_user_with_no_password_set_does_not_consume_the_budget(monkeypatch):
    """That 401 is deterministic and guards no secret, so counting it would only let
    someone burn a passwordless account's bucket."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2)

    passwordless = MagicMock()
    passwordless.user_id = "u-2"
    passwordless.user_email = "nopass@example.com"
    passwordless.user_role = "internal_user"
    passwordless.password = None
    repo = MagicMock()
    repo.return_value.table.find_first = AsyncMock(return_value=passwordless)

    with patch("litellm.proxy.auth.login_utils.UserRepository", repo):
        for _ in range(5):
            with pytest.raises(ProxyException) as exc:
                await authenticate_user(
                    username="nopass@example.com",
                    password="x",
                    master_key="sk-master",
                    prisma_client=MagicMock(),
                    throttle=throttle,
                )
            assert exc.value.code == "401"

    assert await throttle._failures(throttle._key("nopass@example.com")) == 0


@pytest.mark.asyncio
async def test_a_wrong_password_for_a_known_user_also_counts(monkeypatch):
    """The database-user branch must charge the bucket too, not just the unknown-user branch."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=3)

    known = MagicMock()
    known.user_id = "u-1"
    known.user_email = "known@example.com"
    known.user_role = "internal_user"
    known.password = "scrypt:stored"
    repo = MagicMock()
    repo.return_value.table.find_first = AsyncMock(return_value=known)

    async def _attempt():
        return await authenticate_user(
            username="known@example.com",
            password="wrong",
            master_key="sk-master",
            prisma_client=MagicMock(),
            throttle=throttle,
        )

    with patch("litellm.proxy.auth.login_utils.UserRepository", repo), patch(
        "litellm.proxy.auth.login_utils.verify_password", return_value=False
    ):
        for _ in range(3):
            with pytest.raises(ProxyException) as rejected:
                await _attempt()
            assert rejected.value.code == "401"

        with pytest.raises(ProxyException) as blocked:
            await _attempt()
    assert blocked.value.code == "429"


@pytest.mark.asyncio
async def test_two_source_addresses_do_not_share_a_bucket(monkeypatch):
    """The key is the pair, so one address exhausting its budget must not block another.

    Dropping the address from the key would turn this into the username-only counter the
    design rejects, where anyone can lock a named admin out from anywhere.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    shared_store = DualCache()
    attacker = _throttle(max_attempts=2, client_ip="203.0.113.9", cache=shared_store)
    operator = _throttle(max_attempts=2, client_ip="198.51.100.7", cache=shared_store)

    for _ in range(3):
        with pytest.raises(ProxyException):
            await _guess(attacker, username="admin")

    with pytest.raises(ProxyException) as blocked:
        await _guess(attacker, username="admin")
    assert blocked.value.code == "429"

    with pytest.raises(ProxyException) as unaffected:
        await _guess(operator, username="admin")
    assert unaffected.value.code == "401", "the real operator must still reach the credential check"


class _NoExpiryRedis:
    """Redis that stores the counter but never records an expiry for it.

    Models the window between INCRBYFLOAT committing and the TTL call failing.
    """

    def __init__(self):
        self.values: dict = {}
        self.expiry_repairs = 0

    async def async_get_cache(self, key, **kwargs):
        return self.values.get(key)

    async def async_increment(self, key, value, ttl=None, **kwargs):
        if int(value) == 0:
            self.expiry_repairs += 1
        self.values[key] = self.values.get(key, 0) + int(value)
        return self.values[key]

    async def async_get_ttl(self, key):
        return None

    async def async_delete_cache(self, key):
        self.values.pop(key, None)


@pytest.mark.asyncio
async def test_a_counter_left_without_an_expiry_is_repaired(monkeypatch):
    """Regression: a counter with no TTL would refuse the pair forever.

    Redis commits the increment before setting the expiry, and nothing increments the key
    again once the limit is reached, so a TTL that never landed is never repaired on its
    own and the username and source pair stays refused with no way back.
    """
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    redis = _NoExpiryRedis()
    throttle = _throttle(max_attempts=2, redis_cache=redis)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    assert redis.expiry_repairs >= 1, "each recorded failure must leave the counter with an expiry"

    repairs_before_block = redis.expiry_repairs
    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429"
    assert redis.expiry_repairs > repairs_before_block, "the refusal path must repair a missing expiry too"


@pytest.mark.asyncio
async def test_counters_do_not_share_the_key_authentication_cache(monkeypatch):
    """Regression: throttle entries must not evict cached credentials.

    user_api_key_cache holds at most 200 in-memory entries and evicts the soonest to
    expire first, so parking 900s sign-in counters there let a stream of made-up usernames
    push out the much shorter lived credential entries, sending every ordinary API request
    back to the database.
    """
    from litellm.proxy import proxy_server as ps
    from litellm.proxy.auth.login_throttle import _CACHE_KEY_PREFIX, LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setattr(ps, "redis_usage_cache", None)

    auth_cache_keys_before = set(ps.user_api_key_cache.in_memory_cache.cache_dict)

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"
    throttle = LoginThrottle.from_request(request)

    for i in range(25):
        with pytest.raises(Exception):
            await _guess(throttle, username=f"made-up-{i}@example.com")

    added = set(ps.user_api_key_cache.in_memory_cache.cache_dict) - auth_cache_keys_before
    assert not [k for k in added if str(k).startswith(_CACHE_KEY_PREFIX)], (
        "sign-in counters must live in their own cache, not the key-authentication cache"
    )


@pytest.mark.asyncio
async def test_a_refused_username_cannot_forge_log_lines(monkeypatch):
    """The username reaches a warning log, so it must not carry newlines or control bytes."""
    import logging

    from litellm._logging import verbose_proxy_logger
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=1)
    forged = "victim@example.com\nWARNING: sign-in succeeded for attacker\x00"

    with pytest.raises(ProxyException):
        await _guess(throttle, username=forged)

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    verbose_proxy_logger.addHandler(handler)
    try:
        with pytest.raises(ProxyException) as blocked:
            await _guess(throttle, username=forged)
    finally:
        verbose_proxy_logger.removeHandler(handler)

    assert blocked.value.code == "429"
    emitted = [r.getMessage() for r in records if "sign-in attempts exhausted" in r.getMessage()]
    assert emitted, "the refusal must be logged"
    assert "\n" not in emitted[0] and "\x00" not in emitted[0]
    assert "victim@example.com" in emitted[0]


@pytest.mark.asyncio
async def test_a_username_spray_cannot_evict_an_existing_counter(monkeypatch):
    """Regression: the in-memory tier must hold more counters than a spray can create.

    The default in-memory cache keeps 200 entries and evicts the soonest to expire, and
    every counter shares one window, so eviction was effectively oldest-first. A few
    hundred made-up usernames therefore pushed out the attacker's own counter and handed
    back a fresh allowance against the real account.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.login_throttle import _FAILED_LOGIN_CACHE, _MAX_TRACKED_LOGIN_SOURCES

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    assert _MAX_TRACKED_LOGIN_SOURCES >= 10_000
    assert _FAILED_LOGIN_CACHE.in_memory_cache.max_size_in_memory == _MAX_TRACKED_LOGIN_SOURCES

    throttle = _throttle(max_attempts=3, cache=_FAILED_LOGIN_CACHE, client_ip="10.9.9.9")
    victim = "spray-victim@corp.com"
    for _ in range(3):
        with pytest.raises(ProxyException):
            await _guess(throttle, username=victim)

    for i in range(500):
        await throttle.record_failure(f"spray-filler-{i}@corp.com")

    assert await throttle._failures(throttle._key(victim)) == 3, "the counter must survive a spray"
    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle, username=victim)
    assert blocked.value.code == "429"

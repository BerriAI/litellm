"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import os
from contextlib import ExitStack
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _RecordedSleeps:
    """A sleep that records what it was asked to wait for instead of waiting."""

    def __init__(self):
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


@pytest.fixture(autouse=True)
def login_delays(monkeypatch):
    """Replace the failed-login wait, so the suite pays no wall clock and can read it back."""
    from litellm.proxy.auth import login_throttle

    recorded = _RecordedSleeps()
    monkeypatch.setattr(login_throttle, "_sleep", recorded)
    login_throttle._DELAYS_IN_FLIGHT.clear()
    yield recorded
    login_throttle._DELAYS_IN_FLIGHT.clear()


def _unlimited_throttle():
    """A throttle wired to a real in-memory store with a limit no test can reach."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    store: Final = DualCache()
    return LoginThrottle(
        client_ip="1.2.3.4",
        max_attempts=10_000,
        max_attempts_per_source=10_000,
        window_seconds=900,
        username_cache=store,
        source_cache=store,
    )



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
    is_env_credential_login_enabled,
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


def _throttle(
    max_attempts: int = 3,
    window_seconds: int = 900,
    client_ip: str = "1.2.3.4",
    cache=None,
    redis_cache=None,
    max_attempts_per_source: int = 10_000,
):
    """A throttle over a real in-memory store, so the tests exercise the true counters."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    store: Final = cache if cache is not None else DualCache()
    return LoginThrottle(
        client_ip=client_ip,
        max_attempts=max_attempts,
        max_attempts_per_source=max_attempts_per_source,
        window_seconds=window_seconds,
        username_cache=store,
        source_cache=store,
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
async def test_a_correct_admin_password_is_accepted_while_blocked(monkeypatch):
    """The configured admin credentials are compared before the gate, so the operator gets in.

    A throttle that refuses a valid password hands anyone who can reach the login form a
    denial of service against the one account that can fix it.
    """
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(max_attempts=2)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    with pytest.raises(ProxyException) as still_blocked:
        await _guess(throttle)
    assert still_blocked.value.code == "429", "a wrong password is still refused"

    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(  # test-quality-ok: success mints a UI key and persists the user; faked so no DB is needed
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        result = await _guess(throttle, password="right")
    assert result.key == "sk-ui"


@pytest.mark.asyncio
async def test_a_blocked_attempt_does_not_extend_the_window(monkeypatch):
    """Hammering while blocked must not push the counter or refresh its TTL."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2)
    key = throttle._username_key("admin")

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)
    counted_at_limit = await throttle._failures(throttle.username_cache, key)

    for _ in range(5):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    assert await throttle._failures(throttle.username_cache, key) == counted_at_limit == 2


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

    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(  # test-quality-ok: success mints a UI key and persists the user; faked so no DB is needed
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        await _guess(throttle, password="right")

    assert await throttle._failures(throttle.username_cache, throttle._username_key("admin")) == 0


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

    assert await throttle._failures(throttle.username_cache, throttle._username_key("admin")) == 0


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
    """The counters are independent, so one username's failures do not exhaust another's."""
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
    with patch("litellm.proxy.auth.login_utils.UserRepository", repo), patch(  # test-quality-ok: reaches the known-DB-user branch without a database
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

    with patch("litellm.proxy.auth.login_utils.UserRepository", repo):  # test-quality-ok: reaches the passwordless-DB-user branch without a database
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

    assert await throttle._failures(throttle.username_cache, throttle._username_key("nopass@example.com")) == 0


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

    with patch("litellm.proxy.auth.login_utils.UserRepository", repo), patch(  # test-quality-ok: reaches the known-DB-user branch without a database
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
async def test_one_source_exhausting_its_own_budget_does_not_refuse_another_source(monkeypatch):
    """The source counter is per address, so a noisy office does not take its neighbour down."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    shared_store = DualCache()
    attacker = _throttle(
        max_attempts=10_000, max_attempts_per_source=2, client_ip="203.0.113.9", cache=shared_store
    )
    operator = _throttle(
        max_attempts=10_000, max_attempts_per_source=2, client_ip="198.51.100.7", cache=shared_store
    )

    for i in range(2):
        with pytest.raises(ProxyException):
            await _guess(attacker, username=f"target-{i}@corp.com")

    with pytest.raises(ProxyException) as blocked:
        await _guess(attacker, username="target-2@corp.com")
    assert blocked.value.code == "429"

    with pytest.raises(ProxyException) as unaffected:
        await _guess(operator, username="target-3@corp.com")
    assert unaffected.value.code == "401", "the other address must still reach the credential check"


@pytest.mark.asyncio
async def test_a_username_exhausted_from_one_source_is_refused_from_another(monkeypatch):
    """The username counter carries no address, so spreading the guesses buys nothing.

    The pair key this replaced reset the budget for every new address, which is exactly the
    shape of a credential-stuffing run from a proxy pool.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    shared_store = DualCache()
    first_hop = _throttle(max_attempts=2, client_ip="203.0.113.9", cache=shared_store)
    second_hop = _throttle(max_attempts=2, client_ip="198.51.100.7", cache=shared_store)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(first_hop, username="victim@corp.com")

    with pytest.raises(ProxyException) as rotated:
        await _guess(second_hop, username="victim@corp.com")
    assert rotated.value.code == "429"


@pytest.mark.asyncio
async def test_a_source_wide_spray_is_counted_even_though_each_username_is_fresh(monkeypatch):
    """One guess against each of many usernames never trips a username counter, only the source one."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=10_000, max_attempts_per_source=6, client_ip="203.0.113.11")

    for i in range(6):
        with pytest.raises(ProxyException) as rejected:
            await _guess(throttle, username=f"sprayed-{i}@corp.com")
        assert rejected.value.code == "401"

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle, username="sprayed-7@corp.com")
    assert blocked.value.code == "429"
    assert await throttle._failures(throttle.username_cache, throttle._username_key("sprayed-7@corp.com")) == 0


@pytest.mark.asyncio
async def test_a_successful_sign_in_leaves_the_source_counter_alone(monkeypatch):
    """One account's success says nothing about the other attempts the address is making."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(max_attempts=10)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(  # test-quality-ok: success mints a UI key and persists the user; faked so no DB is needed
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        await _guess(throttle, password="right")

    assert await throttle._failures(throttle.username_cache, throttle._username_key("admin")) == 0
    assert await throttle._failures(throttle.source_cache, throttle._source_key()) == 2


@pytest.mark.asyncio
async def test_the_delay_doubles_from_one_second_and_is_capped(monkeypatch, login_delays):
    """Guessing has to cost wall clock, and the cost has to stop short of an unbounded hang."""
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.login_throttle import MAX_DELAY_SECONDS

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=10_000)

    for _ in range(9):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    assert login_delays.seconds == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0], (
        "the first two failures answer immediately, then the wait doubles up to the cap"
    )
    assert max(login_delays.seconds) == MAX_DELAY_SECONDS


@pytest.mark.asyncio
async def test_the_delay_tracks_whichever_counter_is_further_past_its_onset(monkeypatch):
    """A source deep into a spray must not be answered instantly just because the username is fresh."""
    from litellm.proxy.auth.login_throttle import FailureCounts, LoginThrottle

    assert LoginThrottle.delay_seconds(FailureCounts(username=1, source=1)) == 0.0
    assert LoginThrottle.delay_seconds(FailureCounts(username=2, source=24)) == 0.0
    assert LoginThrottle.delay_seconds(FailureCounts(username=3, source=1)) == 1.0
    assert LoginThrottle.delay_seconds(FailureCounts(username=1, source=25)) == 1.0
    assert LoginThrottle.delay_seconds(FailureCounts(username=4, source=28)) == 8.0


@pytest.mark.asyncio
async def test_held_attempts_from_one_source_are_capped(monkeypatch):
    """Holding a rejected attempt open must not let one address park unlimited sockets."""
    import asyncio

    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth import login_throttle as lt
    from litellm.proxy.auth.login_throttle import MAX_CONCURRENT_DELAYS_PER_SOURCE

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    release = asyncio.Event()

    async def _park(_seconds: float) -> None:
        await release.wait()

    monkeypatch.setattr(lt, "_sleep", _park)
    throttle = _throttle(max_attempts=10_000, client_ip="203.0.113.44")
    await throttle.record_failure("admin")
    await throttle.record_failure("admin")

    held = [asyncio.create_task(_guess(throttle)) for _ in range(MAX_CONCURRENT_DELAYS_PER_SOURCE)]
    for _ in range(1000):
        if lt._DELAYS_IN_FLIGHT.get("203.0.113.44") == MAX_CONCURRENT_DELAYS_PER_SOURCE:
            break
        await asyncio.sleep(0)
    assert lt._DELAYS_IN_FLIGHT.get("203.0.113.44") == MAX_CONCURRENT_DELAYS_PER_SOURCE

    try:
        with pytest.raises(ProxyException) as over_cap:
            await _guess(throttle)
        assert over_cap.value.code == "429"
        assert over_cap.value.headers.get("Retry-After") == "30"
    finally:
        release.set()
        for task in held:
            with pytest.raises(ProxyException):
                await task

    with pytest.raises(ProxyException) as after_drain:
        await _guess(throttle)
    assert after_drain.value.code == "401", "the cap must release once the held attempts answer"


@pytest.mark.asyncio
async def test_disabling_the_control_removes_the_delay_as_well(monkeypatch, login_delays):
    """The escape hatch has to turn off the whole control, not only the refusal."""
    import dataclasses

    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = dataclasses.replace(_throttle(max_attempts=2), enabled=False)

    for _ in range(6):
        with pytest.raises(ProxyException) as rejected:
            await _guess(throttle)
        assert rejected.value.code == "401"

    assert login_delays.seconds == []


class _FakeRedis:
    """Redis whose only counter write is the atomic INCRBY-plus-EXPIRE Lua call.

    `async_increment` is deliberately absent: a two-step increment would fail the test
    with AttributeError, because Redis could then commit a count without its expiry.
    """

    def __init__(self):
        self.values: dict = {}
        self.ttls: dict = {}

    async def async_get_cache(self, key, **kwargs):
        return self.values.get(key)

    async def async_batch_get_counts(self, key_list):
        return tuple(self.values.get(key) for key in key_list)

    async def async_increment_with_floor(self, key, value, ttl):
        self.values[key] = self.values.get(key, 0) + value
        self.ttls.setdefault(key, ttl)
        return self.values[key]

    async def async_get_ttl(self, key):
        return self.ttls.get(key)

    async def async_delete_cache(self, key):
        self.values.pop(key, None)
        self.ttls.pop(key, None)

    def persist(self):
        self.ttls.clear()


@pytest.mark.asyncio
async def test_counters_are_written_with_their_expiry_and_re_armed_if_stripped(monkeypatch):
    """Regression: a counter with no TTL would refuse the pair forever.

    Nothing increments a key once the limit is reached, so a counter that ever exists
    without an expiry stays refused with no way back. Every write must therefore carry the
    expiry, and a refusal that finds it stripped (PERSIST) must put the window back.
    """
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    redis = _FakeRedis()
    throttle = _throttle(max_attempts=2, window_seconds=77, redis_cache=redis)

    for _ in range(2):
        with pytest.raises(ProxyException):
            await _guess(throttle)

    assert redis.values, "failures must land in the shared counter"
    assert set(redis.ttls) == set(redis.values), "no counter may exist without its expiry"
    assert set(redis.ttls.values()) == {77}

    redis.persist()
    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429"
    assert blocked.value.headers.get("Retry-After") == "77"
    assert set(redis.ttls) >= {k for k in redis.values if ":user:" in k}, "the refusal must re-arm a stripped expiry"


class _DownRedis(_FakeRedis):
    """Redis whose every call fails, as during an outage or an open circuit breaker.

    `async_get_cache` returns None rather than raising, as the real one does: it swallows the
    error, so a failed GET is indistinguishable from an empty key to anyone reading through it.
    """

    async def async_get_cache(self, key, **kwargs):
        return None

    async def async_batch_get_counts(self, key_list):
        raise ConnectionError("redis is down")

    async def async_increment_with_floor(self, key, value, ttl):
        raise ConnectionError("redis is down")

    async def async_get_ttl(self, key):
        raise ConnectionError("redis is down")

    async def async_delete_cache(self, key):
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_redis_is_the_only_counter_while_it_answers(monkeypatch):
    """Regression: every worker must spend the same budget, and a success must clear it for all.

    Counting in this worker's memory as well as in Redis let the two drift apart: a worker
    whose Redis write failed kept its own count while the others gave the attacker fresh
    guesses, and a stale local count outlived the shared clear after a correct password.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.login_throttle import _CACHE_KEY_PREFIX

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    redis = _FakeRedis()
    first_worker_store = DualCache()
    second_worker_store = DualCache()
    first_worker = _throttle(max_attempts=2, cache=first_worker_store, redis_cache=redis)
    second_worker = _throttle(max_attempts=2, cache=second_worker_store, redis_cache=redis)

    for _ in range(2):
        with pytest.raises(ProxyException, match="Invalid credentials"):
            await _guess(first_worker)

    assert not [k for k in first_worker_store.in_memory_cache.cache_dict if str(k).startswith(_CACHE_KEY_PREFIX)], (
        "with Redis answering, no worker may keep a counter of its own"
    )
    with pytest.raises(ProxyException) as blocked:
        await _guess(second_worker)
    assert blocked.value.code == "429", "the second worker must see the budget the first one spent"

    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(  # test-quality-ok: success mints a UI key and persists the user; faked so no DB is needed
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        await _guess(second_worker, password="right")

    assert not [k for k in redis.values if ":user:" in k], "a success must clear the shared username counter"
    with pytest.raises(ProxyException, match="Invalid credentials"):
        await _guess(first_worker)


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_own_counter(monkeypatch):
    """With Redis raising, guesses are still counted and refused, per worker, instead of unbounded."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(max_attempts=2, redis_cache=_DownRedis())

    for _ in range(2):
        with pytest.raises(ProxyException, match="Invalid credentials"):
            await _guess(throttle)

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429"
    assert blocked.value.headers.get("Retry-After") == "900"


class _WriteRefusingRedis(_FakeRedis):
    """Redis that answers reads but raises on writes until `recover()` is called."""

    def __init__(self):
        super().__init__()
        self.writable = False

    def recover(self):
        self.writable = True

    async def async_increment_with_floor(self, key, value, ttl):
        if not self.writable:
            raise ConnectionError("redis write failed")
        return await super().async_increment_with_floor(key, value, ttl)


@pytest.mark.asyncio
async def test_failures_redis_refused_still_count_once_redis_recovers(monkeypatch):
    """Regression: a guess Redis could not record must not be forgotten when Redis comes back.

    Such a guess lands in this worker's own store. Reading only Redis afterwards handed the
    attacker that guess again, so the budget was the limit plus however many writes failed.
    """
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    redis = _WriteRefusingRedis()
    throttle = _throttle(max_attempts=2, redis_cache=redis)

    with pytest.raises(ProxyException, match="Invalid credentials"):
        await _guess(throttle)
    assert not redis.values, "the refused write must not have reached Redis"

    redis.recover()
    with pytest.raises(ProxyException, match="Invalid credentials"):
        await _guess(throttle)
    assert [v for k, v in redis.values.items() if ":user:" in k] == [1], "only the recorded guess is in Redis"

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429", "the guess Redis missed and the one it took must add up to the limit"


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

    auth_cache_keys_before = set(ps.user_api_key_cache.in_memory_cache.cache_dict)

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"
    throttle = LoginThrottle.from_request(request, general_settings={}, redis_cache=None)

    for i in range(25):
        with pytest.raises(ProxyException, match="Invalid credentials"):
            await _guess(throttle, username=f"made-up-{i}@example.com")

    added = set(ps.user_api_key_cache.in_memory_cache.cache_dict) - auth_cache_keys_before
    assert not [k for k in added if str(k).startswith(_CACHE_KEY_PREFIX)], (
        "sign-in counters must live in their own cache, not the key-authentication cache"
    )


def test_settings_that_arrive_as_environment_strings_are_honored():
    """An `os.environ/VAR` reference in general_settings resolves to a string, not an int.

    Regression: a digit string fell back to the default with only a log line, so an operator
    tightening the limits through environment substitution silently kept the stock ceilings.
    """
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"

    throttle = LoginThrottle.from_request(
        request,
        general_settings={
            "max_failed_login_attempts": "7",
            "max_failed_login_attempts_per_source": " 70 ",
            "failed_login_window_seconds": "not-a-number",
        },
        redis_cache=None,
    )

    assert throttle.max_attempts == 7
    assert throttle.max_attempts_per_source == 70
    assert throttle.window_seconds == 900, "garbage still falls back to the default"


def test_the_disable_flag_is_read_once_not_per_login_attempt(monkeypatch):
    """Regression: the kill switch was read through the secret manager on every unauthenticated request.

    With a hosted secret manager in read mode that is a synchronous network call per guess, so a
    flood of wrong passwords could exhaust the secret manager even after the source was refused.
    """
    from litellm.proxy.auth import login_throttle

    reads: Final[list[str]] = []  # mutable-ok: test-only call recorder
    monkeypatch.setattr(login_throttle, "get_secret_bool", lambda name, default: reads.append(name) or default)
    login_throttle._rate_limit_disabled.cache_clear()
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"

    for _ in range(50):
        assert login_throttle.LoginThrottle.from_request(request, general_settings={}, redis_cache=None).enabled is True

    login_throttle._rate_limit_disabled.cache_clear()
    assert reads == ["LITELLM_DISABLE_LOGIN_RATE_LIMIT"]


def test_a_negative_or_boolean_setting_falls_back_to_the_default():
    """A limit below one would refuse everyone; a bool is a typo, not a count."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"

    throttle = LoginThrottle.from_request(
        request,
        general_settings={"max_failed_login_attempts": "-7", "max_failed_login_attempts_per_source": True},
        redis_cache=None,
    )

    assert throttle.max_attempts == 50
    assert throttle.max_attempts_per_source == 250


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
    back a fresh allowance against the real account. Username and source counters must also
    live in separate stores, or the same spray evicts the source counter meant to stop it.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.login_throttle import (
        LoginThrottle,
        _FAILED_LOGIN_SOURCE_CACHE,
        _FAILED_LOGIN_USERNAME_CACHE,
        _MAX_TRACKED_LOGIN_SOURCES,
        _MAX_TRACKED_LOGIN_USERNAMES,
    )

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    assert _MAX_TRACKED_LOGIN_SOURCES >= 10_000
    assert _MAX_TRACKED_LOGIN_USERNAMES >= 10_000
    assert _FAILED_LOGIN_SOURCE_CACHE.in_memory_cache.max_size_in_memory == _MAX_TRACKED_LOGIN_SOURCES
    assert _FAILED_LOGIN_USERNAME_CACHE.in_memory_cache.max_size_in_memory == _MAX_TRACKED_LOGIN_USERNAMES
    assert _FAILED_LOGIN_SOURCE_CACHE.in_memory_cache is not _FAILED_LOGIN_USERNAME_CACHE.in_memory_cache

    throttle = LoginThrottle(
        client_ip="10.9.9.9",
        max_attempts=3,
        max_attempts_per_source=10_000,
        window_seconds=900,
        username_cache=_FAILED_LOGIN_USERNAME_CACHE,
        source_cache=_FAILED_LOGIN_SOURCE_CACHE,
    )
    victim = "spray-victim@corp.com"
    for _ in range(3):
        with pytest.raises(ProxyException):
            await _guess(throttle, username=victim)

    for i in range(500):
        await throttle.record_failure(f"spray-filler-{i}@corp.com")

    assert await throttle._failures(throttle.username_cache, throttle._username_key(victim)) == 3, "the counter must survive a spray"
    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle, username=victim)
    assert blocked.value.code == "429"


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
                        throttle=_unlimited_throttle(),
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
                        throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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
                    throttle=_unlimited_throttle(),
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

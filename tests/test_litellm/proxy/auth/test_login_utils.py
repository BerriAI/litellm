"""
Tests for login_utils module.

This module tests the refactored login logic that was moved from proxy_server.py
to login_utils.py for better reusability.
"""

import os
from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from litellm.proxy.auth.login_throttle import LoginThrottle


def _unlimited_throttle():
    """A throttle wired to real in-memory stores with limits no test can reach."""
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    return LoginThrottle(
        client_ip="1.2.3.4",
        source_limit=None,
        user_limit=10_000,
        window_seconds=60,
        block_seconds=300,
        counters=InMemoryCache(),
        blocks=InMemoryCache(),
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
    user_limit: int = 2,
    source_limit: int | None = None,
    window_seconds: int = 60,
    block_seconds: int = 300,
    client_ip: str = "1.2.3.4",
    stores=None,
    redis_cache=None,
):
    """A throttle over real in-memory stores, so the tests exercise the true counters and blocks."""
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.proxy.auth.login_throttle import LoginThrottle

    counters, blocks = stores if stores is not None else (InMemoryCache(), InMemoryCache())
    return LoginThrottle(
        client_ip=client_ip,
        source_limit=source_limit,
        user_limit=user_limit,
        window_seconds=window_seconds,
        block_seconds=block_seconds,
        counters=counters,
        blocks=blocks,
        redis_cache=redis_cache,
    )


def _stores():
    from litellm.caching.in_memory_cache import InMemoryCache

    return InMemoryCache(), InMemoryCache()


async def _guess(throttle, username: str = "admin", password: str = "wrong"):
    from litellm.proxy.auth.login_utils import authenticate_user

    return await authenticate_user(
        username=username,
        password=password,
        master_key="sk-master",
        prisma_client=None,
        throttle=throttle,
    )


async def _fail(throttle, username: str = "admin") -> str:
    """One wrong guess; returns the status code it was answered with."""
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException) as exc:
        await _guess(throttle, username=username)
    return exc.value.code


def _known_user(email: str = "known@example.com"):
    user = MagicMock()
    user.user_id = "u-1"
    user.user_email = email
    user.user_role = "internal_user"
    user.password = "scrypt:stored"
    repo = MagicMock()
    repo.return_value.table.find_first = AsyncMock(return_value=user)
    return repo


async def _db_login(throttle, username: str, password: str, *, correct: bool):
    """A database user's sign-in with the stored hash faked, so no database or scrypt is needed."""
    from litellm.proxy.auth.login_utils import authenticate_user

    with (
        patch(  # test-quality-ok: the user lookup is the database boundary; faked so no DB is needed
            "litellm.proxy.auth.login_utils.UserRepository", _known_user(username)
        ),
        patch(  # test-quality-ok: reaches the known-DB-user branch without a database
            "litellm.proxy.auth.login_utils.verify_password", return_value=correct
        ),
        patch(  # test-quality-ok: the rehash writes to the database; faked so no DB is needed
            "litellm.proxy.auth.login_utils._rehash_password_if_needed", new=AsyncMock()
        ),
        patch(  # test-quality-ok: success mints a UI key; faked so no DB is needed
            "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
        ),
    ):
        return await authenticate_user(
            username=username, password=password, master_key="sk-master", prisma_client=MagicMock(), throttle=throttle
        )


def _local_count(throttle, key: str) -> int:
    return int(throttle.counters.get_cache(key) or 0)


@pytest.mark.asyncio
async def test_too_many_failures_for_one_username_block_that_pair_and_carry_retry_after(monkeypatch):
    """One failure past the pair limit blocks the source for that username; the next guess is answered 429
    with the block's remaining time, and the counter is not touched by blocked guesses."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2, block_seconds=77)
    keys = throttle._keys("admin")

    assert [await _fail(throttle) for _ in range(3)] == ["401", "401", "401"], "the limit itself is a plain 401"
    assert throttle._local_block_ttl(keys.pair_block) == 77

    with pytest.raises(ProxyException) as blocked:
        await _guess(throttle)
    assert blocked.value.code == "429"
    assert blocked.value.headers.get("Retry-After") == "77"
    assert _local_count(throttle, keys.pair_counter) == 3, "a blocked guess is not counted again"


@pytest.mark.asyncio
async def test_a_blocked_key_is_refused_before_the_password_is_looked_at(monkeypatch):
    """The block is the rate cap: once a key is blocked, nothing from it reaches the user lookup or the
    password check, so a guessing script gets no verification work out of the proxy."""
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.login_utils import authenticate_user

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=1)

    assert [await _fail(throttle, username="user@corp.com") for _ in range(3)] == ["401", "401", "429"]

    lookup = _known_user("user@corp.com")
    verify = MagicMock(return_value=True)
    with (
        patch(  # test-quality-ok: the user lookup is the database boundary; a blocked attempt must not reach it
            "litellm.proxy.auth.login_utils.UserRepository", lookup
        ),
        patch(  # test-quality-ok: the password check is the expensive step; a blocked attempt must not reach it
            "litellm.proxy.auth.login_utils.verify_password", verify
        ),
        pytest.raises(ProxyException) as refused,
    ):
        await authenticate_user(
            username="user@corp.com",
            password="right",
            master_key="sk-master",
            prisma_client=MagicMock(),
            throttle=throttle,
        )

    assert refused.value.code == "429"
    assert lookup.return_value.table.find_first.await_count == 0
    assert verify.call_count == 0


@pytest.mark.asyncio
async def test_a_correct_password_is_refused_while_its_pair_is_blocked(monkeypatch):
    """Letting the right password through would give a guesser unlimited tries, so the block is hard: the
    real user waits it out, or uses the master key over the API, which never passes through here."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=1, block_seconds=90)

    assert [await _fail(throttle, username="user@corp.com") for _ in range(3)] == ["401", "401", "429"]

    with pytest.raises(ProxyException) as refused:
        await _db_login(throttle, "user@corp.com", "right", correct=True)
    assert refused.value.code == "429"
    assert refused.value.headers.get("Retry-After") == "90"


@pytest.mark.asyncio
async def test_a_correct_password_is_refused_while_its_source_is_blocked(monkeypatch):
    """Same for the source-wide block: every username from that address is refused until it lapses."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=100, source_limit=2)

    for i in range(3):
        assert await _fail(throttle, username=f"other-{i}@corp.com") == "401"
    assert await _fail(throttle, username="other-9@corp.com") == "429", "the source is blocked for everyone"

    with pytest.raises(ProxyException) as refused:
        await _db_login(throttle, "user@corp.com", "right", correct=True)
    assert refused.value.code == "429"


@pytest.mark.asyncio
async def test_a_successful_sign_in_clears_the_pair_counter_but_not_the_source_counter(monkeypatch):
    """One account's success says nothing about the other guesses the address is making."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=5, source_limit=50)
    keys = throttle._keys("user@corp.com")

    for _ in range(2):
        assert await _fail(throttle, username="user@corp.com") == "401"
    assert _local_count(throttle, keys.pair_counter) == 2
    assert _local_count(throttle, keys.source_counter) == 2

    await _db_login(throttle, "user@corp.com", "right", correct=True)

    assert _local_count(throttle, keys.pair_counter) == 0
    assert _local_count(throttle, keys.source_counter) == 2


@pytest.mark.asyncio
async def test_once_a_pair_is_blocked_its_failures_stop_counting_against_the_source(monkeypatch):
    """A script stuck on one account trips the pair block and then leaves the office's shared address alone."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2, source_limit=4)
    keys = throttle._keys("stuck-script@corp.com")

    assert [await _fail(throttle, username="stuck-script@corp.com") for _ in range(3)] == ["401"] * 3
    assert _local_count(throttle, keys.source_counter) == 2, "failures before the pair block count for the source"

    for _ in range(5):
        assert await _fail(throttle, username="stuck-script@corp.com") == "429"
    assert _local_count(throttle, keys.source_counter) == 2, "blocked-pair failures must not reach the source"

    assert await _fail(throttle, username="colleague@corp.com") == "401", "a colleague still signs in normally"
    assert throttle._local_block_ttl(keys.source_block) == 0


@pytest.mark.asyncio
async def test_the_blocking_failure_itself_does_not_count_against_the_source(monkeypatch):
    """The guess that installs the pair block is the first one that stops counting, so a pair limit of B
    costs the source exactly B, not B plus one."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2, source_limit=2)
    keys = throttle._keys("stuck@corp.com")

    assert [await _fail(throttle, username="stuck@corp.com") for _ in range(3)] == ["401", "401", "401"]

    assert _local_count(throttle, keys.source_counter) == 2
    assert throttle._local_block_ttl(keys.source_block) == 0, "the third guess blocked the pair, not the source"


@pytest.mark.asyncio
async def test_too_many_failures_across_usernames_block_the_whole_source(monkeypatch):
    """A spray of one guess per username never trips a pair; the source counter is what stops it."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=5, source_limit=3, block_seconds=200)

    assert [await _fail(throttle, username=f"sprayed-{i}@corp.com") for i in range(4)] == ["401"] * 4

    assert await _fail(throttle, username="sprayed-99@corp.com") == "429"
    assert throttle._local_block_ttl(throttle._keys("x").source_block) == 200


@pytest.mark.asyncio
async def test_without_trusted_proxy_ranges_the_source_scope_is_off(monkeypatch):
    """Behind an ingress every client shares the peer address, so a source-wide block would block them all.
    The pair scope still applies."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    request = MagicMock()
    request.headers = {"x-forwarded-for": "203.0.113.9"}
    request.client = MagicMock()
    request.client.host = "10.0.0.1"
    throttle = LoginThrottle.from_request(
        request, general_settings={"max_failed_login_attempts_per_source": 1}, redis_cache=None
    )

    assert throttle.source_limit is None
    assert throttle.client_ip == "10.0.0.1", "the header is not trusted without a configured proxy range"
    assert [await _fail(throttle, username=f"user-{i}@corp.com") for i in range(6)] == ["401"] * 6


@pytest.mark.asyncio
async def test_an_empty_trusted_proxy_ranges_means_the_peer_is_the_client_and_the_source_scope_is_on(monkeypatch):
    """An explicit empty list says there are no proxies: the peer address is the client, the forwarded header
    is ignored, and the source-wide limit applies. Only an unset key means the topology is unknown."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    request = MagicMock()
    request.headers = {"x-forwarded-for": "203.0.113.9"}
    request.client = MagicMock()
    request.client.host = "198.51.100.7"
    throttle = LoginThrottle.from_request(
        request,
        general_settings={"trusted_proxy_ranges": [], "max_failed_login_attempts_per_source": 3},
        redis_cache=None,
    )

    assert throttle.client_ip == "198.51.100.7"
    assert throttle.source_limit == 3
    assert [await _fail(throttle, username=f"user-{i}@corp.com") for i in range(4)] == ["401"] * 4
    assert await _fail(throttle, username="user-99@corp.com") == "429", "the spray is stopped by the source limit"


@pytest.mark.parametrize(
    "configured",
    [
        None,
        5,
        {"10.0.0.0/8": True},
        ["", "  "],
        ["not-a-range"],
        ["10.0.0.0/8, 172.16.0.0/12"],
        ["10.0.0.0/8", "10.0.0.0/33"],
        "10.0.0.0/8;172.16.0.0/12",
    ],
)
def test_a_trusted_proxy_ranges_value_that_names_no_ranges_leaves_the_topology_unknown(configured):
    """Only a list of valid ranges or an explicit empty list counts as a declaration; anything else, including a
    list with one bad entry, is the same as unset, so a typo cannot switch the source-wide block on against
    the shared ingress address and lock out everyone behind it."""
    from litellm.proxy.auth.login_throttle import LoginThrottle, declared_proxy_ranges

    settings = {"trusted_proxy_ranges": configured} if configured is not None else {}
    assert declared_proxy_ranges(settings) is None

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "198.51.100.7"
    throttle = LoginThrottle.from_request(request, general_settings=settings, redis_cache=None)
    assert throttle.source_limit is None
    assert throttle.client_ip == "198.51.100.7"


def test_declared_proxy_ranges_distinguishes_none_from_empty_from_configured():
    from litellm.proxy.auth.login_throttle import declared_proxy_ranges

    assert declared_proxy_ranges({}) is None
    assert declared_proxy_ranges({"trusted_proxy_ranges": []}) == ()
    assert declared_proxy_ranges({"trusted_proxy_ranges": ["10.0.0.0/8", " 192.168.1.1 "]}) == (
        "10.0.0.0/8",
        "192.168.1.1",
    )
    assert declared_proxy_ranges({"trusted_proxy_ranges": "10.0.0.0/8,172.16.0.0/12"}) == (
        "10.0.0.0/8",
        "172.16.0.0/12",
    )


@pytest.mark.asyncio
async def test_with_trusted_proxy_ranges_the_source_is_the_forwarded_client(monkeypatch):
    """The header is walked right to left past the trusted hops, so a forged left-most entry cannot pick the bucket."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    settings = {"trusted_proxy_ranges": ["10.0.0.0/8"], "max_failed_login_attempts_per_source": 2}

    def _from(peer: str, forwarded: str):
        request = MagicMock()
        request.headers = {"x-forwarded-for": forwarded}
        request.client = MagicMock()
        request.client.host = peer
        return LoginThrottle.from_request(request, general_settings=settings, redis_cache=None)

    via_proxy = _from("10.0.0.1", "1.1.1.1, 203.0.113.9, 10.0.0.2")
    assert via_proxy.client_ip == "203.0.113.9"
    assert via_proxy.source_limit == 2

    direct = _from("198.51.100.7", "203.0.113.9")
    assert direct.client_ip == "198.51.100.7", "a peer outside the trusted ranges cannot forward anything"


def test_source_overrides_pick_the_most_specific_matching_range():
    """An exact address beats a /16 beats a /8; an address in none of them keeps the default."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    settings = {
        "trusted_proxy_ranges": ["10.0.0.0/8"],
        "max_failed_login_attempts_per_source": 7,
        "max_failed_login_attempts_per_source_overrides": {
            "203.0.0.0/8": 100,
            "203.0.113.0/24": 200,
            "203.0.113.9": 300,
            "not-an-address": 999,
            "198.51.100.0/24": "not-a-number",
        },
    }

    def _limit(client: str) -> int | None:
        request = MagicMock()
        request.headers = {"x-forwarded-for": client}
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        return LoginThrottle.from_request(request, general_settings=settings, redis_cache=None).source_limit

    assert _limit("203.0.113.9") == 300
    assert _limit("203.0.113.10") == 200
    assert _limit("203.0.1.1") == 100
    assert _limit("192.0.2.1") == 7
    assert _limit("198.51.100.1") == 7, "a garbage limit falls back to the default rather than a huge or zero budget"
    assert _limit("::ffff:203.0.113.9") == 300, "a mapped address gets the limit of the IPv4 bucket it is counted in"
    assert _limit("::ffff:203.0.113.10") == 200


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"203.0.113.7": 0, "203.0.113.7/32": 5}, None),
        ({"203.0.113.7/32": 5, "203.0.113.7": 0}, None),
        ({"203.0.113.0/24": 3, "203.0.113.9/24": 8}, 8),
        ({"203.0.113.9/24": 8, "203.0.113.0/24": 3}, 8),
    ],
    ids=["exact-then-slash32", "slash32-then-exact", "low-then-high", "high-then-low"],
)
def test_equivalent_override_keys_resolve_to_the_exemption_then_the_higher_limit(overrides, expected):
    """Two spellings of the same network are a config mistake, so precedence must not depend on dict order."""
    settings = {"trusted_proxy_ranges": ["10.0.0.0/8"], "max_failed_login_attempts_per_source_overrides": overrides}

    assert _throttle_behind_trusted_proxy("203.0.113.7", settings).source_limit == expected


def test_ipv6_sources_are_grouped_by_their_64_bit_prefix():
    """A /64 holder has 2^64 addresses; counting each one separately would hand them unlimited fresh buckets."""
    from litellm.proxy.auth.login_throttle import source_group

    assert source_group("2001:db8:1:2::1") == source_group("2001:db8:1:2:ffff:ffff:ffff:ffff") == "2001:db8:1:2::/64"
    assert source_group("2001:db8:1:3::1") != source_group("2001:db8:1:2::1")
    assert source_group("::ffff:203.0.113.9") == source_group("203.0.113.9") == "203.0.113.9"
    assert source_group("unknown") == "unknown"


@pytest.mark.asyncio
async def test_two_ipv6_addresses_in_one_64_share_the_source_budget(monkeypatch):
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    stores = _stores()
    first = _throttle(user_limit=50, source_limit=2, client_ip="2001:db8:1:2::1", stores=stores)
    second = _throttle(user_limit=50, source_limit=2, client_ip="2001:db8:1:2::2", stores=stores)

    assert [await _fail(first, username=f"a-{i}@corp.com") for i in range(3)] == ["401"] * 3
    assert await _fail(second, username="b@corp.com") == "429"


@pytest.mark.asyncio
async def test_one_source_being_blocked_does_not_touch_another(monkeypatch):
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    stores = _stores()
    attacker = _throttle(user_limit=50, source_limit=2, client_ip="203.0.113.9", stores=stores)
    neighbour = _throttle(user_limit=50, source_limit=2, client_ip="198.51.100.7", stores=stores)

    assert [await _fail(attacker, username=f"t-{i}@corp.com") for i in range(3)] == ["401"] * 3
    assert await _fail(attacker, username="t-9@corp.com") == "429"
    assert await _fail(neighbour, username="t-9@corp.com") == "401"


@pytest.mark.asyncio
async def test_the_same_username_from_another_source_has_its_own_budget(monkeypatch):
    """The pair carries the address on purpose: an attacker elsewhere cannot lock a user out of their own office."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    stores = _stores()
    attacker = _throttle(user_limit=1, client_ip="203.0.113.9", stores=stores)
    office = _throttle(user_limit=1, client_ip="198.51.100.7", stores=stores)

    assert [await _fail(attacker, username="victim@corp.com") for _ in range(3)] == ["401", "401", "429"]
    assert await _fail(office, username="victim@corp.com") == "401"


@pytest.mark.asyncio
async def test_the_counting_window_is_anchored_at_the_first_failure(monkeypatch):
    """Later failures must not push the expiry out, or a slow guesser keeps their own count alive forever."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=50, window_seconds=60)
    key = throttle._keys("admin").pair_counter

    await _fail(throttle)
    first_expiry = throttle.counters.ttl_dict[key]
    for _ in range(3):
        await _fail(throttle)

    assert throttle.counters.ttl_dict[key] == first_expiry


@pytest.mark.asyncio
async def test_the_block_outlives_the_counting_window(monkeypatch):
    """Counters expire after the window and blocks after the block time; the two are separate keys."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=1, window_seconds=10, block_seconds=300)
    keys = throttle._keys("admin")

    assert [await _fail(throttle) for _ in range(2)] == ["401", "401"]

    throttle.counters.delete_cache(keys.pair_counter)

    assert await _fail(throttle) == "429", "an expired counter must not lift an active block"
    assert 290 <= throttle._local_block_ttl(keys.pair_block) <= 300


@pytest.mark.asyncio
async def test_the_block_time_is_fixed_and_not_refreshed_by_blocked_guesses(monkeypatch):
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=1, block_seconds=300)
    key = throttle._keys("admin").pair_block

    assert [await _fail(throttle) for _ in range(2)] == ["401", "401"]
    installed_at = throttle.blocks.ttl_dict[key]

    for _ in range(4):
        assert await _fail(throttle) == "429"

    assert throttle.blocks.ttl_dict[key] == installed_at


@pytest.mark.asyncio
async def test_the_configured_admin_credentials_are_not_exempt_from_the_block(monkeypatch):
    """Exempting the env credentials would make them the one password worth guessing without limit, so the
    right UI_PASSWORD is refused while its pair is blocked, and signs in normally once the block lapses."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=1)

    assert [await _fail(throttle) for _ in range(3)] == ["401", "401", "429"]

    with (
        patch(  # test-quality-ok: the admin sign-in upserts the admin row; faked so no DB is needed
            "litellm.proxy.auth.login_utils.user_update", new=AsyncMock()
        ),
        patch(  # test-quality-ok: success mints a UI key and persists the user; faked so no DB is needed
            "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
        ),
    ):
        with pytest.raises(ProxyException) as refused:
            await _guess(throttle, password="right")
        assert refused.value.code == "429"

        throttle.blocks.delete_cache(throttle._keys("admin").pair_block)
        result = await _guess(throttle, password="right")
    assert result.key == "sk-ui"


@pytest.mark.asyncio
async def test_the_master_key_used_as_the_ui_password_is_not_exempt_from_the_block(monkeypatch):
    """Without UI_PASSWORD the master key doubles as the admin password; it gets no special treatment here
    either. Lockout recovery is the master key as a bearer token over the API, which never enters this path."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=1)

    assert [await _fail(throttle) for _ in range(3)] == ["401", "401", "429"]

    with pytest.raises(ProxyException) as refused:
        await _guess(throttle, password="sk-master")
    assert refused.value.code == "429"


@pytest.mark.asyncio
async def test_a_configuration_error_never_counts(monkeypatch):
    """A 500 from an unset master key is not a guess and must not consume the budget."""
    from litellm.proxy._types import ProxyException

    throttle = _throttle(user_limit=2)
    for _ in range(5):
        with pytest.raises(ProxyException) as exc:
            await authenticate_user(
                username="admin", password="x", master_key=None, prisma_client=None, throttle=throttle
            )
        assert exc.value.code == "500"

    assert _local_count(throttle, throttle._keys("admin").pair_counter) == 0


@pytest.mark.asyncio
async def test_the_username_is_case_folded_into_one_pair(monkeypatch):
    """The DB lookup is case-insensitive, so casing must not multiply the budget."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=4)

    for name in ("admin@corp.com", "ADMIN@corp.com", "Admin@corp.com", "aDmIn@corp.com", "admin@CORP.com"):
        assert await _fail(throttle, username=name) == "401"

    assert await _fail(throttle, username="admin@Corp.com") == "429"


@pytest.mark.asyncio
async def test_both_credential_rejections_are_indistinguishable(monkeypatch):
    """One message for the known and the unknown username, so responses do not enumerate."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")

    with pytest.raises(ProxyException) as unknown:
        await _guess(_throttle(user_limit=99), username="nobody@example.com")
    with pytest.raises(ProxyException) as known:
        await _db_login(_throttle(user_limit=99), "known@example.com", "wrong", correct=False)

    assert unknown.value.message == known.value.message
    assert "known@example.com" not in unknown.value.message + known.value.message


@pytest.mark.asyncio
async def test_a_user_with_no_password_set_does_not_consume_the_budget(monkeypatch):
    """That 401 is deterministic and guards no secret, so counting it would only let someone burn the pair."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2)

    passwordless = MagicMock()
    passwordless.user_id = "u-2"
    passwordless.user_email = "nopass@example.com"
    passwordless.user_role = "internal_user"
    passwordless.password = None
    repo = MagicMock()
    repo.return_value.table.find_first = AsyncMock(return_value=passwordless)

    with patch(  # test-quality-ok: reaches the passwordless-DB-user branch without a database
        "litellm.proxy.auth.login_utils.UserRepository", repo
    ):
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

    assert _local_count(throttle, throttle._keys("nopass@example.com").pair_counter) == 0


@pytest.mark.asyncio
async def test_a_wrong_password_for_a_known_user_also_counts(monkeypatch):
    """The database-user branch must charge the pair too, not just the unknown-user branch."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2)

    for _ in range(3):
        with pytest.raises(ProxyException) as rejected:
            await _db_login(throttle, "known@example.com", "wrong", correct=False)
        assert rejected.value.code == "401"

    with pytest.raises(ProxyException) as blocked:
        await _db_login(throttle, "known@example.com", "wrong", correct=False)
    assert blocked.value.code == "429"


@pytest.mark.asyncio
async def test_a_source_block_outranks_a_pair_block_in_the_retry_after(monkeypatch):
    """When both scopes are blocked, the answer carries the source block's time, which is the one that
    still applies to every other username from that address."""
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=1, source_limit=3, block_seconds=120, client_ip="203.0.113.45")
    assert [await _fail(throttle) for _ in range(2)] == ["401", "401"], "the admin pair is now blocked"
    throttle.blocks.set_cache(throttle._keys("admin").pair_block, 1, ttl=30)
    assert [await _fail(throttle, username=f"spray-{i}@corp.com") for i in range(3)] == ["401"] * 3
    assert throttle._local_block_ttl(throttle._keys("admin").source_block) == 120, "the source is now blocked too"

    for name in ("admin", "spray-0@corp.com", "never-seen@corp.com"):
        with pytest.raises(ProxyException) as refused:
            await _guess(throttle, username=name)
        assert refused.value.code == "429"
        assert refused.value.headers.get("Retry-After") == "120", name


@pytest.mark.asyncio
async def test_disabling_the_control_lets_every_attempt_through(monkeypatch):
    """The escape hatch has to turn off the whole control: no counting and no refusal."""
    import dataclasses

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = dataclasses.replace(_throttle(user_limit=1), enabled=False)

    assert [await _fail(throttle) for _ in range(6)] == ["401"] * 6
    assert _local_count(throttle, throttle._keys("admin").pair_counter) == 0


class _FakeRedis:
    """Redis whose only writes are the throttle's two scripts, run atomically as one call each.

    Mirrors the Lua: a blocked key returns its remaining block time and is not counted; a counter
    is expired on first write; one over the limit installs the block; a blocked pair stops the
    source from being counted. The real scripts are exercised against a live Redis in the PR's
    proof, this fake only has to be faithful enough for the worker-sharing tests.
    """

    def __init__(self):
        self.values: dict = {}
        self.ttls: dict = {}
        self.scripts: list[str] = []

    def async_register_script(self, script: str):
        from litellm.proxy.auth import login_throttle as lt

        async def _run(keys, args):
            self.scripts.append(script)
            if script == lt._BLOCK_TTLS_LUA:
                return [self._ttl(keys[1]), self._ttl(keys[3])]
            assert script == lt._RECORD_FAILURE_LUA
            user_limit, source_limit, window, block = (int(a) for a in args)
            user_block = self._bump(keys[0], keys[1], user_limit, window, block)
            if source_limit > 0 and user_block == 0:
                return [user_block, self._bump(keys[2], keys[3], source_limit, window, block)]
            return [user_block, 0]

        return _run

    def _ttl(self, key: str) -> int:
        return self.ttls.get(key, -2) if key in self.values else -2

    def _bump(self, count_key: str, block_key: str, limit: int, window: int, block: int) -> int:
        if self._ttl(block_key) > 0:
            return self._ttl(block_key)
        self.values[count_key] = self.values.get(count_key, 0) + 1
        self.ttls.setdefault(count_key, window)
        if self.values[count_key] > limit:
            self.values[block_key] = 1
            self.ttls[block_key] = block
            return block
        return 0

    async def async_delete_cache(self, key):
        self.values.pop(key, None)
        self.ttls.pop(key, None)


class _DownRedis(_FakeRedis):
    """Redis whose every call fails, as during an outage or an open circuit breaker."""

    def async_register_script(self, script: str):
        async def _run(keys, args):
            raise ConnectionError("redis is down")

        return _run

    async def async_delete_cache(self, key):
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_redis_is_the_only_counter_while_it_answers(monkeypatch):
    """Every worker must spend the same budget, see the same block, and a success must clear the pair for all."""
    from litellm.constants import LOGIN_THROTTLE_CACHE_KEY_PREFIX

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    redis = _FakeRedis()
    first_worker = _throttle(user_limit=2, stores=_stores(), redis_cache=redis)
    second_worker = _throttle(user_limit=2, stores=_stores(), redis_cache=redis)

    assert [await _fail(first_worker, username="user@corp.com") for _ in range(3)] == ["401"] * 3
    assert not [k for k in first_worker.counters.cache_dict if str(k).startswith(LOGIN_THROTTLE_CACHE_KEY_PREFIX)], (
        "with Redis answering, no worker may keep a counter of its own"
    )
    assert not first_worker.blocks.cache_dict

    assert await _fail(second_worker, username="user@corp.com") == "429", "the second worker sees the block"

    block_keys = [k for k in redis.values if ":block:user:" in k]
    assert block_keys, "the block lives in Redis, where every worker reads it"
    for key in block_keys:
        await redis.async_delete_cache(key)
    await _db_login(second_worker, "user@corp.com", "right", correct=True)

    assert not [k for k in redis.values if ":user:" in k and ":block:" not in k], (
        "success clears the shared pair counter"
    )


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_own_counter(monkeypatch):
    """With Redis raising, guesses are still counted and blocked per worker, with a warning, instead of unbounded."""
    import logging

    from litellm._logging import verbose_proxy_logger
    from litellm.proxy._types import ProxyException

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=2, block_seconds=300, redis_cache=_DownRedis())

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    verbose_proxy_logger.addHandler(handler)
    try:
        assert [await _fail(throttle) for _ in range(3)] == ["401"] * 3
        with pytest.raises(ProxyException) as blocked:
            await _guess(throttle)
    finally:
        verbose_proxy_logger.removeHandler(handler)

    assert blocked.value.code == "429"
    assert blocked.value.headers.get("Retry-After") == "300"
    assert any("Redis failed while counting Admin UI sign-in attempts" in r.getMessage() for r in records)


@pytest.mark.asyncio
async def test_a_failed_redis_delete_still_clears_this_workers_counter(monkeypatch):
    """The fail-open tradeoff: when Redis cannot clear the pair, the worker clears what it holds and moves on."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    throttle = _throttle(user_limit=5, redis_cache=_DownRedis())
    key = throttle._keys("user@corp.com").pair_counter

    assert [await _fail(throttle, username="user@corp.com") for _ in range(2)] == ["401", "401"]
    assert _local_count(throttle, key) == 2

    await _db_login(throttle, "user@corp.com", "right", correct=True)
    assert _local_count(throttle, key) == 0


@pytest.mark.asyncio
async def test_counters_do_not_share_the_key_authentication_cache(monkeypatch):
    """Regression: throttle entries must not evict cached credentials from user_api_key_cache."""
    from litellm.constants import LOGIN_THROTTLE_CACHE_KEY_PREFIX
    from litellm.proxy import proxy_server as ps
    from litellm.proxy.auth.login_throttle import LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    auth_cache_keys_before = set(ps.user_api_key_cache.in_memory_cache.cache_dict)

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"
    throttle = LoginThrottle.from_request(request, general_settings={}, redis_cache=None)

    for i in range(25):
        assert await _fail(throttle, username=f"made-up-{i}@example.com") == "401"

    added = set(ps.user_api_key_cache.in_memory_cache.cache_dict) - auth_cache_keys_before
    assert not [k for k in added if str(k).startswith(LOGIN_THROTTLE_CACHE_KEY_PREFIX)]


def test_settings_that_arrive_as_environment_strings_are_honored():
    """An `os.environ/VAR` reference in general_settings resolves to a string, not an int."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {"x-forwarded-for": "203.0.113.9"}
    request.client = MagicMock()
    request.client.host = "10.0.0.1"

    throttle = LoginThrottle.from_request(
        request,
        general_settings={
            "trusted_proxy_ranges": "10.0.0.0/8",
            "max_failed_login_attempts_per_source": " 70 ",
            "failed_login_window_seconds": "not-a-number",
            "failed_login_block_seconds": "-5",
        },
        redis_cache=None,
    )

    assert throttle.source_limit == 70
    assert throttle.user_limit == 35, "the per-username allowance is half the address allowance"
    assert throttle.window_seconds == 60, "garbage falls back to the default"
    assert throttle.block_seconds == 300, "a value below one would block nothing or forever"


def test_the_defaults_are_the_agreed_ones():
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {"x-forwarded-for": "203.0.113.9"}
    request.client = MagicMock()
    request.client.host = "10.0.0.1"
    throttle = LoginThrottle.from_request(
        request, general_settings={"trusted_proxy_ranges": ["10.0.0.0/8"]}, redis_cache=None
    )

    assert (throttle.source_limit, throttle.user_limit, throttle.window_seconds, throttle.block_seconds) == (
        10,
        5,
        60,
        300,
    )


@pytest.mark.parametrize(
    ("source_limit", "expected_user_limit"),
    [(1, 1), (2, 1), (3, 1), (10, 5), (11, 5), (70, 35)],
    ids=["one-stays-one", "two-halves-to-one", "odd-rounds-down", "default", "eleven-rounds-down", "even"],
)
def test_the_per_username_allowance_is_half_the_address_allowance_rounded_down_at_least_one(
    source_limit, expected_user_limit
):
    from litellm.proxy.auth.login_throttle import user_limit_for

    assert user_limit_for(source_limit) == expected_user_limit


def _throttle_behind_trusted_proxy(client_ip: str, settings: Mapping[str, object]) -> "LoginThrottle":
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {"x-forwarded-for": client_ip}
    request.client = MagicMock()
    request.client.host = "10.0.0.1"
    return LoginThrottle.from_request(request, general_settings=settings, redis_cache=None)


def test_a_per_address_override_also_raises_that_address_per_username_allowance():
    """One override sizes both limits for an address, so operators need no second override table."""
    settings = {
        "trusted_proxy_ranges": ["10.0.0.0/8"],
        "max_failed_login_attempts_per_source": 10,
        "max_failed_login_attempts_per_source_overrides": {"203.0.113.0/24": 50},
    }

    raised = _throttle_behind_trusted_proxy("203.0.113.9", settings)
    assert (raised.source_limit, raised.user_limit) == (50, 25)

    ordinary = _throttle_behind_trusted_proxy("198.51.100.4", settings)
    assert (ordinary.source_limit, ordinary.user_limit) == (10, 5)


@pytest.mark.asyncio
async def test_an_override_of_zero_exempts_that_address_from_both_limits():
    """Regression: opting an address out used to mean guessing a large enough number."""
    settings = {
        "trusted_proxy_ranges": ["10.0.0.0/8"],
        "max_failed_login_attempts_per_source": 1,
        "max_failed_login_attempts_per_source_overrides": {"203.0.113.7": 0, "203.0.113.0/24": 3},
    }

    exempt = _throttle_behind_trusted_proxy("203.0.113.7", settings)
    assert exempt.enabled is False
    assert exempt.source_limit is None
    attempt = await exempt.attempt("scanner@example.com")
    for _ in range(5):
        await attempt.failed()
    await exempt.attempt("scanner@example.com")

    sibling = _throttle_behind_trusted_proxy("203.0.113.8", settings)
    assert sibling.enabled is True
    assert (sibling.source_limit, sibling.user_limit) == (3, 1)


def test_the_per_username_allowance_follows_the_peer_override_when_the_source_scope_is_off():
    """Without trusted_proxy_ranges the address is not blocked, but its override still sizes the pair limit."""
    from litellm.proxy.auth.login_throttle import LoginThrottle

    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "192.0.2.8"
    throttle = LoginThrottle.from_request(
        request,
        general_settings={"max_failed_login_attempts_per_source_overrides": {"192.0.2.8": 40}},
        redis_cache=None,
    )

    assert throttle.source_limit is None
    assert throttle.user_limit == 20


def test_the_disable_flag_is_read_once_not_per_login_attempt(monkeypatch):
    """Regression: the kill switch was read through the secret manager on every unauthenticated request."""
    from litellm.proxy.auth import login_throttle

    reads: Final[list[str]] = []  # mutable-ok: test-only call recorder
    monkeypatch.setattr(
        login_throttle, "get_secret_bool", lambda name, default_value: reads.append(name) or default_value
    )
    login_throttle._rate_limit_disabled.cache_clear()
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "1.2.3.4"

    for _ in range(50):
        assert login_throttle.LoginThrottle.from_request(request, general_settings={}, redis_cache=None).enabled is True

    login_throttle._rate_limit_disabled.cache_clear()
    assert reads == ["LITELLM_DISABLE_LOGIN_RATE_LIMIT"]


@pytest.mark.asyncio
async def test_a_blocked_username_cannot_forge_log_lines(monkeypatch):
    """The username reaches a warning log, so it must not carry newlines or control bytes."""
    import logging

    from litellm._logging import verbose_proxy_logger

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    throttle = _throttle(user_limit=1)
    forged = "victim@example.com\nWARNING: sign-in succeeded for attacker\x00"

    assert await _fail(throttle, username=forged) == "401"

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    verbose_proxy_logger.addHandler(handler)
    try:
        assert await _fail(throttle, username=forged) == "401"
    finally:
        verbose_proxy_logger.removeHandler(handler)

    emitted = [r.getMessage() for r in records if "Admin UI sign-in blocked" in r.getMessage()]
    assert emitted, "installing the block must be logged"
    assert "\n" not in emitted[0] and "\x00" not in emitted[0]
    assert "victim@example.com" in emitted[0]


@pytest.mark.asyncio
async def test_a_username_spray_cannot_evict_an_active_block(monkeypatch):
    """Counters and blocks live in separate bounded stores, so a flood of made-up pairs fills the counter
    store while the blocks it already earned stay in force."""
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.constants import LOGIN_THROTTLE_MAX_TRACKED_BLOCKS, LOGIN_THROTTLE_MAX_TRACKED_COUNTERS
    from litellm.proxy.auth.login_throttle import _BLOCKS, _COUNTERS, LoginThrottle

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right")
    assert LOGIN_THROTTLE_MAX_TRACKED_COUNTERS >= 10_000 and LOGIN_THROTTLE_MAX_TRACKED_BLOCKS >= 10_000
    assert _COUNTERS is not _BLOCKS
    counters, blocks = InMemoryCache(max_size_in_memory=50), InMemoryCache(max_size_in_memory=50)
    throttle = LoginThrottle(
        client_ip="10.9.9.9",
        source_limit=None,
        user_limit=1,
        window_seconds=60,
        block_seconds=300,
        counters=counters,
        blocks=blocks,
    )
    victim = "spray-victim@corp.com"
    assert [await _fail(throttle, username=victim) for _ in range(2)] == ["401", "401"]

    for i in range(200):
        await throttle.record_failure(f"spray-filler-{i}@corp.com")

    assert len(counters.cache_dict) <= 50, "the counter store is bounded"
    assert counters.get_cache(throttle._keys(victim).pair_counter) is None, "the victim's counter was evicted"
    assert await _fail(throttle, username=victim) == "429", "the block survived the spray"


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

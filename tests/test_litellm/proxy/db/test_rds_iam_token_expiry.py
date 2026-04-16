"""
Tests for the RDS IAM token proactive refresh implementation.

Tests for GitHub Issue #16220: RDS IAM authentication connection failures after 15 minutes.

The fix implements:
1. Proactive background token refresh (refreshes 3 min before expiration)
2. Precise sleep timing (1 wake-up per token cycle instead of polling)
3. Proper locking during reconnection
4. Fixed __getattr__ fallback that now waits for reconnection

Run these tests:
    poetry run pytest tests/test_litellm/proxy/db/test_rds_iam_token_expiry.py -v -s
"""

import asyncio
import os
import urllib.parse
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestPrismaWrapperTokenRefresh:
    """Tests for the PrismaWrapper RDS IAM token refresh implementation."""

    @pytest.fixture
    def setup_env(self):
        """Setup environment variables for testing."""
        os.environ["DATABASE_HOST"] = "test-host.rds.amazonaws.com"
        os.environ["DATABASE_PORT"] = "5432"
        os.environ["DATABASE_USER"] = "test_user"
        os.environ["DATABASE_NAME"] = "test_db"
        os.environ["IAM_TOKEN_DB_AUTH"] = "True"
        yield
        # Cleanup
        for key in [
            "DATABASE_HOST",
            "DATABASE_PORT",
            "DATABASE_USER",
            "DATABASE_NAME",
            "DATABASE_URL",
            "IAM_TOKEN_DB_AUTH",
            "DATABASE_SCHEMA",
        ]:
            os.environ.pop(key, None)

    def _generate_mock_token(self, expires_in_seconds: int = 900) -> str:
        """Generate a mock IAM token with expiration info."""
        now = datetime.utcnow()
        date_str = now.strftime("%Y%m%dT%H%M%SZ")
        # Build the token like AWS does
        token = f"mock-token?X-Amz-Date={date_str}&X-Amz-Expires={expires_in_seconds}&X-Amz-Signature=abc123"
        return urllib.parse.quote(token, safe="")

    def _set_database_url_with_token(self, expires_in_seconds: int = 900):
        """Set DATABASE_URL with a mock token."""
        token = self._generate_mock_token(expires_in_seconds)
        os.environ[
            "DATABASE_URL"
        ] = f"postgresql://test_user:{token}@test-host:5432/test_db"

    @pytest.mark.asyncio
    async def test_is_token_expired_fresh(self, setup_env):
        """Test that fresh token is not detected as expired."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        self._set_database_url_with_token(expires_in_seconds=900)
        db_url = os.getenv("DATABASE_URL")

        assert wrapper.is_token_expired(db_url) is False

    @pytest.mark.asyncio
    async def test_is_token_expired_old(self, setup_env):
        """Test that old token is detected as expired."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        # Create an expired token
        old_date = datetime.utcnow() - timedelta(seconds=901)
        date_str = old_date.strftime("%Y%m%dT%H%M%SZ")
        token = (
            f"mock-token?X-Amz-Date={date_str}&X-Amz-Expires=900&X-Amz-Signature=abc"
        )
        encoded_token = urllib.parse.quote(token, safe="")
        db_url = f"postgresql://test_user:{encoded_token}@test-host:5432/test_db"

        assert wrapper.is_token_expired(db_url) is True

    @pytest.mark.asyncio
    async def test_start_stop_token_refresh_task(self, setup_env):
        """Test that token refresh task starts and stops correctly."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        # Set a valid token
        self._set_database_url_with_token(expires_in_seconds=900)

        # Start the task
        await wrapper.start_token_refresh_task()
        assert wrapper._token_refresh_task is not None
        assert not wrapper._token_refresh_task.done()

        # Stop the task
        await wrapper.stop_token_refresh_task()
        assert wrapper._token_refresh_task is None

    @pytest.mark.asyncio
    async def test_start_task_not_enabled(self, setup_env):
        """Test that task doesn't start when IAM auth is not enabled."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        # IAM auth disabled
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

        await wrapper.start_token_refresh_task()
        assert wrapper._token_refresh_task is None

    @pytest.mark.asyncio
    async def test_is_token_expired_null(self, setup_env):
        """Test that None token is treated as expired."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        assert wrapper.is_token_expired(None) is True


class TestTokenExpirationParsing:
    """Tests for token expiration parsing utilities."""

    def test_parse_token_expiration_valid(self):
        """Test parsing expiration from a valid token."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        # Create a token with known expiration
        token = "mock-token?X-Amz-Date=20240101T120000Z&X-Amz-Expires=900&X-Amz-Signature=abc"

        expiration = wrapper._parse_token_expiration(token)

        assert expiration is not None
        expected = datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=900)
        assert expiration == expected

    def test_parse_token_expiration_invalid(self):
        """Test that invalid token returns None."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        # Invalid tokens
        assert wrapper._parse_token_expiration(None) is None
        assert wrapper._parse_token_expiration("no-query-params") is None
        assert wrapper._parse_token_expiration("?missing=params") is None


class TestBackgroundRefreshLoop:
    """Tests for the background refresh loop timing."""

    @pytest.fixture
    def setup_env(self):
        """Setup environment variables for testing."""
        os.environ["DATABASE_HOST"] = "test-host.rds.amazonaws.com"
        os.environ["DATABASE_PORT"] = "5432"
        os.environ["DATABASE_USER"] = "test_user"
        os.environ["DATABASE_NAME"] = "test_db"
        yield
        # Cleanup
        for key in [
            "DATABASE_HOST",
            "DATABASE_PORT",
            "DATABASE_USER",
            "DATABASE_NAME",
            "DATABASE_URL",
        ]:
            os.environ.pop(key, None)

    @pytest.mark.asyncio
    async def test_calculate_seconds_fallback_when_no_url(self, setup_env):
        """Test that fallback is used when DATABASE_URL is not set."""
        from litellm.proxy.db.prisma_client import PrismaWrapper

        mock_prisma = MagicMock()
        wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

        # Don't set DATABASE_URL
        seconds = wrapper._calculate_seconds_until_refresh()

        # Should return fallback interval
        assert seconds == wrapper.FALLBACK_REFRESH_INTERVAL_SECONDS


# ============================================================================
# DEMONSTRATION SCRIPT
# ============================================================================


async def demonstrate_fix():
    """
    Demonstrates the fix for the RDS IAM token expiration bug.

    Shows how the proactive refresh prevents the 15-minute connection failure.
    """
    # Import the actual implementation
    try:
        from litellm.proxy.db.prisma_client import PrismaWrapper
    except ImportError:
        return

    # Setup mock environment
    os.environ["DATABASE_HOST"] = "mock-rds.region.rds.amazonaws.com"
    os.environ["DATABASE_PORT"] = "5432"
    os.environ["DATABASE_USER"] = "iam_user"
    os.environ["DATABASE_NAME"] = "litellm"

    # Create initial token (expires in 10 seconds for demo)
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%dT%H%M%SZ")
    token = f"mock-token?X-Amz-Date={date_str}&X-Amz-Expires=10&X-Amz-Signature=abc123"
    encoded_token = urllib.parse.quote(token, safe="")
    os.environ[
        "DATABASE_URL"
    ] = f"postgresql://iam_user:{encoded_token}@mock-rds:5432/litellm"

    # Create mock prisma client
    mock_prisma = MagicMock()

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=True)

    # Override buffer for faster demo
    wrapper.TOKEN_REFRESH_BUFFER_SECONDS = 3
    wrapper.FALLBACK_REFRESH_INTERVAL_SECONDS = 5
    _ = wrapper._calculate_seconds_until_refresh()  # Verify calculation works
    db_url = os.getenv("DATABASE_URL")
    is_expired = wrapper.is_token_expired(db_url)
    assert is_expired is False, "Fresh token should not be expired!"

    # Mock the _token_refresh_loop to prevent it from actually running
    async def mock_loop():
        try:
            await asyncio.sleep(1000)
        except asyncio.CancelledError:
            pass

    with patch.object(wrapper, "_token_refresh_loop", side_effect=mock_loop):
        await wrapper.start_token_refresh_task()
        await wrapper.stop_token_refresh_task()

    # Cleanup
    for key in [
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_USER",
        "DATABASE_NAME",
        "DATABASE_URL",
    ]:
        os.environ.pop(key, None)


if __name__ == "__main__":
    asyncio.run(demonstrate_fix())

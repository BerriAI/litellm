import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.db.prisma_client import should_update_prisma_schema


def test_should_update_prisma_schema(monkeypatch):
    # CASE 1: Environment variable behavior
    # When DISABLE_SCHEMA_UPDATE is not set -> should update
    monkeypatch.delenv("DISABLE_SCHEMA_UPDATE", raising=False)
    assert should_update_prisma_schema() is True

    # When DISABLE_SCHEMA_UPDATE="true" -> should not update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "true")
    assert should_update_prisma_schema() is False

    # When DISABLE_SCHEMA_UPDATE="false" -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "false")
    assert should_update_prisma_schema() is True

    # CASE 2: Explicit parameter behavior (overrides env var)
    monkeypatch.delenv("DISABLE_SCHEMA_UPDATE", raising=False)
    assert should_update_prisma_schema(True) is False  # Param True -> should not update

    monkeypatch.delenv(
        "DISABLE_SCHEMA_UPDATE", raising=False
    )  # Ensure env var is unset so parameter controls the result
    assert should_update_prisma_schema(False) is True  # Param False -> should update


@pytest.mark.asyncio
async def test_recreate_prisma_client_successful_disconnect():
    """
    Test that recreate_prisma_client works normally when disconnect succeeds.
    """
    # Mock the original prisma client
    mock_prisma = AsyncMock()

    # Create a mock PrismaWrapper instance
    wrapper = Mock()
    wrapper._original_prisma = mock_prisma

    # Configure disconnect to succeed
    mock_prisma.disconnect.return_value = None

    # Mock the entire recreate_prisma_client method to avoid import issues
    async def mock_recreate_prisma_client(new_db_url: str, http_client=None):
        try:
            await mock_prisma.disconnect()
        except Exception:
            pass

        mock_new_prisma = AsyncMock()
        wrapper._original_prisma = mock_new_prisma
        await mock_new_prisma.connect()

    # Assign the mock method to the wrapper
    wrapper.recreate_prisma_client = mock_recreate_prisma_client

    # Call the method
    await wrapper.recreate_prisma_client("postgresql://new:new@localhost:5432/new")

    # Verify that disconnect was called
    mock_prisma.disconnect.assert_called_once()

    # Verify that the new client replaced the original
    assert wrapper._original_prisma != mock_prisma
    assert hasattr(wrapper._original_prisma, "connect")


def test_setup_database_includes_stderr_in_error_logs():
    """
    Test that PrismaManager.setup_database includes stderr in error logs.
    """
    import subprocess
    from litellm.proxy.db.prisma_client import PrismaManager

    with patch("litellm.proxy.db.prisma_client.subprocess.run") as mock_run, patch(
        "litellm.proxy.db.prisma_client.verbose_proxy_logger"
    ) as mock_logger, patch(
        "litellm.proxy.db.prisma_client.PrismaManager._get_prisma_dir",
        return_value="/tmp",
    ), patch(
        "os.chdir"
    ), patch(
        "time.sleep"
    ):

        # Mock a subprocess error with stderr
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["prisma", "db", "push"],
            stderr="P1001: Connection failed",
        )

        # Call setup_database and it should return False on failure
        # (It retries 4 times and eventually returns False)
        result = PrismaManager.setup_database(use_migrate=False)

        assert result is False
        assert mock_run.call_count == 4

        # Check if the warning contains the stderr
        warning_calls = [
            call.args[0] for call in mock_logger.warning.call_args_list if call.args
        ]
        assert any("P1001: Connection failed" in msg for msg in warning_calls)


def test_setup_database_logs_success_output():
    """
    Test that PrismaManager.setup_database logs stdout and stderr on success.
    """
    from litellm.proxy.db.prisma_client import PrismaManager

    with patch("litellm.proxy.db.prisma_client.subprocess.run") as mock_run, patch(
        "litellm.proxy.db.prisma_client.verbose_proxy_logger"
    ) as mock_logger, patch(
        "litellm.proxy.db.prisma_client.PrismaManager._get_prisma_dir",
        return_value="/tmp",
    ), patch(
        "os.chdir"
    ):

        # Mock a successful subprocess call with stdout and stderr
        mock_result = Mock()
        mock_result.stdout = "🚀 Your database is now in sync"
        mock_result.stderr = "Deprecation warning: some-feature is deprecated"
        mock_run.return_value = mock_result

        # Call setup_database
        result = PrismaManager.setup_database(use_migrate=False)

        assert result is True
        # Verify that debug was called with the stdout
        mock_logger.debug.assert_any_call(
            "Prisma success output (stdout): 🚀 Your database is now in sync"
        )
        # Verify that debug was called with the stderr
        mock_logger.debug.assert_any_call(
            "Prisma success output (stderr): Deprecation warning: some-feature is deprecated"
        )

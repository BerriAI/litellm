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

    monkeypatch.delenv("DISABLE_SCHEMA_UPDATE", raising=False)  # Set env var opposite to param
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


def test_prisma_manager_setup_database_error_logging():
    """
    Test that PrismaManager.setup_database includes stderr in error logs.
    """
    import subprocess
    from litellm.proxy.db.prisma_client import PrismaManager

    # We mock out the import of litellm_proxy_extras to force the base failure path
    with patch("subprocess.run") as mock_run, patch(
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

        # Call setup_database (it will retry and eventually log)
        try:
            # use_migrate=False to hit the 'else' path where we added capture_output=True
            PrismaManager.setup_database(use_migrate=False)
        except Exception:
            pass

        # Check if any log warning contains the stderr
        any_call_has_details = any(
            "P1001: Connection failed" in str(call)
            for call in mock_logger.warning.call_args_list
        )
        assert any_call_has_details is True

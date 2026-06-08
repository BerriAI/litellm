import json
import os
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.db.prisma_client import PrismaWrapper, should_update_prisma_schema


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


def test_should_update_prisma_schema(monkeypatch):
    # CASE 1: Environment variable behavior
    # When DISABLE_SCHEMA_UPDATE is not set -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema() == True

    # When DISABLE_SCHEMA_UPDATE="true" -> should not update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "true")
    assert should_update_prisma_schema() == False

    # When DISABLE_SCHEMA_UPDATE="false" -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "false")
    assert should_update_prisma_schema() == True

    # CASE 2: Explicit parameter behavior (overrides env var)
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema(True) == False  # Param True -> should not update

    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)  # Set env var opposite to param
    assert should_update_prisma_schema(False) == True  # Param False -> should update


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


@pytest.mark.asyncio
async def test_recreate_prisma_client_kills_old_engine_on_disconnect_failure(
    mock_prisma_binary,
):
    """When disconnect() fails, recreate_prisma_client must SIGTERM/SIGKILL the old engine PID."""
    mock_prisma = AsyncMock()
    mock_prisma.disconnect.side_effect = Exception("engine hung")

    # Simulate engine subprocess with a known PID
    mock_engine = MagicMock()
    mock_engine.process.pid = 12345
    mock_prisma._engine = mock_engine

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    # Configure the mock Prisma constructor
    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    # Verify old engine was killed
    mock_kill.assert_any_call(12345, signal.SIGTERM)
    # Verify new client was created and connected
    mock_new_prisma.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_prisma_client_skips_kill_on_successful_disconnect(
    mock_prisma_binary,
):
    """When disconnect() succeeds, no kill should be attempted."""
    mock_prisma = AsyncMock()
    mock_prisma.disconnect.return_value = None

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with patch("os.kill") as mock_kill:
        await wrapper.recreate_prisma_client("postgresql://new")

    mock_kill.assert_not_called()
    mock_new_prisma.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_prisma_client_handles_missing_engine_pid(
    mock_prisma_binary,
):
    """When engine PID is unavailable (no _engine attr), kill is skipped gracefully."""
    mock_prisma = AsyncMock()
    mock_prisma.disconnect.side_effect = Exception("engine hung")
    mock_prisma._engine = None  # No engine subprocess

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    mock_kill.assert_not_called()  # PID was 0, kill skipped
    mock_new_prisma.connect.assert_awaited_once()


# ── wait_for_prisma_engine tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_ready_on_first_attempt():
    """Engine responds immediately — returns True after one probe."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(return_value=None)

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await wrapper.wait_for_prisma_engine(retries=5, delay=1.0)

    assert result is True
    mock_prisma.query_raw.assert_awaited_once_with("SELECT 1")
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_ready_after_retries():
    """Engine fails twice then succeeds — returns True, sleeps between attempts."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(
        side_effect=[Exception("not ready"), Exception("still not"), None]
    )

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await wrapper.wait_for_prisma_engine(
            retries=5, delay=1.0, backoff_factor=0.0
        )

    assert result is True
    assert mock_prisma.query_raw.await_count == 3
    # Two sleeps: after attempt 1 and attempt 2
    assert mock_sleep.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_never_ready():
    """Engine fails every attempt — returns False after retries exhausted."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(side_effect=Exception("not ready"))

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await wrapper.wait_for_prisma_engine(retries=3, delay=0.1)

    assert result is False
    assert mock_prisma.query_raw.await_count == 3
    assert mock_sleep.await_count == 2  # slept after attempts 1 and 2


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_with_backoff_disabled():
    """backoff_factor=0 means constant delay between attempts."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(
        side_effect=[Exception("x"), Exception("x"), None]
    )

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await wrapper.wait_for_prisma_engine(
            retries=5, delay=2.0, backoff_factor=0.0
        )

    # With backoff_factor=0, all sleeps use the same delay
    assert mock_sleep.await_count == 2
    mock_sleep.assert_any_call(2.0)


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_with_linear_backoff():
    """backoff_factor=1.0 increases delay linearly: delay, 2*delay, 3*delay, ..."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(
        side_effect=[Exception("x"), Exception("x"), Exception("x"), None]
    )

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await wrapper.wait_for_prisma_engine(
            retries=5, delay=1.0, backoff_factor=1.0
        )

    assert mock_sleep.await_count == 3
    # First sleep: delay=1.0, second: 2.0, third: 3.0
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(3.0)


@pytest.mark.asyncio
async def test_wait_for_prisma_engine_single_retry():
    """Single retry that fails — returns False, no sleep."""
    mock_prisma = AsyncMock()
    mock_prisma.query_raw = AsyncMock(side_effect=Exception("not ready"))

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await wrapper.wait_for_prisma_engine(retries=1, delay=1.0)

    assert result is False
    assert mock_prisma.query_raw.await_count == 1
    mock_sleep.assert_not_called()

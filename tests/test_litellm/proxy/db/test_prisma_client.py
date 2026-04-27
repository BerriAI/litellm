import json
import os
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_recreate_prisma_client_kills_old_engine_without_disconnect(
    mock_prisma_binary,
):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/26191.

    recreate_prisma_client must kill the old engine subprocess directly
    and must NOT call disconnect() on the underlying Prisma client, which
    blocks the asyncio event loop via a synchronous `process.wait()`.
    """
    mock_prisma = AsyncMock()
    # disconnect is never expected to be called — fail loudly if it is.
    mock_prisma.disconnect.side_effect = AssertionError(
        "disconnect would block the event loop"
    )

    # Simulate engine subprocess with a known PID
    mock_engine = MagicMock()
    mock_engine.process.pid = 12345
    mock_prisma._engine = mock_engine

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    mock_kill.assert_any_call(12345, signal.SIGTERM)
    mock_prisma.disconnect.assert_not_called()
    mock_new_prisma.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_prisma_client_handles_missing_engine_pid(
    mock_prisma_binary,
):
    """When engine PID is unavailable (no _engine attr), kill is skipped gracefully."""
    mock_prisma = AsyncMock()
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

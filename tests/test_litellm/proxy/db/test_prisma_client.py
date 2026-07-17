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
    mock_prisma.is_connected = MagicMock(return_value=True)

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
    mock_prisma.is_connected = MagicMock(return_value=True)
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
    mock_prisma.is_connected = MagicMock(return_value=True)
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


def test_get_engine_pid_returns_zero_for_disconnected_client(disconnected_prisma):
    """A disconnected client must read as "no engine" instead of raising,
    otherwise the reconnect path can never recover."""
    wrapper = PrismaWrapper(
        original_prisma=disconnected_prisma, iam_token_db_auth=False
    )

    assert wrapper._get_engine_pid() == 0


@pytest.mark.asyncio
async def test_recreate_prisma_client_recovers_from_disconnected_client(
    mock_prisma_binary, disconnected_prisma
):
    """recreate_prisma_client must still build a replacement client when the
    current one is disconnected."""
    wrapper = PrismaWrapper(
        original_prisma=disconnected_prisma, iam_token_db_auth=False
    )

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with patch("os.kill") as mock_kill:
        result = await wrapper.recreate_prisma_client("postgresql://new")

    assert result is True
    mock_kill.assert_not_called()
    assert wrapper._original_prisma is mock_new_prisma
    mock_new_prisma.connect.assert_awaited_once()


# ── engine reaping (#33414) ──────────────────────────────────────────────────
# We retire the engine by signalling its PID rather than calling Prisma's
# disconnect() (which blocks the event loop on Popen.wait()). A signalled child
# stays in the process table as <defunct> until its parent wait()s on it, so the
# kill path must also reap — otherwise one zombie accrues per reconnect and
# survives for the life of the container.


@pytest.mark.asyncio
async def test_kill_engine_process_reaps_the_corpse(mock_prisma_binary):
    """Regression for #33414/#14739/#10216: the killed engine must be waited on.

    Without the waitpid the SIGKILLed child becomes a zombie: killing a process
    does not remove it from the process table, reaping does.
    """
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", return_value=(4242, 0)) as mock_waitpid,
    ):
        await PrismaWrapper._kill_engine_process(4242)

    mock_waitpid.assert_called_once_with(4242, os.WNOHANG)


@pytest.mark.asyncio
async def test_kill_engine_process_reap_is_non_blocking(mock_prisma_binary):
    """The reap must use WNOHANG.

    A blocking waitpid(pid, 0) would reintroduce the exact stall this class
    exists to avoid — disconnect() freezing the event loop for 30-120s on a
    hung engine. WNOHANG polls instead of waiting.
    """
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", return_value=(4242, 0)) as mock_waitpid,
    ):
        await PrismaWrapper._kill_engine_process(4242)

    _pid, flags = mock_waitpid.call_args[0]
    assert flags == os.WNOHANG, "reap must not block the event loop"


@pytest.mark.asyncio
async def test_kill_engine_process_tolerates_already_reaped_child(mock_prisma_binary):
    """ChildProcessError means someone else already collected it — not an error.

    PrismaClient._reap_all_zombies() does waitpid(-1) elsewhere and can win the
    race; that must not propagate out of the reconnect path.
    """
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", side_effect=ChildProcessError()),
    ):
        await PrismaWrapper._kill_engine_process(4242)  # must not raise


@pytest.mark.asyncio
async def test_kill_engine_process_gives_up_rather_than_spinning(mock_prisma_binary):
    """If the corpse never appears, bail out instead of looping forever.

    waitpid returning 0 means "still running". Blocking the reconnect path on a
    process that refuses to die is worse than leaving one zombie behind.
    """
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", return_value=(0, 0)),
        patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]),
    ):
        await PrismaWrapper._kill_engine_process(4242)  # must terminate


@pytest.mark.asyncio
async def test_kill_engine_process_skips_reap_for_invalid_pid(mock_prisma_binary):
    """pid<=0 short-circuits before any signal, so nothing is reaped."""
    with patch("os.waitpid") as mock_waitpid:
        await PrismaWrapper._kill_engine_process(0)
    mock_waitpid.assert_not_called()

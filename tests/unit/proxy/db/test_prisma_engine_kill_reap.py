"""Engine-reap coverage for PrismaWrapper._kill_engine_process (#33414)."""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


@pytest.mark.asyncio
async def test_kill_engine_process_reaps_the_corpse(mock_prisma_binary):
    """Regression for #33414/#14739/#10216: the killed engine must be waited on."""
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", return_value=(4242, 0)) as mock_waitpid,
    ):
        await PrismaWrapper._kill_engine_process(4242)

    mock_waitpid.assert_called_once_with(4242, os.WNOHANG)


@pytest.mark.asyncio
async def test_kill_engine_process_reap_is_non_blocking(mock_prisma_binary):
    """The reap must use WNOHANG so reconnect never blocks the event loop."""
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
    """ChildProcessError means someone else already collected it — not an error."""
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", side_effect=ChildProcessError()),
    ):
        await PrismaWrapper._kill_engine_process(4242)  # must not raise


@pytest.mark.asyncio
async def test_kill_engine_process_gives_up_rather_than_spinning(mock_prisma_binary):
    """If the corpse never appears, bail out instead of looping forever."""
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


@pytest.mark.asyncio
async def test_kill_engine_process_no_reap_on_windows(mock_prisma_binary):
    """Windows has no POSIX zombies and no os.WNOHANG, so the reap is a no-op."""
    import litellm.proxy.db.prisma_client as pc

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.object(pc.os, "WNOHANG", None, create=False),
        patch("os.waitpid") as mock_waitpid,
    ):
        await PrismaWrapper._kill_engine_process(4242)

    mock_waitpid.assert_not_called()


@pytest.mark.asyncio
async def test_kill_engine_process_swallows_oserror_from_waitpid(mock_prisma_binary):
    """A non-ECHILD OSError from waitpid must not abort reconnect."""
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", side_effect=OSError("EINVAL")),
    ):
        await PrismaWrapper._kill_engine_process(4242)  # must not raise

"""Engine-reap coverage for PrismaWrapper._kill_engine_process (#33414)."""
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


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


@pytest.mark.asyncio
async def test_kill_engine_process_no_reap_on_windows(mock_prisma_binary):
    """Windows has no POSIX zombies and no os.WNOHANG, so the reap is a no-op.

    The kill still happens (best-effort), but waitpid must not be attempted —
    calling it without WNOHANG would be meaningless here.
    """
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
    """A non-ECHILD OSError from waitpid (e.g. EINVAL) must not escape.

    This runs inside the reconnect path; any exception here would abort the
    reconnect that the kill was clearing the way for.
    """
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("os.waitpid", side_effect=OSError("EINVAL")),
    ):
        await PrismaWrapper._kill_engine_process(4242)  # must not raise

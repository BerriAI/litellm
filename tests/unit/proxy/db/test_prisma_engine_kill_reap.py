"""Engine-reap coverage for PrismaWrapper._kill_engine_process (#33414).

These tests drive real child processes so they assert what the reconnect path
actually leaves behind in the process table, not which syscalls it made.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX zombies only exist on POSIX")


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


@pytest.fixture
def engine_stand_in() -> Iterator[subprocess.Popen[bytes]]:
    """A long-lived child standing in for prisma-query-engine."""
    proc = subprocess.Popen(["sleep", "60"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _is_unreaped_child(pid: int) -> bool:
    """True while `pid` is still our child in the process table (alive or zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.asyncio
async def test_kill_engine_process_leaves_no_zombie(engine_stand_in):
    """Regression for #33414/#14739/#10216: the killed engine must not linger as <defunct>."""
    pid = engine_stand_in.pid

    await PrismaWrapper._kill_engine_process(pid)

    assert not _is_unreaped_child(pid)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


@pytest.mark.asyncio
async def test_reap_tolerates_already_reaped_child(engine_stand_in):
    """ChildProcessError means someone else already collected it, which is not an error."""
    engine_stand_in.kill()
    engine_stand_in.wait()

    assert await PrismaWrapper._reap_engine_process(engine_stand_in.pid) is None


@pytest.mark.asyncio
async def test_reap_gives_up_on_a_live_child_instead_of_blocking(engine_stand_in):
    """A child that never exits is left alone after the bounded wait, not waited on forever."""
    await PrismaWrapper._reap_engine_process(engine_stand_in.pid)

    assert engine_stand_in.poll() is None, "a live engine must not be killed by the reap"


@pytest.mark.asyncio
async def test_reap_is_a_no_op_without_wnohang(engine_stand_in):
    """Windows has no os.WNOHANG, so the reap must leave the process table untouched."""
    import litellm.proxy.db.prisma_client as pc

    engine_stand_in.kill()
    with patch.object(pc.os, "WNOHANG", None, create=False):
        await PrismaWrapper._reap_engine_process(engine_stand_in.pid)

    assert _is_unreaped_child(engine_stand_in.pid)
    assert engine_stand_in.wait() != 0


@pytest.mark.asyncio
async def test_reap_swallows_unexpected_oserror_from_waitpid():
    """A non-ECHILD OSError from waitpid must not abort reconnect."""
    with patch("os.waitpid", side_effect=OSError("EINVAL")):
        assert await PrismaWrapper._reap_engine_process(4242) is None

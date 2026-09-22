"""``process_is_gone`` has to say gone for every shape a killed process can take, and never for a live one."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from tests._process_helpers import process_is_gone

SLEEP_FOREVER: Final = (sys.executable, "-I", "-c", "import time; time.sleep(600)")

LEAVE_A_ZOMBIE_BEHIND: Final = """
import os, signal, subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
os.kill(grandchild.pid, signal.SIGKILL)
while open(f"/proc/{grandchild.pid}/stat").read().rpartition(")")[2].split()[0] != "Z":
    time.sleep(0.01)
print(grandchild.pid, flush=True)
time.sleep(600)
"""


def test_a_live_process_is_not_gone() -> None:
    child: Final = subprocess.Popen(SLEEP_FOREVER)
    try:
        assert not process_is_gone(child.pid, within_seconds=0.3)
    finally:
        child.kill()
        child.wait()


def test_a_reaped_child_is_gone() -> None:
    child: Final = subprocess.Popen(SLEEP_FOREVER)
    child.kill()
    child.wait()
    assert process_is_gone(child.pid, within_seconds=1)


@pytest.mark.skipif(os.name == "nt", reason="zombies are a POSIX thing")
def test_an_unreaped_child_is_reaped_and_gone() -> None:
    child: Final = subprocess.Popen(SLEEP_FOREVER)
    os.kill(child.pid, signal.SIGKILL)
    assert process_is_gone(child.pid, within_seconds=1)
    with pytest.raises(ChildProcessError):
        os.waitpid(child.pid, os.WNOHANG)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs to see a zombie that is not our child")
def test_a_zombie_left_by_another_process_is_gone() -> None:
    zombie_factory: Final = subprocess.Popen(
        [sys.executable, "-I", "-c", LEAVE_A_ZOMBIE_BEHIND], stdout=subprocess.PIPE, text=True
    )
    try:
        assert zombie_factory.stdout is not None
        zombie_pid: Final = int(zombie_factory.stdout.readline())
        with pytest.raises(ChildProcessError):
            os.waitpid(zombie_pid, os.WNOHANG)
        assert process_is_gone(zombie_pid, within_seconds=1)
    finally:
        zombie_factory.kill()
        zombie_factory.wait()

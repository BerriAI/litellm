import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Final

import pytest

from litellm.litellm_core_utils.process_liveness import pid_exists

_SLEEP_FOREVER: Final = [sys.executable, "-I", "-c", "import time; time.sleep(120)"]


@pytest.fixture
def live_child() -> Iterator[subprocess.Popen[bytes]]:
    child: Final = subprocess.Popen(_SLEEP_FOREVER)
    yield child
    child.kill()
    child.wait(timeout=30)


def test_current_process_exists() -> None:
    assert pid_exists(os.getpid()) is True


def test_checking_a_live_process_reports_it_alive_and_leaves_it_running(
    live_child: subprocess.Popen[bytes],
) -> None:
    assert pid_exists(live_child.pid) is True
    with pytest.raises(subprocess.TimeoutExpired):
        live_child.wait(timeout=1)
    assert pid_exists(live_child.pid) is True


def test_exited_process_does_not_exist() -> None:
    child: Final = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    child.wait(timeout=30)
    assert pid_exists(child.pid) is False


def test_process_we_may_not_inspect_still_exists() -> None:
    assert pid_exists(4 if sys.platform == "win32" else 1) is True

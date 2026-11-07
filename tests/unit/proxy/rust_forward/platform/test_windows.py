import signal
import subprocess
import sys

import pytest

from litellm.proxy.rust_forward.platform.windows import CREATE_NEW_PROCESS_GROUP, CTRL_BREAK_EVENT, SIGBREAK


@pytest.mark.skipif(sys.platform != "win32", reason="these names exist only on Windows")
def test_windows_constants_match_the_stdlib() -> None:
    assert (CTRL_BREAK_EVENT, SIGBREAK) == (signal.CTRL_BREAK_EVENT, signal.SIGBREAK)
    assert CREATE_NEW_PROCESS_GROUP == subprocess.CREATE_NEW_PROCESS_GROUP

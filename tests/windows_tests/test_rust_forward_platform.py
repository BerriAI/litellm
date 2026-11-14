import signal
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.platform.windows import (
    CREATE_NEW_PROCESS_GROUP,
    CTRL_BREAK_EVENT,
    SIGBREAK,
    WindowsPlatform,
)
from litellm.proxy.rust_forward.supervisor.preflight import gateway_binary


@pytest.mark.skipif(sys.platform != "win32", reason="these names exist only on Windows")
def test_windows_constants_match_the_stdlib() -> None:
    assert (CTRL_BREAK_EVENT, SIGBREAK) == (signal.CTRL_BREAK_EVENT, signal.SIGBREAK)
    assert CREATE_NEW_PROCESS_GROUP == subprocess.CREATE_NEW_PROCESS_GROUP


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows searches the working directory for an executable")
def test_gateway_binary_in_the_working_directory_is_never_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: Final = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "litellm-gateway.exe").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert gateway_binary(WindowsPlatform(), str(scripts)) is None
    installed: Final = scripts / "litellm-gateway.exe"
    installed.write_bytes(b"")
    assert gateway_binary(WindowsPlatform(), str(scripts)) == str(installed)

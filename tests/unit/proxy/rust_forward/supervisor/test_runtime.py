import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.platform import Platform, current_platform
from litellm.proxy.rust_forward.platform.windows import WindowsPlatform
from litellm.proxy.rust_forward.supervisor import supervise
from tests.unit.proxy.rust_forward.supervisor._support import python_server


PYTHON_CHILD: Final = """
import json, os, pathlib, socket, time
time.sleep(0.5)
pathlib.Path(os.environ["MARKERS"], "listening").touch()
private = json.loads(os.environ["_LITELLM_GATEWAY_SUPERVISED"])["listener"]
listener = socket.socket(fileno=private["fd"]) if "fd" in private else socket.create_server(("127.0.0.1", private["port"]))
listener.listen()
time.sleep(10)
"""
RUST_CHILD: Final = """
import os, pathlib
markers = pathlib.Path(os.environ["MARKERS"])
(markers / "rust_saw_python_listening").write_text(str((markers / "listening").exists()))
"""


def _fake_gateway(scripts: Path) -> Platform:
    if sys.platform != "win32":
        gateway: Final = scripts / "litellm-gateway"
        gateway.write_text(f"#!{sys.executable} -I\n{RUST_CHILD}")
        gateway.chmod(0o755)
        return current_platform()
    script: Final = scripts / "rust_child.py"
    script.write_text(RUST_CHILD)
    (scripts / "litellm-gateway.cmd").write_text(f'@"{sys.executable}" -I "{script}"\r\n')
    return WindowsPlatform(gateway_name="litellm-gateway.cmd")


def test_rust_opens_the_public_port_only_once_python_listens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKERS", str(tmp_path))
    command: Final = (sys.executable, "-I", "-c", PYTHON_CHILD)
    supervise(python_server(), command, scripts_dir=str(tmp_path), platform=_fake_gateway(tmp_path))
    assert (tmp_path / "rust_saw_python_listening").read_text() == "True"

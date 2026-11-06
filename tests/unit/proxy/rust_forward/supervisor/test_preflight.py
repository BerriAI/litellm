import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.platform.posix import PosixPlatform
from litellm.proxy.rust_forward.platform.windows import WindowsPlatform
from litellm.proxy.rust_forward.server import Listener, PythonServer, Server
from litellm.proxy.rust_forward.supervisor import supervise
from litellm.proxy.rust_forward.supervisor.preflight import StartupError, gateway_binary, startup_check
from tests.unit.proxy.rust_forward.supervisor._support import python_server

POSIX: Final = PosixPlatform()
WINDOWS: Final = WindowsPlatform()
POSIX_ONLY: Final = pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shebang gateway")


@pytest.mark.parametrize(
    ("server", "reason"),
    [
        (python_server(Server.HYPERCORN), "LITELLM_RUST does not support Hypercorn"),
        (python_server(Server.GRANIAN), "LITELLM_RUST does not support Granian"),
        (python_server(listener=Listener.TLS), "LITELLM_RUST does not support TLS on the public port"),
        (
            python_server(listener=Listener.UNIX_SOCKET),
            "LITELLM_RUST does not support a Unix socket on the public port",
        ),
        (
            python_server(Server.GUNICORN, Listener.INHERITED_FD),
            "LITELLM_RUST does not support an inherited file descriptor on the public port",
        ),
    ],
)
def test_unsupported_servers_are_refused_before_anything_starts(
    server: PythonServer, reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert startup_check(POSIX, server, "gateway") == StartupError(reason)
    prepared: Final[list[bool]] = []
    refused: Final = supervise(server, ("python",), before_start=lambda: prepared.append(True), platform=POSIX)
    assert refused == StartupError(reason)
    assert (capsys.readouterr().err, prepared) == (
        f"{reason}, so Python serves the public port without Rust forwarding\n",
        [],
    )


@pytest.mark.parametrize("server", [Server.UVICORN, Server.GUNICORN])
def test_plain_tcp_uvicorn_and_gunicorn_can_be_forwarded(server: Server) -> None:
    assert startup_check(POSIX, python_server(server), "gateway") == "gateway"


def test_windows_forwards_uvicorn_and_refuses_gunicorn() -> None:
    assert startup_check(WINDOWS, python_server(), "gateway") == "gateway"
    assert startup_check(WINDOWS, python_server(Server.GUNICORN), "gateway") == StartupError(
        "LITELLM_RUST does not support gunicorn on Windows"
    )


def test_missing_binary_is_refused_before_anything_starts(tmp_path: Path) -> None:
    prepared: Final[list[bool]] = []
    refused: Final = supervise(
        python_server(),
        ("python",),
        before_start=lambda: prepared.append(True),
        scripts_dir=str(tmp_path),
        platform=POSIX,
    )
    assert (refused, prepared) == (
        StartupError("LITELLM_RUST requires the litellm-gateway binary from the litellm-proxy package"),
        [],
    )


@POSIX_ONLY
def test_binary_is_found_only_in_the_environment_scripts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: Final = tmp_path / "scripts"
    elsewhere: Final = tmp_path / "elsewhere"
    scripts.mkdir()
    elsewhere.mkdir()
    binary: Final = elsewhere / "litellm-gateway"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(elsewhere))
    assert gateway_binary(POSIX, str(scripts)) is None
    assert gateway_binary(POSIX, str(elsewhere)) == str(binary)

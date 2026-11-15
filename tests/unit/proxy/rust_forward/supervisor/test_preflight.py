import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.platform import Platform
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
    ("platform", "server", "refusal"),
    [
        (POSIX, python_server(), None),
        (POSIX, python_server(Server.GUNICORN), None),
        (WINDOWS, python_server(), None),
        (WINDOWS, python_server(Server.GUNICORN), "does not support gunicorn on Windows"),
        (POSIX, python_server(Server.HYPERCORN), "does not support Hypercorn"),
        (POSIX, python_server(Server.GRANIAN), "does not support Granian"),
        (POSIX, python_server(listener=Listener.TLS), "does not support TLS on the public port"),
        (POSIX, python_server(listener=Listener.UNIX_SOCKET), "does not support a Unix socket on the public port"),
        (
            POSIX,
            python_server(Server.GUNICORN, Listener.INHERITED_FD),
            "does not support an inherited file descriptor on the public port",
        ),
    ],
)
def test_only_plain_tcp_on_a_supported_server_can_be_forwarded(
    platform: Platform, server: PythonServer, refusal: str | None
) -> None:
    expected: Final = "gateway" if refusal is None else StartupError(f"LITELLM_RUST_FIRST {refusal}")
    assert startup_check(platform, server, "gateway") == expected


@pytest.mark.parametrize(
    ("server", "refusal"),
    [
        (python_server(Server.HYPERCORN), "does not support Hypercorn"),
        (python_server(), "requires the litellm-gateway binary from the litellm-proxy package"),
    ],
)
def test_a_refused_start_says_why_in_one_line_and_starts_nothing(
    server: PythonServer, refusal: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared: Final[list[bool]] = []
    refused: Final = supervise(
        server, ("python",), before_start=lambda: prepared.append(True), scripts_dir=str(tmp_path), platform=POSIX
    )
    assert (refused, prepared) == (StartupError(f"LITELLM_RUST_FIRST {refusal}"), [])
    assert (
        capsys.readouterr().err
        == f"LITELLM_RUST_FIRST {refusal}, so Python serves the public port without Rust forwarding\n"
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

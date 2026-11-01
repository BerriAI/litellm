import sys
from pathlib import Path
from typing import Final

import pytest
from click.testing import CliRunner

from litellm.proxy.proxy_cli import ProxyInitializationHelpers, run_server
from litellm.proxy.rust_gateway import child_argv, cli_command, gateway_binary, inherited_fd, startup_error


@pytest.mark.parametrize(
    "flag", ["--ssl_keyfile_path", "--ssl_certfile_path", "--ciphers", "--run_hypercorn", "--run_granian"]
)
@pytest.mark.parametrize("enable", ["flag", "environment"])
def test_unsupported_modes_fail_before_starting_children(
    flag: str, enable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST_GATEWAY", "true" if enable == "environment" else "false")
    arguments: Final = ([] if enable == "environment" else ["--rust_gateway"]) + [flag]
    values: Final = ["unused"] if flag.startswith("--ssl_") or flag == "--ciphers" else []
    result: Final = CliRunner().invoke(run_server, arguments + values)
    assert result.exit_code != 0
    assert f"--rust_gateway does not support {flag}" in result.output


def test_child_arguments_preserve_options_and_force_private_listener() -> None:
    arguments: Final = ("--rust_gateway", "--host", "0.0.0.0", "--port", "4000", "--num_workers", "2", "--reload")
    params: Final = run_server.make_context("litellm", list(child_argv(arguments, 45555))).params
    assert params["host"] == "127.0.0.1"
    assert params["port"] == 45555
    assert params["num_workers"] == 2
    assert params["reload"] is True
    assert params["rust_gateway"] is False
    assert cli_command(()) == (sys.executable, "-m", "litellm.proxy.proxy_cli")


def test_missing_binary_gives_install_instruction() -> None:
    assert startup_error(None, {}) == (
        "--rust_gateway requires the litellm-gateway binary. Install litellm[proxy] in this environment"
    )


def test_current_environment_binary_wins_over_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary: Final = tmp_path / "litellm-gateway"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr("sysconfig.get_path", lambda _: str(tmp_path))
    monkeypatch.setenv("PATH", "")
    assert gateway_binary() == str(binary)


@pytest.mark.parametrize("fd", [None, "17"])
def test_inherited_socket_reaches_uvicorn(fd: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    if fd is None:
        monkeypatch.delenv("_LITELLM_GATEWAY_FD", raising=False)
    else:
        monkeypatch.setenv("_LITELLM_GATEWAY_FD", fd)
    expected: Final = int(fd) if fd is not None else None
    options: Final = ProxyInitializationHelpers.get_default_unvicorn_init_args("127.0.0.1", 45555)
    assert inherited_fd() == expected
    assert options.get("fd") == expected

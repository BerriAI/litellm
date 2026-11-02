import asyncio
import signal
import socket
import sys
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from litellm.proxy.proxy_cli import ProxyInitializationHelpers, run_server
from litellm.proxy.rust_gateway import (
    child_argv,
    cli_command,
    gateway_binary,
    inherited_fd,
    metrics_host,
    startup_error,
    supervise,
)


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


@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "::"])
def test_metrics_keep_configured_host_separate_from_private_listener(
    supervised: bool, host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("_LITELLM_GATEWAY_METRICS_HOST", host)
    if supervised:
        monkeypatch.setenv("_LITELLM_GATEWAY_FD", "17")
    else:
        monkeypatch.delenv("_LITELLM_GATEWAY_FD", raising=False)
    assert metrics_host("127.0.0.1") == (host if supervised else "127.0.0.1")


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


@pytest.mark.parametrize(
    ("event", "python_exit", "rust_exit", "expected"),
    [
        ("signal", 0, 0, 0),
        ("signal", 1, 0, 1),
        ("signal", 0, 1, 1),
        ("python-exit", 0, 0, 1),
        ("rust-exit", 0, 0, 1),
        ("python-during-drain", 1, 0, 1),
        ("spawn-error", 0, 0, 1),
    ],
)
def test_supervisor_drains_before_stopping_python_and_reaps_both_children(
    event: str, python_exit: int, rust_exit: int, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    loop: Final = asyncio.new_event_loop()
    handlers: Final = Mock()
    listener: Final = MagicMock(spec=socket.socket)
    listener.__enter__.return_value = listener
    listener.getsockname.return_value = ("127.0.0.1", 45555)
    listener.fileno.return_value = 17
    children: Final = {}
    sent: Final = Mock()

    def current_loop():
        return Mock(wraps=loop, add_signal_handler=handlers)

    async def spawn(*argv, **kwargs):
        done: Final = asyncio.events.get_running_loop().create_future()
        pid: Final = 101 if argv[0] == "python" else 102

        async def wait() -> int:
            return await done

        def send_signal(signum: signal.Signals) -> None:
            sent(pid, signum)
            if pid == 102 and event == "python-during-drain":
                children[101][1].set_result(python_exit)
                return
            if pid == 101:
                assert children[102][1].done(), "Python must remain alive until Rust has drained"
            done.set_result(python_exit if pid == 101 else rust_exit)

        process: Final = Mock(spec=asyncio.subprocess.Process, pid=pid, wait=wait, send_signal=send_signal)
        if pid == 102 and event == "spawn-error":
            raise OSError("gateway cannot start")
        children[pid] = (process, done)
        if pid == 102:
            if event == "python-exit":
                children[101][1].set_result(python_exit)
            elif event == "rust-exit":
                done.set_result(rust_exit)
            else:
                _, stop, signum = handlers.call_args_list[0].args
                stop(signum)
                stop(signum)
        return process

    def kill_group(pid: int, signum: signal.Signals) -> None:
        assert signum == signal.SIGTERM
        _, done = children[pid]
        if done.done():
            raise ProcessLookupError
        done.set_result(1)

    with (
        patch("asyncio.events.new_event_loop", return_value=loop),
        patch("asyncio.get_running_loop", side_effect=current_loop),
        patch("asyncio.create_subprocess_exec", side_effect=spawn) as launch,
        patch("socket.socket", return_value=listener),
        patch("os.killpg", side_effect=kill_group) as kill,
    ):
        assert supervise("gateway", ("python", "--rust_gateway"), "0.0.0.0", 4000) == expected

    assert all(done.done() for _, done in children.values())
    assert call(101, signal.SIGTERM) in kill.call_args_list
    python_call, rust_call = launch.call_args_list
    assert python_call.args == ("python", "--host", "127.0.0.1", "--port", "45555")
    assert python_call.kwargs["pass_fds"] == (17,)
    assert python_call.kwargs["env"]["LITELLM_RUST_GATEWAY"] == "false"
    assert python_call.kwargs["env"]["FORWARDED_ALLOW_IPS"] == "127.0.0.1"
    assert rust_call.kwargs["env"]["LITELLM_FORWARD_UPSTREAM"] == "http://127.0.0.1:45555"
    assert rust_call.kwargs["env"]["PORT"] == "4000"
    if event == "spawn-error":
        assert "Rust gateway startup failed: gateway cannot start" in capsys.readouterr().err
    elif event == "signal" and rust_exit == 0:
        assert sent.call_args_list == [call(102, signal.SIGTERM), call(101, signal.SIGTERM)]
    else:
        assert call(101, signal.SIGTERM) not in sent.call_args_list

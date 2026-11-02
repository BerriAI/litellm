import functools
import signal
import sys
from pathlib import Path
from typing import Final

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from litellm.proxy.proxy_cli import ProxyInitializationHelpers, run_server
from litellm.proxy.rust_gateway import (
    Child,
    Disabled,
    Event,
    ExitCode,
    Exited,
    Finished,
    Running,
    Signal,
    State,
    Stop,
    Supervised,
    Supervisor,
    cli_command,
    gateway_binary,
    python_spec,
    rust_gateway_mode,
    rust_spec,
    startup_error,
    step,
    trusted_proxies,
)

TERM: Final = signal.SIGTERM
INT: Final = signal.SIGINT


@pytest.mark.parametrize(
    "flag", ["--ssl_keyfile_path", "--ssl_certfile_path", "--ciphers", "--run_hypercorn", "--run_granian"]
)
def test_unsupported_modes_fail_before_starting_children(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST_GATEWAY", "true")
    monkeypatch.delenv("_LITELLM_GATEWAY_FD", raising=False)
    values: Final = ["unused"] if flag.startswith("--ssl_") or flag == "--ciphers" else []
    result: Final = CliRunner().invoke(run_server, [flag, *values])
    assert result.exit_code != 0
    assert f"LITELLM_RUST_GATEWAY does not support {flag}" in result.output


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, Disabled()),
        ({"LITELLM_RUST_GATEWAY": "false"}, Disabled()),
        ({"LITELLM_RUST_GATEWAY": "off"}, Disabled()),
        ({"LITELLM_RUST_GATEWAY": "0"}, Disabled()),
        ({"LITELLM_RUST_GATEWAY": "true"}, Supervisor()),
        ({"LITELLM_RUST_GATEWAY": "1"}, Supervisor()),
        ({"LITELLM_RUST_GATEWAY": "yes"}, Supervisor()),
        ({"LITELLM_RUST_GATEWAY": "on"}, Supervisor()),
        (
            {
                "LITELLM_RUST_GATEWAY": "true",
                "_LITELLM_GATEWAY_FD": "17",
                "_LITELLM_GATEWAY_METRICS_HOST": "0.0.0.0",
                "FORWARDED_ALLOW_IPS": "127.0.0.1,10.0.0.5",
            },
            Supervised(17, "0.0.0.0", ("127.0.0.1", "10.0.0.5")),
        ),
    ],
)
def test_mode_is_resolved_from_environment(environ: dict[str, str], expected: object) -> None:
    assert rust_gateway_mode(environ) == expected


def test_unrecognized_switch_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        rust_gateway_mode({"LITELLM_RUST_GATEWAY": "enabled"})


@pytest.mark.parametrize(
    ("operator_value", "expected"),
    [
        (None, ("127.0.0.1",)),
        ("", ("127.0.0.1",)),
        ("10.0.0.5", ("127.0.0.1", "10.0.0.5")),
        ("127.0.0.1, 10.0.0.5", ("127.0.0.1", "10.0.0.5")),
        ("10.0.0.5,127.0.0.1", ("127.0.0.1", "10.0.0.5")),
        ("*", ("*",)),
        ("10.0.0.5,*", ("*",)),
    ],
)
def test_operator_trust_is_kept_and_rust_hop_added(operator_value: str | None, expected: tuple[str, ...]) -> None:
    assert trusted_proxies(operator_value) == expected


@pytest.mark.parametrize("operator_value", [None, "", "10.0.0.5", "10.0.0.5, 192.0.2.0/24", "127.0.0.1"])
def test_python_child_always_trusts_the_rust_hop(operator_value: str | None) -> None:
    environ: Final = {} if operator_value is None else {"FORWARDED_ALLOW_IPS": operator_value}
    spec: Final = python_spec(("python",), 17, 45555, "0.0.0.0", environ)
    assert spec.env["FORWARDED_ALLOW_IPS"].split(",")[0] == "127.0.0.1"
    assert rust_gateway_mode(spec.env) == Supervised(17, "0.0.0.0", trusted_proxies(operator_value))


def test_python_child_listens_privately_and_keeps_options() -> None:
    arguments: Final = ("--host", "0.0.0.0", "--port", "4000", "--num_workers", "2", "--reload")
    spec: Final = python_spec(arguments, 17, 45555, "0.0.0.0", {})
    params: Final = run_server.make_context("litellm", list(spec.argv)).params
    assert params["host"] == "127.0.0.1"
    assert params["port"] == 45555
    assert params["num_workers"] == 2
    assert params["reload"] is True
    assert spec.pass_fds == (17,)
    assert cli_command(()) == (sys.executable, "-m", "litellm.proxy.proxy_cli")


def test_rust_child_serves_public_port_and_forwards_to_private_listener() -> None:
    spec: Final = rust_spec("gateway", 45555, "0.0.0.0", 4000, {"RUST_LOG": "info"})
    assert spec.argv == ("gateway",)
    assert spec.pass_fds == ()
    assert dict(spec.env) == {
        "RUST_LOG": "info",
        "HOST": "0.0.0.0",
        "PORT": "4000",
        "LITELLM_FORWARD_UPSTREAM": "http://127.0.0.1:45555",
    }


def test_missing_binary_gives_install_instruction() -> None:
    assert startup_error(None, {}) == (
        "LITELLM_RUST_GATEWAY requires the litellm-gateway binary. Install litellm[proxy] in this environment"
    )


def test_binary_is_found_only_in_the_environment_scripts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: Final = tmp_path / "scripts"
    elsewhere: Final = tmp_path / "elsewhere"
    scripts.mkdir()
    elsewhere.mkdir()
    binary: Final = elsewhere / "litellm-gateway"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(elsewhere))
    assert gateway_binary(str(scripts)) is None
    assert gateway_binary(str(elsewhere)) == str(binary)


@pytest.mark.parametrize("listener_fd", [None, 17])
def test_inherited_socket_reaches_uvicorn(listener_fd: int | None) -> None:
    options: Final = ProxyInitializationHelpers.get_default_unvicorn_init_args(
        "127.0.0.1", 45555, listener_fd=listener_fd
    )
    assert options.get("fd") == listener_fd


def _fold(events: tuple[Event, ...]) -> tuple[State, tuple[Signal, ...]]:
    def advance(acc: tuple[State, tuple[Signal, ...]], event: Event) -> tuple[State, tuple[Signal, ...]]:
        state, sent = acc
        following, action = step(state, event)
        return following, sent if action is None else (*sent, action)

    return functools.reduce(advance, events, (Running(), ()))


RUST: Final = Child.RUST
PYTHON: Final = Child.PYTHON


@pytest.mark.parametrize(
    ("events", "signals", "code"),
    [
        (
            (Stop(TERM), Exited(RUST, 0), Exited(PYTHON, 0)),
            (Signal(RUST, TERM), Signal(PYTHON, TERM)),
            ExitCode.CLEAN,
        ),
        ((Exited(PYTHON, 0),), (), ExitCode.CHILD_FAILED),
        ((Exited(RUST, 1),), (), ExitCode.CHILD_FAILED),
        ((Stop(TERM), Exited(PYTHON, 0)), (Signal(RUST, TERM),), ExitCode.CHILD_FAILED),
        ((Stop(TERM), Exited(RUST, 1)), (Signal(RUST, TERM),), ExitCode.CHILD_FAILED),
        (
            (Stop(TERM), Exited(RUST, 0), Exited(PYTHON, 1)),
            (Signal(RUST, TERM), Signal(PYTHON, TERM)),
            ExitCode.CHILD_FAILED,
        ),
        (
            (Stop(INT), Stop(INT), Exited(RUST, 0), Exited(PYTHON, 0)),
            (Signal(RUST, INT), Signal(PYTHON, INT)),
            ExitCode.CLEAN,
        ),
        (
            (Exited(RUST, 1), Stop(TERM), Exited(PYTHON, 0)),
            (),
            ExitCode.CHILD_FAILED,
        ),
    ],
)
def test_supervisor_drains_rust_before_python_and_fails_on_unasked_exit(
    events: tuple[Event, ...], signals: tuple[Signal, ...], code: ExitCode
) -> None:
    assert _fold(events) == (Finished(code), signals)

import asyncio
import functools
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

if TYPE_CHECKING:
    from uvicorn._types import (
        ASGIReceiveCallable,
        ASGISendCallable,
        ASGISendEvent,
        HTTPDisconnectEvent,
        HTTPScope,
        Scope,
    )

from litellm.proxy.proxy_cli import ProxyInitializationHelpers, run_server
from litellm.proxy.rust_gateway import (
    Child,
    Disabled,
    Event,
    ExitCode,
    Exited,
    Finished,
    Running,
    RustPeerMiddleware,
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
            Supervised(17, "0.0.0.0", True, "127.0.0.1,10.0.0.5"),
        ),
    ],
)
def test_mode_is_resolved_from_environment(environ: dict[str, str], expected: object) -> None:
    assert rust_gateway_mode(environ) == expected


def test_unrecognized_switch_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        rust_gateway_mode({"LITELLM_RUST_GATEWAY": "enabled"})


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, Supervised(17, "0.0.0.0", True, "127.0.0.1")),
        ({"FORWARDED_ALLOW_IPS": "10.0.0.5"}, Supervised(17, "0.0.0.0", True, "10.0.0.5")),
        ({"FORWARDED_ALLOW_IPS": "*"}, Supervised(17, "0.0.0.0", True, "*")),
        ({"_LITELLM_GATEWAY_PROXY_HEADERS": "false"}, Supervised(17, "0.0.0.0", False, "127.0.0.1")),
    ],
)
def test_python_child_keeps_the_trust_python_alone_would_use(environ: dict[str, str], expected: Supervised) -> None:
    spec: Final = python_spec(("python",), 17, 45555, "0.0.0.0", environ)
    assert rust_gateway_mode(spec.env) == expected


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
    assert options.get("proxy_headers", True) is (listener_fd is None)


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


Headers = tuple[tuple[bytes, bytes], ...]


async def _receive() -> "HTTPDisconnectEvent":
    return {"type": "http.disconnect"}


async def _send(message: "ASGISendEvent") -> None:
    return None


async def _seen_by_app(
    behind_rust: bool, client: tuple[str, int], headers: Headers, mode: Supervised
) -> tuple[tuple[str, int] | None, str, Headers]:
    seen: Final[asyncio.Future[tuple[tuple[str, int] | None, str, Headers]]] = (
        asyncio.get_running_loop().create_future()
    )

    async def probe(scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable") -> None:
        assert scope["type"] == "http"
        seen.set_result((scope["client"], scope["scheme"], tuple(scope["headers"])))

    direct: Final = ProxyHeadersMiddleware(probe, mode.trusted_proxies) if mode.proxy_headers else probe
    app: Final = RustPeerMiddleware(probe, mode) if behind_rust else direct
    scope: Final[HTTPScope] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": client,
        "server": ("127.0.0.1", 4000),
    }
    await app(scope, _receive, _send)
    return seen.result()


PEER: Final = ("203.0.113.7", 51234)
TRUSTED_PEER: Final = ("10.0.12.34", 40000)
FORWARDED: Final[Headers] = (
    (b"x-forwarded-for", b"198.51.100.1, 203.0.113.7"),
    (b"x-forwarded-proto", b"https"),
)


@pytest.mark.parametrize("peer", [PEER, TRUSTED_PEER, ("2001:db8::1", 443)])
@pytest.mark.parametrize("headers", [(), FORWARDED, ((b"x-forwarded-for", b"10.1.2.3"), (b"host", b"public.example"))])
@pytest.mark.parametrize(
    "mode",
    [
        Supervised(17, "0.0.0.0", True, "127.0.0.1"),
        Supervised(17, "0.0.0.0", True, "10.0.0.0/8"),
        Supervised(17, "0.0.0.0", True, "*"),
        Supervised(17, "0.0.0.0", False, "10.0.0.0/8"),
    ],
)
async def test_python_behind_rust_sees_what_a_direct_connection_would(
    peer: tuple[str, int], headers: Headers, mode: Supervised
) -> None:
    rendered: Final = f"[{peer[0]}]:{peer[1]}" if ":" in peer[0] else f"{peer[0]}:{peer[1]}"
    behind_rust: Final = await _seen_by_app(
        True, ("127.0.0.1", 60000), (*headers, (b"x-litellm-rust-peer", rendered.encode())), mode
    )
    direct: Final = await _seen_by_app(False, peer, headers, mode)
    assert behind_rust == direct


async def test_untrusted_peer_cannot_set_the_scheme_behind_rust() -> None:
    client, scheme, _ = await _seen_by_app(
        True,
        ("127.0.0.1", 60000),
        ((b"x-forwarded-proto", b"https"), (b"x-litellm-rust-peer", b"203.0.113.7:51234")),
        Supervised(17, "0.0.0.0", True, "127.0.0.1"),
    )
    assert (client, scheme) == (("203.0.113.7", 51234), "http")


async def test_peer_header_from_anyone_but_the_rust_hop_is_dropped_and_ignored() -> None:
    client, _, headers = await _seen_by_app(
        True,
        ("192.0.2.9", 1000),
        ((b"x-litellm-rust-peer", b"10.0.12.34:1"),),
        Supervised(17, "0.0.0.0", False, "127.0.0.1"),
    )
    assert (client, headers) == (("192.0.2.9", 1000), ())

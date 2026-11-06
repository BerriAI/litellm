import os
import threading
from typing import Final

import pytest
import uvicorn
from pydantic import ValidationError

from litellm.proxy.proxy_cli import WORKER_APP, ProxyInitializationHelpers
from litellm.proxy.rust_forward.mode import (
    SUPERVISED_ENV,
    AssignedPort,
    InheritedFd,
    Mode,
    RustForwardMode,
    Supervised,
    gunicorn_bind,
    rust_forward_mode,
    server_proxy_headers,
    stop_with_supervisor,
    supervised_environment,
    uvicorn_options,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, Mode.DISABLED),
        ({"LITELLM_RUST": "false"}, Mode.DISABLED),
        ({"LITELLM_RUST": "off"}, Mode.DISABLED),
        ({"LITELLM_RUST": "0"}, Mode.DISABLED),
        ({"LITELLM_RUST": "true"}, Mode.SUPERVISOR),
        ({"LITELLM_RUST": "1"}, Mode.SUPERVISOR),
        ({"LITELLM_RUST": "yes"}, Mode.SUPERVISOR),
        ({"LITELLM_RUST": "on"}, Mode.SUPERVISOR),
        ({"LITELLM_RUST": " 1 "}, Mode.SUPERVISOR),
        ({"LITELLM_RUST": ""}, Mode.DISABLED),
        ({"LITELLM_RUST": " "}, Mode.DISABLED),
    ],
)
def test_mode_is_resolved_from_environment(environ: dict[str, str], expected: RustForwardMode) -> None:
    assert rust_forward_mode(environ) == expected


@pytest.mark.parametrize(
    ("environ", "invalid"),
    [
        ({"LITELLM_RUST": "enabled"}, "LITELLM_RUST"),
        ({"LITELLM_RUST": "2"}, "LITELLM_RUST"),
        ({SUPERVISED_ENV: '{"listener": {"fd": "seventeen"}, "proxy_headers": true, "trusted_proxies": "*"}'}, "fd"),
        (
            {SUPERVISED_ENV: '{"listener": {"fd": 17}, "proxy_headers": "maybe", "trusted_proxies": "*"}'},
            "proxy_headers",
        ),
        ({SUPERVISED_ENV: '{"listener": {"fd": 17}, "proxy_headers": true}'}, "trusted_proxies"),
    ],
)
def test_invalid_forwarding_value_stops_startup_naming_it(environ: dict[str, str], invalid: str) -> None:
    with pytest.raises(ValidationError, match=invalid):
        rust_forward_mode(environ)


@pytest.mark.parametrize(
    "written",
    [
        Supervised(InheritedFd(17), True, "127.0.0.1"),
        Supervised(InheritedFd(17), True, "127.0.0.1,10.0.0.5"),
        Supervised(InheritedFd(17), False, "127.0.0.1"),
        Supervised(AssignedPort(45555), True, "*"),
        Supervised(AssignedPort(45555), False, "*"),
    ],
)
def test_child_reads_back_exactly_what_the_supervisor_wrote(written: Supervised) -> None:
    inherited: Final = {"LITELLM_RUST": "true", "FORWARDED_ALLOW_IPS": "10.9.9.9"}
    assert rust_forward_mode({**inherited, **supervised_environment(written)}) == written


def _served_uvicorn_config(mode: RustForwardMode) -> uvicorn.Config:
    options: Final = {
        **ProxyInitializationHelpers.get_default_unvicorn_init_args("0.0.0.0", 45555),
        **uvicorn_options(mode, WORKER_APP),
    }
    return uvicorn.Config(**options)


def test_supervised_uvicorn_serves_the_inherited_socket_without_trusting_the_rust_hop() -> None:
    config: Final = _served_uvicorn_config(Supervised(InheritedFd(17), True, "*"))
    assert (config.fd, config.proxy_headers) == (17, False)
    assert (config.app, config.factory) == (WORKER_APP, True)


@pytest.mark.parametrize("mode", list(Mode))
def test_unsupervised_servers_keep_their_own_socket_and_proxy_headers(mode: RustForwardMode) -> None:
    config: Final = _served_uvicorn_config(mode)
    assert (config.fd, config.proxy_headers, config.port) == (None, True, 45555)
    assert gunicorn_bind(mode, "0.0.0.0:4000") == "0.0.0.0:4000"
    assert server_proxy_headers(mode) is True


def test_supervised_gunicorn_binds_the_inherited_socket_without_trusting_the_rust_hop() -> None:
    mode: Final = Supervised(InheritedFd(17), True, "*")
    assert gunicorn_bind(mode, "0.0.0.0:4000") == "fd://17"
    assert server_proxy_headers(mode) is False


def test_assigned_port_serves_only_the_loopback_hop_without_trusting_it() -> None:
    mode: Final = Supervised(AssignedPort(45556), True, "*")
    config: Final = _served_uvicorn_config(mode)
    assert (config.host, config.port, config.fd, config.proxy_headers) == ("127.0.0.1", 45556, None, False)
    assert gunicorn_bind(mode, "0.0.0.0:4000") == "127.0.0.1:45556"
    assert server_proxy_headers(mode) is False


@pytest.mark.parametrize(
    ("mode", "stops"),
    [(Supervised(InheritedFd(17), False, "127.0.0.1"), True), (Mode.DISABLED, False), (Mode.SUPERVISOR, False)],
)
def test_only_a_supervised_server_stops_once_its_supervisor_is_gone(mode: RustForwardMode, stops: bool) -> None:
    read_end, write_end = os.pipe()
    stopped: Final = threading.Event()
    try:
        stop_with_supervisor(mode, read_end, stopped.set)
        assert not stopped.wait(0.2)
        os.close(write_end)
        assert stopped.wait(5 if stops else 0.5) is stops
    finally:
        os.close(read_end)

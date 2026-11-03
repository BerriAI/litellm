from typing import Final

import pytest
import uvicorn
from pydantic import ValidationError

from litellm.proxy.proxy_cli import WORKER_APP, ProxyInitializationHelpers
from litellm.proxy.rust_forward.mode import (
    AssignedPort,
    Disabled,
    InheritedFd,
    RustForwardMode,
    Supervised,
    Supervisor,
    gunicorn_bind,
    rust_forward_mode,
    server_proxy_headers,
    supervised_environment,
    uvicorn_options,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, Disabled()),
        ({"LITELLM_RUST": "false"}, Disabled()),
        ({"LITELLM_RUST": "off"}, Disabled()),
        ({"LITELLM_RUST": "0"}, Disabled()),
        ({"LITELLM_RUST": "true"}, Supervisor()),
        ({"LITELLM_RUST": "1"}, Supervisor()),
        ({"LITELLM_RUST": "yes"}, Supervisor()),
        ({"LITELLM_RUST": "on"}, Supervisor()),
        ({"LITELLM_RUST": " 1 "}, Supervisor()),
        ({"LITELLM_RUST": ""}, Disabled()),
        ({"LITELLM_RUST": " "}, Disabled()),
        (
            {"LITELLM_RUST": "true", "_LITELLM_GATEWAY_FD": "17", "FORWARDED_ALLOW_IPS": "127.0.0.1,10.0.0.5"},
            Supervised(InheritedFd(17), True, "127.0.0.1,10.0.0.5"),
        ),
    ],
)
def test_mode_is_resolved_from_environment(environ: dict[str, str], expected: RustForwardMode) -> None:
    assert rust_forward_mode(environ) == expected


@pytest.mark.parametrize(
    ("environ", "variable"),
    [
        ({"LITELLM_RUST": "enabled"}, "LITELLM_RUST"),
        ({"LITELLM_RUST": "2"}, "LITELLM_RUST"),
        ({"_LITELLM_GATEWAY_FD": "seventeen"}, "_LITELLM_GATEWAY_FD"),
        ({"_LITELLM_GATEWAY_FD": "17", "_LITELLM_GATEWAY_PROXY_HEADERS": "maybe"}, "_LITELLM_GATEWAY_PROXY_HEADERS"),
    ],
)
def test_invalid_forwarding_value_stops_startup_naming_the_variable(environ: dict[str, str], variable: str) -> None:
    with pytest.raises(ValidationError, match=variable):
        rust_forward_mode(environ)


@pytest.mark.parametrize(
    ("proxy_headers", "trusted_proxies", "inherited", "expected"),
    [
        (True, None, {}, Supervised(InheritedFd(17), True, "127.0.0.1")),
        (True, None, {"FORWARDED_ALLOW_IPS": "10.0.0.5"}, Supervised(InheritedFd(17), True, "10.0.0.5")),
        (True, "*", {"FORWARDED_ALLOW_IPS": "10.0.0.5"}, Supervised(InheritedFd(17), True, "*")),
        (False, None, {}, Supervised(InheritedFd(17), False, "127.0.0.1")),
        (True, None, {}, Supervised(AssignedPort(45555), True, "127.0.0.1")),
        (False, "*", {}, Supervised(AssignedPort(45555), False, "*")),
    ],
)
def test_child_reads_back_the_listener_and_trust_the_supervisor_wrote(
    proxy_headers: bool, trusted_proxies: str | None, inherited: dict[str, str], expected: Supervised
) -> None:
    written: Final = supervised_environment(expected.listener, proxy_headers, trusted_proxies)
    assert rust_forward_mode({**inherited, **written}) == expected


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


@pytest.mark.parametrize("mode", [Disabled(), Supervisor()])
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

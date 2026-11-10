import os
import threading
from typing import Final

import pytest
import uvicorn
from pydantic import ValidationError

from litellm.proxy.proxy_cli import WORKER_APP, ProxyInitializationHelpers
from litellm.proxy.rust_forward.mode import (
    SUPERVISED_ENV,
    Mode,
    RustForwardMode,
    Supervised,
    rust_forward_mode,
    private_listener,
    stop_with_supervisor,
    supervised_environment,
    uvicorn_options,
)
from litellm.proxy.rust_forward.server import NO_TRUSTED_PROXIES
from litellm.proxy.server_listener import AssignedPort, InheritedFd, PrivateListener


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        *((value, Mode.DISABLED) for value in (None, "", " ", "0", "false", "off")),
        *((value, Mode.SUPERVISOR) for value in ("1", "true", "yes", "on", " 1 ")),
    ],
)
def test_mode_is_resolved_from_environment(value: str | None, expected: RustForwardMode) -> None:
    assert rust_forward_mode({} if value is None else {"LITELLM_RUST": value}) == expected


@pytest.mark.parametrize(
    ("environ", "invalid"),
    [
        ({"LITELLM_RUST": "enabled"}, "LITELLM_RUST"),
        ({"LITELLM_RUST": "2"}, "LITELLM_RUST"),
        ({SUPERVISED_ENV: '{"listener": {"fd": "seventeen"}, "trusted_proxies": "*"}'}, "fd"),
        ({SUPERVISED_ENV: '{"listener": {"fd": 17}, "trusted_proxies": 7}'}, "trusted_proxies"),
        ({SUPERVISED_ENV: '{"listener": {"fd": 17}}'}, "trusted_proxies"),
    ],
)
def test_invalid_forwarding_value_stops_startup_naming_it(environ: dict[str, str], invalid: str) -> None:
    with pytest.raises(ValidationError, match=invalid):
        rust_forward_mode(environ)


@pytest.mark.parametrize(
    "written",
    [
        Supervised(InheritedFd(17), "127.0.0.1"),
        Supervised(InheritedFd(17), "127.0.0.1,10.0.0.5"),
        Supervised(InheritedFd(17), NO_TRUSTED_PROXIES),
        Supervised(AssignedPort(45555), "*"),
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


PUBLIC: Final = ("0.0.0.0", 45555, None)


@pytest.mark.parametrize(
    ("mode", "app", "uvicorn_listener", "gunicorn_listener", "proxy_headers"),
    [
        (Mode.DISABLED, "litellm.proxy.proxy_server:app", PUBLIC, None, True),
        (Mode.SUPERVISOR, "litellm.proxy.proxy_server:app", PUBLIC, None, True),
        (Supervised(InheritedFd(17), "*"), WORKER_APP, ("0.0.0.0", 45555, 17), InheritedFd(17), False),
        (Supervised(AssignedPort(45556), "*"), WORKER_APP, ("127.0.0.1", 45556, None), AssignedPort(45556), False),
    ],
)
def test_supervised_server_serves_the_private_hop_and_never_trusts_it(
    mode: RustForwardMode,
    app: str,
    uvicorn_listener: tuple[str, int, int | None],
    gunicorn_listener: PrivateListener | None,
    proxy_headers: bool,
) -> None:
    config: Final = _served_uvicorn_config(mode)
    assert (config.app, (config.host, config.port, config.fd), config.proxy_headers) == (
        app,
        uvicorn_listener,
        proxy_headers,
    )
    assert config.factory is (app == WORKER_APP)
    assert private_listener(mode) == gunicorn_listener


@pytest.mark.parametrize(
    ("mode", "stops"),
    [(Supervised(InheritedFd(17), "127.0.0.1"), True), (Mode.DISABLED, False), (Mode.SUPERVISOR, False)],
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

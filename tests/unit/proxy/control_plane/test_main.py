import logging
from dataclasses import dataclass, field
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm._logging import _get_uvicorn_json_log_config
from litellm.proxy.control_plane.app import LicenseRequired
from litellm.proxy.control_plane.main import Tls, _uvicorn, serve
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from tests.unit.proxy.control_plane.support import MASTER_KEY


@dataclass(frozen=True)
class _License:
    premium: bool

    def is_premium(self) -> bool:
        return self.premium


@dataclass
class _Runner:  # mutable-ok: records the app it was asked to serve
    served: list[tuple[FastAPI, str, int, Tls | None, int | None, str | None, int | None, int | None]] = field(
        default_factory=list
    )

    def __call__(
        self,
        app: FastAPI,
        *,
        host: str,
        port: int,
        tls: Tls | None,
        limit_concurrency: int | None,
        log_config: str | None,
        keepalive_timeout: int | None,
        max_request_header_size: int | None,
    ) -> None:
        self.served.append(
            (app, host, port, tls, limit_concurrency, log_config, keepalive_timeout, max_request_header_size)
        )


@dataclass(frozen=True)
class _LogConfigUnset:
    pass


_LOG_CONFIG_UNSET: Final = _LogConfigUnset()


@dataclass(frozen=True)
class _LogLevelUnset:
    pass


_LOG_LEVEL_UNSET: Final = _LogLevelUnset()


@dataclass(frozen=True)
class _UvicornCall:
    app: FastAPI
    host: str
    port: int
    server_header: bool
    timeout_graceful_shutdown: int | None
    timeout_keep_alive: int
    h11_max_incomplete_event_size: int | None
    ssl_keyfile: str | None
    ssl_certfile: str | None
    limit_concurrency: int | None
    log_config: str | dict[str, object] | None | _LogConfigUnset
    log_level: int | None | _LogLevelUnset


@dataclass
class _UvicornCallRecorder:  # mutable-ok: records calls made to the injected Uvicorn runner
    calls: list[_UvicornCall] = field(default_factory=list)

    def __call__(
        self,
        *,
        app: FastAPI,
        host: str,
        port: int,
        server_header: bool,
        timeout_graceful_shutdown: int | None,
        timeout_keep_alive: int,
        h11_max_incomplete_event_size: int | None,
        ssl_keyfile: str | None,
        ssl_certfile: str | None,
        limit_concurrency: int | None,
        log_config: str | dict[str, object] | None | _LogConfigUnset = _LOG_CONFIG_UNSET,
        log_level: int | None | _LogLevelUnset = _LOG_LEVEL_UNSET,
    ) -> None:
        self.calls.append(
            _UvicornCall(
                app,
                host,
                port,
                server_header,
                timeout_graceful_shutdown,
                timeout_keep_alive,
                h11_max_incomplete_event_size,
                ssl_keyfile,
                ssl_certfile,
                limit_concurrency,
                log_config,
                log_level,
            )
        )


def _settings() -> ControlPlaneSettings:
    return ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})


def test_serve_builds_the_control_plane_app_and_hands_it_to_the_runner() -> None:
    runner: Final = _Runner()
    serve("127.0.0.1", 8100, settings=_settings, license_check=lambda: _License(True), run=runner)
    [(app, host, port, _tls, _limit_concurrency, _log_config, _keepalive, _header_size)] = runner.served
    assert (host, port) == ("127.0.0.1", 8100)
    client: Final = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/api/v1/dataplanes", headers={"Authorization": f"Bearer {MASTER_KEY}"}).json() == {"items": []}
    assert client.get("/v1/chat/completions").status_code == 404, "the proxy's routes must not be on this app"


def test_serve_refuses_without_a_license_and_never_binds() -> None:
    runner: Final = _Runner()
    with pytest.raises(LicenseRequired):
        serve("0.0.0.0", 4000, settings=_settings, license_check=lambda: _License(False), run=runner)
    assert runner.served == []


def test_serve_hands_configured_tls_to_the_runner() -> None:
    runner: Final = _Runner()
    tls: Final = Tls(keyfile="/certs/tls.key", certfile="/certs/tls.crt")
    serve("0.0.0.0", 8443, tls=tls, settings=_settings, license_check=lambda: _License(True), run=runner)
    assert [entry[3] for entry in runner.served] == [tls]


def test_serve_hands_limit_concurrency_to_the_runner() -> None:
    runner: Final = _Runner()
    serve(
        "0.0.0.0",
        8100,
        settings=_settings,
        license_check=lambda: _License(True),
        run=runner,
        limit_concurrency=250,
    )
    assert [entry[4] for entry in runner.served] == [250]


def test_uvicorn_keeps_its_default_logging_config_when_unset() -> None:
    runner: Final = _UvicornCallRecorder()
    _uvicorn(
        FastAPI(),
        host="127.0.0.1",
        port=8100,
        tls=None,
        limit_concurrency=None,
        log_config=None,
        keepalive_timeout=None,
        max_request_header_size=None,
        uvicorn_runner=runner,
        json_logs=False,
        environ={},
    )
    assert len(runner.calls) == 1
    assert runner.calls[0].log_config is _LOG_CONFIG_UNSET
    assert runner.calls[0].log_level is _LOG_LEVEL_UNSET


def test_uvicorn_uses_json_logging_before_the_environment_log_level() -> None:
    runner: Final = _UvicornCallRecorder()
    _uvicorn(
        FastAPI(),
        host="127.0.0.1",
        port=8100,
        tls=None,
        limit_concurrency=None,
        log_config=None,
        keepalive_timeout=None,
        max_request_header_size=None,
        uvicorn_runner=runner,
        json_logs=True,
        environ={"LITELLM_LOG": "DEBUG"},
    )

    call: Final = runner.calls[0]
    assert call.log_config == _get_uvicorn_json_log_config()
    assert call.log_level is _LOG_LEVEL_UNSET


def test_uvicorn_uses_the_environment_log_level_when_json_logging_is_disabled() -> None:
    runner: Final = _UvicornCallRecorder()
    _uvicorn(
        FastAPI(),
        host="127.0.0.1",
        port=8100,
        tls=None,
        limit_concurrency=None,
        log_config=None,
        keepalive_timeout=None,
        max_request_header_size=None,
        uvicorn_runner=runner,
        json_logs=False,
        environ={"LITELLM_LOG": "WARNING"},
    )

    assert runner.calls[0].log_config is _LOG_CONFIG_UNSET
    assert runner.calls[0].log_level == logging.WARNING


def test_uvicorn_forwards_an_explicit_logging_config() -> None:
    runner: Final = _UvicornCallRecorder()
    _uvicorn(
        FastAPI(),
        host="127.0.0.1",
        port=8100,
        tls=None,
        limit_concurrency=None,
        log_config="logging.yaml",
        keepalive_timeout=None,
        max_request_header_size=None,
        uvicorn_runner=runner,
        json_logs=True,
        environ={"LITELLM_LOG": "DEBUG"},
    )
    assert len(runner.calls) == 1
    assert runner.calls[0].log_config == "logging.yaml"
    assert runner.calls[0].log_level is _LOG_LEVEL_UNSET

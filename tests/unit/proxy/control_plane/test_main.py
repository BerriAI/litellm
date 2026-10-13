import importlib
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.control_plane.app import LicenseRequired
from litellm.proxy.control_plane.main import APP_FACTORY, Tls, serve
from litellm.proxy.control_plane.settings import ControlPlaneSettings


@dataclass(frozen=True)
class _License:
    premium: bool

    def is_premium(self) -> bool:
        return self.premium


@dataclass
class _Runner:  # mutable-ok: records the app it was asked to serve
    served: list[tuple[FastAPI, str, int, int, Tls | None]] = field(default_factory=list)

    def __call__(self, app: FastAPI, *, host: str, port: int, workers: int, tls: Tls | None) -> None:
        self.served.append((app, host, port, workers, tls))


def _settings() -> ControlPlaneSettings:
    return ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-cp"})


def test_serve_builds_the_control_plane_app_and_hands_it_to_the_runner() -> None:
    runner = _Runner()
    serve("127.0.0.1", 8100, settings=_settings, license_check=lambda: _License(True), run=runner)
    [(app, host, port, workers, _tls)] = runner.served
    assert (host, port, workers) == ("127.0.0.1", 8100, 1)
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/api/v1/dataplanes", headers={"Authorization": "Bearer sk-cp"}).json() == {"items": []}
    assert client.get("/v1/chat/completions").status_code == 404, "the proxy's routes must not be on this app"


def test_serve_refuses_without_a_license_and_never_binds() -> None:
    runner = _Runner()
    with pytest.raises(LicenseRequired):
        serve("0.0.0.0", 4000, settings=_settings, license_check=lambda: _License(False), run=runner)
    assert runner.served == []


@pytest.mark.parametrize(("requested", "expected"), [(4, 4), (0, 1), (-2, 1)])
def test_serve_passes_the_worker_count_through_to_the_runner(requested: int, expected: int) -> None:
    runner = _Runner()
    serve("0.0.0.0", 8100, workers=requested, settings=_settings, license_check=lambda: _License(True), run=runner)
    [(_, _, _, workers, _)] = runner.served
    assert workers == expected


def test_app_factory_is_importable_by_uvicorn_workers_and_builds_the_same_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-cp")
    monkeypatch.delenv("LITELLM_LICENSE", raising=False)
    module_path, attribute = APP_FACTORY.split(":")
    factory = getattr(importlib.import_module(module_path), attribute)
    with pytest.raises(LicenseRequired):
        factory()  # no LITELLM_LICENSE in the test environment, so every worker refuses too


def test_serve_hands_configured_tls_to_the_runner() -> None:
    runner = _Runner()
    tls = Tls(keyfile="/certs/tls.key", certfile="/certs/tls.crt")
    serve("0.0.0.0", 8443, tls=tls, settings=_settings, license_check=lambda: _License(True), run=runner)
    assert [entry[4] for entry in runner.served] == [tls]

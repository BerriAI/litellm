"""Entrypoint for `litellm --control-plane`: the same image and CLI, a different app. Refuses to serve unlicensed."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from fastapi import FastAPI

from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.control_plane.app import License, create_app
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings, load_settings

APP_FACTORY: Final = "litellm.proxy.control_plane.main:app_factory"


@dataclass(frozen=True, slots=True)
class Tls:
    keyfile: str
    certfile: str


class Runner(Protocol):
    def __call__(self, app: FastAPI, *, host: str, port: int, workers: int, tls: Tls | None) -> None: ...


def _uvicorn(app: FastAPI, *, host: str, port: int, workers: int, tls: Tls | None) -> None:
    """One worker serves the app already built (and license checked) by ``serve``. More workers need an importable
    factory because uvicorn re-imports the app in every worker process, each of which runs the license gate again."""
    import uvicorn

    keyfile: Final = None if tls is None else tls.keyfile
    certfile: Final = None if tls is None else tls.certfile
    if workers <= 1:
        uvicorn.run(app, host=host, port=port, ssl_keyfile=keyfile, ssl_certfile=certfile)
        return
    uvicorn.run(
        APP_FACTORY, host=host, port=port, workers=workers, factory=True, ssl_keyfile=keyfile, ssl_certfile=certfile
    )


def build_app(settings: ControlPlaneSettings, license_check: License) -> FastAPI:
    return create_app(build_container(settings), license_check)


def app_factory() -> FastAPI:
    return build_app(load_settings(), LicenseCheck())


def serve(
    host: str,
    port: int,
    workers: int = 1,
    tls: Tls | None = None,
    settings: Callable[[], ControlPlaneSettings] = load_settings,
    license_check: Callable[[], License] = LicenseCheck,
    run: Runner = _uvicorn,
) -> None:
    app: Final = build_app(settings(), license_check())
    run(app, host=host, port=port, workers=max(1, workers), tls=tls)

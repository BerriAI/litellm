"""App factory for the LiteLLM global control plane. Enterprise only: refuses to build without a valid license."""

from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol

from fastapi import FastAPI

from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.router import router as audit_router
from litellm.proxy.control_plane.auth.dependencies import get_telemetry, get_token_verifier
from litellm.proxy.control_plane.auth.router import get_login_service
from litellm.proxy.control_plane.auth.router import router as auth_router
from litellm.proxy.control_plane.common.exceptions import install_exception_handlers
from litellm.proxy.control_plane.container import Container
from litellm.proxy.control_plane.dataplanes.dependencies import get_dataplane_client, get_fanout, get_registry
from litellm.proxy.control_plane.dataplanes.router import router as dataplanes_router
from litellm.proxy.control_plane.fleet.router import router as fleet_router
from litellm.proxy.control_plane.health.router import router as health_router
from litellm.proxy.control_plane.proxy.router import router as proxy_router
from litellm.proxy.control_plane.usage.router import router as usage_router

TITLE: Final = "LiteLLM Control Plane"


class License(Protocol):
    def is_premium(self) -> bool: ...


class LicenseRequired(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The LiteLLM control plane is an enterprise feature. Set LITELLM_LICENSE to a valid license key."
        )


@dataclass(frozen=True, slots=True)
class _Lifespan:
    container: Container

    def __call__(self, app: FastAPI) -> "_Lifespan":
        return self

    async def __aenter__(self) -> None:
        self.container.audit_log.pipeline.start()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            await self.container.audit_log.pipeline.close()
        finally:
            self.container.telemetry.shutdown()


def create_app(container: Container, license_check: License) -> FastAPI:
    if not license_check.is_premium():
        raise LicenseRequired()

    app: Final = FastAPI(
        title=TITLE,
        version="1",
        lifespan=_Lifespan(container),
        docs_url=f"{container.settings.api_prefix}/docs",
        openapi_url=f"{container.settings.api_prefix}/openapi.json",
    )
    container.telemetry.instrument(app)
    app.dependency_overrides[get_token_verifier] = lambda: container.token_verifier
    app.dependency_overrides[get_telemetry] = lambda: container.telemetry
    app.dependency_overrides[get_login_service] = lambda: container.login_service
    app.dependency_overrides[get_registry] = lambda: container.registry
    app.dependency_overrides[get_dataplane_client] = lambda: container.dataplane_client
    app.dependency_overrides[get_fanout] = lambda: container.fanout
    app.dependency_overrides[get_audit_log] = lambda: container.audit_log
    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(auth_router, prefix=container.settings.api_prefix)
    app.include_router(dataplanes_router, prefix=container.settings.api_prefix)
    app.include_router(usage_router, prefix=container.settings.api_prefix)
    app.include_router(fleet_router, prefix=container.settings.api_prefix)
    app.include_router(proxy_router, prefix=container.settings.api_prefix)
    app.include_router(audit_router, prefix=container.settings.api_prefix)
    return app

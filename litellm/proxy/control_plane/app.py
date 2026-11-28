"""App factory for the LiteLLM global control plane. Enterprise only: refuses to build without a valid license."""

import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol

from fastapi import FastAPI, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._version import version
from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.router import router as audit_router
from litellm.proxy.control_plane.auth.dependencies import get_telemetry, get_token_verifier
from litellm.proxy.control_plane.auth.router import get_login_service
from litellm.proxy.control_plane.auth.router import router as auth_router
from litellm.proxy.control_plane.common.exceptions import (
    error_body,
    error_response,
    install_exception_handlers,
)
from litellm.proxy.control_plane.container import Container
from litellm.proxy.control_plane.dataplanes.dependencies import (
    get_control_plane_settings,
    get_dataplane_client,
    get_fanout,
    get_registry,
)
from litellm.proxy.control_plane.dataplanes.router import router as dataplanes_router
from litellm.proxy.control_plane.fleet.router import get_fleet_teams
from litellm.proxy.control_plane.fleet.router import router as fleet_router
from litellm.proxy.control_plane.fleet.teams import FleetTeams
from litellm.proxy.control_plane.health.router import router as health_router
from litellm.proxy.control_plane.proxy.router import router as proxy_router
from litellm.proxy.control_plane.roles.router import router as roles_router
from litellm.proxy.control_plane.store.dependencies import get_runtime_config
from litellm.proxy.control_plane.usage.router import router as usage_router
from litellm.proxy.middleware.request_size_limit_middleware import RequestSizeLimitMiddleware
from litellm.proxy.middleware.security_headers_middleware import SecurityHeadersMiddleware
from litellm.proxy.scalar_reference import mount_api_references
from litellm.proxy.ui_static import mount_admin_ui
from litellm.proxy.utils import get_server_root_path
from litellm.types.proxy.discovery_endpoints.ui_discovery_endpoints import (
    LITELLM_UI_CONFIG_PATH,
    UI_CONFIG_PATH,
    ControlPlaneUiDiscovery,
)

TITLE: Final = "LiteLLM Control Plane"
MAX_REQUEST_SIZE_MB: Final = 1
REQUEST_TOO_LARGE: Final = error_response(
    error_body("request-too-large", "Payload Too Large", 413, "Request size is too large. Max size is 1 MB.")
)


class License(Protocol):
    def is_premium(self) -> bool: ...


class LicenseRequired(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The LiteLLM control plane is an enterprise feature. Set LITELLM_LICENSE to a valid license key."
        )


class ControlPlaneRequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._middleware = RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: MAX_REQUEST_SIZE_MB,
            is_request_size_limit_enabled=lambda: True,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        replacing_413: Final = ContextVar("control_plane_request_too_large", default=False)

        async def send_problem(message: Message) -> None:
            if message["type"] == "http.response.start" and message["status"] == 413:
                replacing_413.set(True)
                await send(
                    {
                        "type": "http.response.start",
                        "status": REQUEST_TOO_LARGE.status_code,
                        "headers": list(REQUEST_TOO_LARGE.raw_headers),
                    }
                )
                return
            if replacing_413.get() and message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    await send({"type": "http.response.body", "body": REQUEST_TOO_LARGE.body, "more_body": False})
                return
            await send(message)

        await self._middleware(scope, receive, send_problem)


@dataclass(frozen=True, slots=True)
class _Lifespan:
    container: Container
    fleet: FleetTeams

    def __call__(self, app: FastAPI) -> "_Lifespan":
        return self

    async def __aenter__(self) -> None:
        await self.container.runtime_config.start()
        self.container.audit_log.pipeline.start()
        self.fleet.start()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            try:
                await self.fleet.close()
            finally:
                await self.container.runtime_config.close()
        finally:
            try:
                await self.container.audit_log.pipeline.close()
            finally:
                self.container.telemetry.shutdown()


def _mount_ui_config(app: FastAPI, api_prefix: str) -> None:
    async def get_ui_config(request: Request) -> ControlPlaneUiDiscovery:
        scope: Final[Mapping[str, object]] = request.scope
        root_path: Final = scope.get("root_path", "")
        return ControlPlaneUiDiscovery(
            server_root_path=root_path if isinstance(root_path, str) else "",
            proxy_base_url=None,
            auto_redirect_to_sso=False,
            admin_ui_disabled=os.getenv("DISABLE_ADMIN_UI", "false").lower() == "true",
            sso_configured=False,
            hide_default_credentials_hint=True,
            control_plane_api_prefix=api_prefix,
            litellm_version=None if version == "unknown" else version,
        )

    app.add_api_route(UI_CONFIG_PATH, get_ui_config, methods=["GET"])
    app.add_api_route(LITELLM_UI_CONFIG_PATH, get_ui_config, methods=["GET"])


def create_app(container: Container, license_check: License) -> FastAPI:
    if not license_check.is_premium():
        raise LicenseRequired()

    api_prefix: Final = container.settings.api_prefix
    server_root_path: Final = get_server_root_path()
    docs_url: Final = f"{api_prefix}/docs"
    swagger_url: Final = f"{api_prefix}/swagger"
    fleet: Final = FleetTeams(
        container.runtime_config, container.dataplane_client, container.fanout, container.audit_log
    )
    app: Final = FastAPI(
        title=TITLE,
        version="1",
        lifespan=_Lifespan(container, fleet),
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{api_prefix}/openapi.json",
        root_path=server_root_path,
    )
    app.add_middleware(ControlPlaneRequestSizeLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    container.telemetry.instrument(app, documentation_paths=(docs_url, swagger_url))
    app.dependency_overrides[get_token_verifier] = lambda: container.token_verifier
    app.dependency_overrides[get_telemetry] = lambda: container.telemetry
    app.dependency_overrides[get_login_service] = lambda: container.login_service
    app.dependency_overrides[get_registry] = lambda: container.registry
    app.dependency_overrides[get_control_plane_settings] = lambda: container.settings
    app.dependency_overrides[get_dataplane_client] = lambda: container.dataplane_client
    app.dependency_overrides[get_fanout] = lambda: container.fanout
    app.dependency_overrides[get_audit_log] = lambda: container.audit_log
    app.dependency_overrides[get_runtime_config] = lambda: container.runtime_config
    app.dependency_overrides[get_fleet_teams] = lambda: fleet
    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(auth_router, prefix=container.settings.api_prefix)
    app.include_router(dataplanes_router, prefix=container.settings.api_prefix)
    app.include_router(usage_router, prefix=container.settings.api_prefix)
    app.include_router(fleet_router, prefix=container.settings.api_prefix)
    app.include_router(proxy_router, prefix=container.settings.api_prefix)
    app.include_router(audit_router, prefix=container.settings.api_prefix)
    app.include_router(roles_router, prefix=container.settings.api_prefix)
    mount_api_references(app, docs_url, swagger_url)
    _mount_ui_config(app, api_prefix)
    mount_admin_ui(app, request_root_path=True)
    return app

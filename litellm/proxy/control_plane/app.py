"""App factory for the LiteLLM global control plane. Enterprise only: refuses to build without a valid license."""

from contextvars import ContextVar
from typing import Final, Protocol

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.control_plane.auth.dependencies import get_token_verifier
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
from litellm.proxy.control_plane.fleet.router import router as fleet_router
from litellm.proxy.control_plane.health.router import router as health_router
from litellm.proxy.control_plane.usage.router import router as usage_router
from litellm.proxy.middleware.request_size_limit_middleware import RequestSizeLimitMiddleware

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


def create_app(container: Container, license_check: License) -> FastAPI:
    if not license_check.is_premium():
        raise LicenseRequired()
    app: Final = FastAPI(
        title=TITLE,
        version="1",
        docs_url=f"{container.settings.api_prefix}/docs",
        openapi_url=f"{container.settings.api_prefix}/openapi.json",
    )
    app.add_middleware(ControlPlaneRequestSizeLimitMiddleware)
    app.dependency_overrides[get_token_verifier] = lambda: container.token_verifier
    app.dependency_overrides[get_login_service] = lambda: container.login_service
    app.dependency_overrides[get_registry] = lambda: container.registry
    app.dependency_overrides[get_control_plane_settings] = lambda: container.settings
    app.dependency_overrides[get_dataplane_client] = lambda: container.dataplane_client
    app.dependency_overrides[get_fanout] = lambda: container.fanout
    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(auth_router, prefix=container.settings.api_prefix)
    app.include_router(dataplanes_router, prefix=container.settings.api_prefix)
    app.include_router(usage_router, prefix=container.settings.api_prefix)
    app.include_router(fleet_router, prefix=container.settings.api_prefix)
    return app

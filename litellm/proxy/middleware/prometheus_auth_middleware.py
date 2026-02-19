"""
Prometheus Auth Middleware

Pure ASGI middleware — avoids Starlette's BaseHTTPMiddleware which wraps
streaming responses with receive_or_disconnect per chunk, blocking the
event loop and causing severe throughput degradation under concurrent
streaming load.
"""
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import litellm
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class PrometheusAuthMiddleware:
    """
    Middleware to authenticate requests to the metrics endpoint

    By default, auth is not run on the metrics endpoint

    Enabled by setting the following in proxy_config.yaml:

    ```yaml
    litellm_settings:
        require_auth_for_metrics_endpoint: true
    ```
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        if self._is_prometheus_metrics_endpoint(request):
            if self._should_run_auth_on_metrics_endpoint() is True:
                try:
                    await user_api_key_auth(
                        request=request,
                        api_key=request.headers.get(
                            SpecialHeaders.openai_authorization.value
                        )
                        or "",
                    )
                except Exception as e:
                    response = JSONResponse(
                        status_code=401,
                        content=f"Unauthorized access to metrics endpoint: {getattr(e, 'message', str(e))}",
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)

    @staticmethod
    def _is_prometheus_metrics_endpoint(request: Request):
        try:
            if "/metrics" in request.url.path:
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _should_run_auth_on_metrics_endpoint():
        """
        Returns True if auth should be run on the metrics endpoint

        False by default, set to True in proxy_config.yaml to enable

        ```yaml
        litellm_settings:
            require_auth_for_metrics_endpoint: true
        ```
        """
        return litellm.require_auth_for_metrics_endpoint

"""
Prometheus Auth Middleware
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import litellm
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class PrometheusAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to authenticate requests to the metrics endpoint

    By default, auth is not run on the metrics endpoint

    Enabled by setting the following in proxy_config.yaml:

    ```yaml
    litellm_settings:
        require_auth_for_metrics_endpoint: true
    ```
    """

    async def dispatch(self, request: Request, call_next):
        # Check if this is a request to the metrics endpoint

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
                    return JSONResponse(
                        status_code=401,
                        content=f"Unauthorized access to metrics endpoint: {getattr(e, 'message', str(e))}",
                    )

        # Process the request and get the response
        response = await call_next(request)

        return response

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

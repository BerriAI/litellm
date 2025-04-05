"""
Prometheus Auth Middleware
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

import litellm
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class PrometheusAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if this is a request to the metrics endpoint

        if self._is_prometheus_metrics_endpoint(request):
            try:
                await user_api_key_auth(
                    request=request, api_key=request.headers.get("Authorization") or ""
                )
            except Exception as e:
                raise e

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
        return litellm.require_auth_for_metrics_endpoint

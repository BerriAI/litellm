"""Reject requests not bearing the Cloudflare-injected origin secret header.

The gateway runs behind Cloudflare. Cloudflare's Transform Rule injects
``x-rayward-cf-secret`` on every request that goes to origin. Origin sees the
real Cloudflare IP only because Container Apps' ingress IP-allowlist already
pre-filters; the secret header is the second control that makes Cloudflare-IP
spoofing alone insufficient to reach the gateway.

Health endpoints are exempted so Container Apps' liveness / readiness probes
keep working (they originate from inside the Container Apps environment, not
through Cloudflare).

The expected secret is read from ``CF_ORIGIN_SECRET`` at startup. The gateway
refuses to start if the env var is unset, instead of silently treating any
request as authorized.
"""

import os
from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp


class CloudflareOriginSecretMiddleware(BaseHTTPMiddleware):
    HEADER_NAME = "x-rayward-cf-secret"
    DEFAULT_EXEMPT_PATHS = ("/health/liveness", "/health/readiness")

    def __init__(
        self,
        app: ASGIApp,
        *,
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        super().__init__(app)
        self._expected = os.environ.get("CF_ORIGIN_SECRET")
        if not self._expected:
            raise RuntimeError(
                "CF_ORIGIN_SECRET env var not set; "
                "CloudflareOriginSecretMiddleware refuses to start. "
                "Either provide the secret via Key Vault → Container Apps env, "
                "or remove the middleware from proxy_server.py if running "
                "the gateway in a non-Cloudflare context."
            )
        self._exempt = set(exempt_paths)

    async def dispatch(self, request, call_next):
        if request.url.path in self._exempt:
            return await call_next(request)
        provided = request.headers.get(self.HEADER_NAME)
        if provided is None or provided != self._expected:
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "type": "origin_unauthorized",
                        "message": (
                            "missing or invalid origin secret; "
                            "this gateway only accepts traffic forwarded by Cloudflare"
                        ),
                    }
                },
            )
        return await call_next(request)

"""
Adds anti-framing / content-type security headers to every HTTP response.

X-Frame-Options and Content-Security-Policy: frame-ancestors 'none' stop the
admin UI and login pages from being embedded cross-origin (clickjacking).
X-Content-Type-Options: nosniff stops MIME sniffing.

Strict-Transport-Security is opt-in via LITELLM_ENABLE_HSTS because it only
makes sense over HTTPS and would lock browsers out of plain-http deployments.

Headers are set with setdefault so a route that intentionally sets its own
value is never overridden.
"""

import os

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

STATIC_SECURITY_HEADERS = (
    ("X-Frame-Options", "DENY"),
    ("Content-Security-Policy", "frame-ancestors 'none'"),
    ("X-Content-Type-Options", "nosniff"),
)
HSTS_HEADER = ("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


def _hsts_enabled() -> bool:
    return os.getenv("LITELLM_ENABLE_HSTS", "false").strip().lower() == "true"


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                applied = (*STATIC_SECURITY_HEADERS, HSTS_HEADER) if _hsts_enabled() else STATIC_SECURITY_HEADERS
                for name, value in applied:
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)

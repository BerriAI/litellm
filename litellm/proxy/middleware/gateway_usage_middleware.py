"""
ASGI middleware that counts total and successful LLM-route requests for
request-based billing via ``LITELLM_USAGE_ENDPOINT``.

Billable request definition: any request whose gateway response status code
is in the range [1, 500) is counted as "successful" (i.e. the gateway
handled it). 4xx responses (auth errors, bad requests, rate limits) are
included because those are client-side errors and the gateway still handled
them. 5xx responses and requests where no ``http.response.start`` was sent
(status_code remains 0, e.g. client disconnect before response) are counted
as "failed".

Only active when ``LITELLM_USAGE_ENDPOINT`` is set to avoid overhead on
OSS / default deployments.
"""

from __future__ import annotations

import os

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.proxy.auth.route_checks import RouteChecks

_GATEWAY_HANDLED_BELOW = 500


class GatewayUsageMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._enabled = bool(os.environ.get("LITELLM_USAGE_ENDPOINT"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._enabled or scope["type"] != "http" or not self._is_llm_route(scope):
            await self.app(scope, receive, send)
            return

        status_code: int = 0

        async def _send_wrapper(message: dict[str, object]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 0))
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        finally:
            from litellm.proxy.usage_reporting.gateway_usage_reporter import record_request

            await record_request(succeeded=0 < status_code < _GATEWAY_HANDLED_BELOW)

    @staticmethod
    def _is_llm_route(scope: Scope) -> bool:
        path: str = scope.get("path", "")
        return RouteChecks.is_llm_api_route(route=path)

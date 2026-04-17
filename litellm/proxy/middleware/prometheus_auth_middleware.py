"""
Prometheus Auth Middleware - Pure ASGI implementation
"""
import json
from typing import List

from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

import litellm
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# Cache the header name at module level to avoid repeated enum attribute access
_AUTHORIZATION_HEADER = SpecialHeaders.openai_authorization.value  # "Authorization"


class PrometheusAuthMiddleware:
    """
    Middleware to authenticate requests to the metrics endpoint.

    By default, auth is not run on the metrics endpoint.

    Enabled by setting the following in proxy_config.yaml:

    ```yaml
    litellm_settings:
        require_auth_for_metrics_endpoint: true
    ```
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Fast path: only inspect HTTP requests; pass through websocket/lifespan immediately
        if scope["type"] != "http" or "/metrics" not in scope.get("path", ""):
            await self.app(scope, receive, send)
            return

        # Only run auth if configured to do so
        if litellm.require_auth_for_metrics_endpoint is True:
            # user_api_key_auth reads the request body, which consumes ASGI `receive`.
            # Buffer those messages and replay them for the inner app; otherwise a
            # successful auth would forward an exhausted receive and /metrics hangs.
            buffered_messages: List[dict] = []

            async def receive_for_auth() -> dict:
                message = await receive()
                buffered_messages.append(message)
                return message

            request = Request(scope, receive_for_auth)
            api_key = request.headers.get(_AUTHORIZATION_HEADER) or ""

            try:
                await user_api_key_auth(request=request, api_key=api_key)
            except Exception as e:
                # Send 401 response directly via ASGI protocol
                error_message = getattr(e, "message", str(e))
                body = json.dumps(
                    f"Unauthorized access to metrics endpoint: {error_message}"
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            [b"content-type", b"application/json"],
                            [b"content-length", str(len(body)).encode("ascii")],
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                    }
                )
                return

            replay_idx = 0

            async def receive_replay() -> dict:
                nonlocal replay_idx
                if replay_idx < len(buffered_messages):
                    msg = buffered_messages[replay_idx]
                    replay_idx += 1
                    return msg
                return await receive()

            await self.app(scope, receive_replay, send)
            return

        # Pass through to the inner application
        await self.app(scope, receive, send)

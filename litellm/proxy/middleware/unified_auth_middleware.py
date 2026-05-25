"""
Unified Authentication Middleware for LiteLLM Proxy

This middleware provides consistent authentication for all endpoints,
following FastAPI best practices from https://fastapi.tiangolo.com/tutorial/security/

This middleware consolidates the various auth approaches in LiteLLM:
1. PrometheusAuthMiddleware - for /metrics endpoint
2. MCP auth - for MCP server endpoints
3. Standard Depends(user_api_key_auth) - for all other endpoints

Usage:
    from litellm.proxy.middleware.unified_auth_middleware import UnifiedAuthMiddleware

    app.add_middleware(UnifiedAuthMiddleware)
"""

import json
from typing import Any, Callable, List, MutableMapping, Optional, Set

from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

import litellm
from litellm._logging import verbose_logger
from litellm.proxy._types import SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_request_route,
    route_in_additonal_public_routes,
)


class UnifiedAuthMiddleware:
    """
    ASGI middleware that provides unified authentication for all endpoints.

    This middleware:
    1. Intercepts all HTTP requests
    2. Applies consistent auth logic based on route patterns
    3. Handles special cases (public routes, /metrics, MCP, etc.)
    4. Stores auth context for downstream handlers

    The middleware follows FastAPI's recommended patterns while providing
    backward compatibility with existing auth configurations.
    """

    # Routes that should always be public (no auth required)
    DEFAULT_PUBLIC_ROUTES: Set[str] = {
        "/",
        "/health",
        "/health/liveliness",
        "/health/liveness",
        "/health/readiness",
        "/openapi.json",
        "/docs",
        "/redoc",
    }

    # Route prefixes that are public
    PUBLIC_ROUTE_PREFIXES: tuple = ("/.well-known/",)

    def __init__(
        self,
        app: ASGIApp,
        *,
        exclude_routes: Optional[Set[str]] = None,
        include_routes: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the unified auth middleware.

        Args:
            app: The ASGI application to wrap
            exclude_routes: Routes to exclude from auth (added to default public routes)
            include_routes: Routes to always require auth (overrides public routes)
        """
        self.app = app
        self.exclude_routes = exclude_routes or set()
        self.include_routes = include_routes or set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process incoming requests and apply unified auth.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if self._is_public_route(path):
            await self.app(scope, receive, send)
            return

        if self._requires_special_auth(path):
            await self._handle_special_auth(scope, receive, send, path)
            return

        await self.app(scope, receive, send)

    def _is_public_route(self, path: str) -> bool:
        """
        Check if the route is public (no auth required).
        """
        normalized_path = path.rstrip("/") if path != "/" else path

        if normalized_path in self.include_routes:
            return False

        if normalized_path in self.DEFAULT_PUBLIC_ROUTES:
            return True

        if normalized_path in self.exclude_routes:
            return True

        if path.startswith(self.PUBLIC_ROUTE_PREFIXES):
            return True

        if route_in_additonal_public_routes(normalized_path):
            return True

        return False

    def _requires_special_auth(self, path: str) -> bool:
        """
        Check if the route requires special auth handling.

        Currently this includes:
        - /metrics (when require_auth_for_metrics_endpoint is True)
        """
        if "/metrics" in path:
            return litellm.require_auth_for_metrics_endpoint is not False

        return False

    async def _handle_special_auth(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        path: str,
    ) -> None:
        """
        Handle special auth cases like /metrics endpoint.
        """
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        buffered_messages: List[MutableMapping[str, Any]] = []

        async def receive_for_auth() -> MutableMapping[str, Any]:
            message = await receive()
            buffered_messages.append(message)
            return message

        request = Request(scope, receive_for_auth)

        try:
            await user_api_key_auth(
                request=request,
                api_key=request.headers.get(SpecialHeaders.openai_authorization.value)
                or "",
                azure_api_key_header=request.headers.get(
                    SpecialHeaders.azure_authorization.value
                )
                or "",
                anthropic_api_key_header=request.headers.get(
                    SpecialHeaders.anthropic_authorization.value
                ),
                google_ai_studio_api_key_header=request.headers.get(
                    SpecialHeaders.google_ai_studio_authorization.value
                ),
                azure_apim_header=request.headers.get(
                    SpecialHeaders.azure_apim_authorization.value
                )
                or "",
                custom_litellm_key_header=request.headers.get(
                    SpecialHeaders.custom_litellm_api_key.value
                ),
            )
        except Exception as e:
            error_message = getattr(e, "message", str(e))
            body = json.dumps(
                {
                    "error": "Unauthorized",
                    "message": f"Authentication failed: {error_message}",
                    "hint": "Provide a valid API key in the Authorization header.",
                }
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

        async def receive_replay() -> MutableMapping[str, Any]:
            nonlocal replay_idx
            if replay_idx < len(buffered_messages):
                msg = buffered_messages[replay_idx]
                replay_idx += 1
                return msg
            return await receive()

        await self.app(scope, receive_replay, send)


class MCPAuthMiddleware:
    """
    ASGI middleware specifically for MCP endpoints.

    This middleware provides MCP-specific auth handling while using
    the same underlying auth system as other endpoints, ensuring
    consistency across the proxy.

    Key features:
    1. Uses the same user_api_key_auth function as standard endpoints
    2. Supports OAuth2 passthrough for configured servers
    3. Sets auth context for MCP tool handlers
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process MCP requests with unified auth.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            scope["state"] = scope.get("state", {})
            scope["state"]["user_api_key_auth"] = user_api_key_auth
            scope["state"]["mcp_auth_header"] = mcp_auth_header
            scope["state"]["mcp_servers"] = mcp_servers
            scope["state"]["mcp_server_auth_headers"] = mcp_server_auth_headers
            scope["state"]["oauth2_headers"] = oauth2_headers
            scope["state"]["raw_headers"] = raw_headers

        except Exception as e:
            verbose_logger.exception(f"MCP auth failed: {e}")
            error_message = getattr(e, "detail", getattr(e, "message", str(e)))
            status_code = getattr(e, "status_code", 401)
            body = json.dumps(
                {
                    "error": "Authentication failed",
                    "message": error_message,
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": status_code,
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

        await self.app(scope, receive, send)

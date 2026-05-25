"""
Unified Authentication Module for LiteLLM Proxy

This module provides a centralized authentication system following FastAPI best practices
as documented at https://fastapi.tiangolo.com/tutorial/security/

Key features:
1. Uses FastAPI's built-in security schemes (OAuth2, APIKey, etc.)
2. Provides consistent auth across all endpoints (including MCP)
3. Supports multiple auth header patterns for backward compatibility
4. Integrates with OpenAPI for proper documentation

Usage:
    from litellm.proxy.auth.unified_auth import UnifiedAuth

    # As a FastAPI dependency
    @router.get("/endpoint")
    async def endpoint(auth: UserAPIKeyAuth = Depends(UnifiedAuth())):
        ...

    # Or use the global instance
    from litellm.proxy.auth.unified_auth import unified_auth_dependency

    @router.get("/endpoint", dependencies=[Depends(unified_auth_dependency)])
    async def endpoint():
        ...
"""

from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security.base import SecurityBase

from litellm._logging import verbose_logger
from litellm.proxy._types import (
    ProxyException,
    SpecialHeaders,
    UserAPIKeyAuth,
)


class LiteLLMSecurityScheme(SecurityBase):
    """
    Custom security scheme that supports multiple authentication methods:
    - API Key (via various headers)
    - Bearer Token
    - JWT
    - OAuth2

    This follows FastAPI's security patterns while supporting LiteLLM's
    flexible authentication requirements.
    """

    def __init__(
        self,
        *,
        scheme_name: Optional[str] = "LiteLLM API Key",
        description: str = "API key authentication supporting multiple header formats",
        auto_error: bool = True,
    ):
        self.scheme_name = scheme_name
        self.description = description
        self.auto_error = auto_error
        self.model = {
            "type": "apiKey",
            "in": "header",
            "name": SpecialHeaders.openai_authorization.value,
            "description": description,
        }

    async def __call__(self, request: Request) -> Optional[str]:
        """Extract API key from request headers using LiteLLM's multi-header pattern."""
        return self._extract_api_key_from_headers(request.headers)

    def _extract_api_key_from_headers(self, headers) -> Optional[str]:
        """
        Extract API key from headers, checking multiple header names for compatibility.

        Priority order:
        1. x-litellm-api-key (custom LiteLLM header)
        2. Authorization (standard Bearer token)
        3. api-key (Azure-style)
        4. x-api-key (Anthropic-style)
        5. x-goog-api-key (Google AI Studio)
        6. Ocp-Apim-Subscription-Key (Azure APIM)
        """
        api_key = headers.get(SpecialHeaders.custom_litellm_api_key.value)
        if api_key:
            return api_key

        api_key = headers.get(SpecialHeaders.openai_authorization.value)
        if api_key:
            return api_key

        api_key = headers.get(SpecialHeaders.azure_authorization.value)
        if api_key:
            return api_key

        api_key = headers.get(SpecialHeaders.anthropic_authorization.value)
        if api_key:
            return api_key

        api_key = headers.get(SpecialHeaders.google_ai_studio_authorization.value)
        if api_key:
            return api_key

        api_key = headers.get(SpecialHeaders.azure_apim_authorization.value)
        if api_key:
            return api_key

        return None


class UnifiedAuth:
    """
    Unified authentication dependency for FastAPI routes.

    This class provides a callable that can be used as a FastAPI dependency
    to authenticate requests across all endpoints, including MCP.

    Example:
        @router.get("/endpoint")
        async def endpoint(auth: UserAPIKeyAuth = Depends(UnifiedAuth())):
            return {"user_id": auth.user_id}
    """

    def __init__(
        self,
        *,
        auto_error: bool = True,
        allow_public_routes: bool = True,
        require_api_key: bool = False,
    ):
        """
        Initialize the unified auth dependency.

        Args:
            auto_error: If True, raise HTTPException on auth failure
            allow_public_routes: If True, allow unauthenticated access to public routes
            require_api_key: If True, always require an API key (no public routes)
        """
        self.auto_error = auto_error
        self.allow_public_routes = allow_public_routes
        self.require_api_key = require_api_key
        self.security_scheme = LiteLLMSecurityScheme(auto_error=auto_error)

    async def __call__(
        self,
        request: Request,
    ) -> UserAPIKeyAuth:
        """
        Authenticate the request and return UserAPIKeyAuth.

        This method:
        1. Extracts the API key from headers
        2. Validates the key using the existing user_api_key_auth function
        3. Returns the authenticated user context

        Args:
            request: The FastAPI request object

        Returns:
            UserAPIKeyAuth: The authenticated user context

        Raises:
            HTTPException: If authentication fails and auto_error is True
        """
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        api_key = await self.security_scheme(request)
        if api_key is None:
            api_key = ""

        try:
            return await user_api_key_auth(
                request=request,
                api_key=api_key,
                azure_api_key_header=request.headers.get(
                    SpecialHeaders.azure_authorization.value
                ),
                anthropic_api_key_header=request.headers.get(
                    SpecialHeaders.anthropic_authorization.value
                ),
                google_ai_studio_api_key_header=request.headers.get(
                    SpecialHeaders.google_ai_studio_authorization.value
                ),
                azure_apim_header=request.headers.get(
                    SpecialHeaders.azure_apim_authorization.value
                ),
                custom_litellm_key_header=request.headers.get(
                    SpecialHeaders.custom_litellm_api_key.value
                ),
            )
        except HTTPException:
            if self.auto_error:
                raise
            return UserAPIKeyAuth()
        except ProxyException as e:
            if self.auto_error:
                raise HTTPException(
                    status_code=(
                        int(e.code) if e.code and str(e.code).isdigit() else 401
                    ),
                    detail=e.message,
                )
            return UserAPIKeyAuth()


class UnifiedAuthForMCP(UnifiedAuth):
    """
    Extended unified auth specifically for MCP endpoints.

    This class extends UnifiedAuth to handle MCP-specific auth scenarios:
    1. OAuth2 token passthrough for upstream MCP servers
    2. Server-specific auth headers
    3. Access group restrictions

    The goal is to consolidate MCP auth with the standard auth flow while
    preserving MCP-specific functionality.
    """

    def __init__(
        self,
        *,
        auto_error: bool = True,
        allow_oauth2_passthrough: bool = True,
    ):
        """
        Initialize MCP-specific auth.

        Args:
            auto_error: If True, raise HTTPException on auth failure
            allow_oauth2_passthrough: If True, allow OAuth2 token passthrough
                for servers configured with delegate_auth_to_upstream
        """
        super().__init__(auto_error=auto_error)
        self.allow_oauth2_passthrough = allow_oauth2_passthrough

    async def __call__(
        self,
        request: Request,
    ) -> UserAPIKeyAuth:
        """
        Authenticate MCP request with OAuth2 passthrough support.

        For MCP endpoints, we support an additional auth pattern where
        operators can configure servers to delegate auth to upstream,
        allowing clients to authenticate directly with the MCP server
        using OAuth2 PKCE flow.
        """
        from litellm.proxy.auth.auth_utils import get_request_route

        api_key = await self.security_scheme(request)
        route = get_request_route(request)

        if self.allow_oauth2_passthrough and not api_key:
            if self._should_delegate_to_upstream(route, request):
                verbose_logger.debug(
                    "MCP OAuth2 passthrough: delegating auth to upstream server"
                )
                return UserAPIKeyAuth()

        return await super().__call__(request)

    def _should_delegate_to_upstream(self, route: str, request: Request) -> bool:
        """
        Check if this request should bypass LiteLLM auth and delegate to upstream.

        This is only allowed when:
        1. No LiteLLM API key is provided
        2. All target MCP servers are OAuth2-configured with delegate_auth_to_upstream
        3. None of the servers use client_credentials grant (M2M)
        """
        try:
            from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
                MCPRequestHandler,
            )

            mcp_servers_header = request.headers.get(SpecialHeaders.mcp_servers.value)
            mcp_servers = None
            if mcp_servers_header:
                mcp_servers = [
                    s.strip() for s in mcp_servers_header.split(",") if s.strip()
                ]

            return MCPRequestHandler._target_servers_delegate_auth_to_upstream(
                path=route, mcp_servers=mcp_servers
            )
        except Exception as e:
            verbose_logger.debug(f"Error checking OAuth2 delegation: {e}")
            return False


unified_auth_dependency = UnifiedAuth()
unified_auth_mcp_dependency = UnifiedAuthForMCP()


async def get_unified_auth(request: Request) -> UserAPIKeyAuth:
    """
    Convenience function for getting unified auth as a dependency.

    Usage:
        @router.get("/endpoint")
        async def endpoint(auth: UserAPIKeyAuth = Depends(get_unified_auth)):
            ...
    """
    return await unified_auth_dependency(request)


async def get_unified_auth_mcp(request: Request) -> UserAPIKeyAuth:
    """
    Convenience function for getting unified MCP auth as a dependency.

    Usage:
        @router.get("/mcp/endpoint")
        async def endpoint(auth: UserAPIKeyAuth = Depends(get_unified_auth_mcp)):
            ...
    """
    return await unified_auth_mcp_dependency(request)

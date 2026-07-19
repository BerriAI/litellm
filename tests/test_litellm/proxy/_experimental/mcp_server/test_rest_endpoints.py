import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.auth import (
    user_api_key_auth_mcp as auth_mcp,
)
from litellm.proxy._types import (
    NewMCPServerRequest,
    UpdateMCPServerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.mcp import MCPAuth


def _build_request(
    headers: Optional[Dict[str, str]] = None,
    *,
    path: str = "/mcp-rest/test/tools/list",
    method: str = "POST",
    json_body: Optional[Any] = None,
    body: Optional[bytes] = None,
) -> Request:
    headers = headers or {}
    if json_body is not None:
        body_bytes = json.dumps(json_body).encode("utf-8")
    elif body is not None:
        body_bytes = body
    else:
        body_bytes = b""
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "headers": raw_headers,
    }

    state = {"sent": False}

    async def receive():
        if state["sent"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        state["sent"] = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    return Request(scope, receive=receive)


def _get_route(path: str, method: str):
    for route in rest_endpoints.router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"Route {method} {path} not found")


def _route_has_dependency(route, dependency) -> bool:
    if any(getattr(dep, "dependency", None) == dependency for dep in getattr(route, "dependencies", [])):
        return True
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    return any(getattr(dep, "call", None) == dependency for dep in dependant.dependencies)


class TestExecuteWithMcpClient:
    @pytest.mark.asyncio
    async def test_redacts_stack_trace(self, monkeypatch):
        async def fake_create_client(*args, **kwargs):
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
        )

        async def failing_operation(client):
            raise RuntimeError("boom")

        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=MCPAuth.none,
        )

        result = await rest_endpoints._execute_with_mcp_client(payload, failing_operation)

        assert result["status"] == "error"
        assert "stack_trace" not in result

    @pytest.mark.asyncio
    async def test_forwards_static_headers(self, monkeypatch):
        """Ensure static_headers are forwarded to the MCP client during test calls.

        This is required for `/mcp-rest/test/tools/list` (Issue #19341), where the UI
        sends `static_headers` but the backend must forward them during
        `session.initialize()` and tool discovery.
        """
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured["extra_headers"] = kwargs.get("extra_headers")
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=MCPAuth.none,
            static_headers={"Authorization": "STATIC token"},
        )

        result = await rest_endpoints._execute_with_mcp_client(
            payload,
            ok_operation,
            oauth2_headers={"X-OAuth": "1"},
            raw_headers={"x-test": "y"},
        )

        assert result["status"] == "ok"
        assert captured["extra_headers"] == {
            "X-OAuth": "1",
            "Authorization": "STATIC token",
        }

    @pytest.mark.asyncio
    async def test_m2m_credentials_forwarded_to_server_model(self, monkeypatch):
        """M2M OAuth credentials (client_id, client_secret) from the nested
        ``credentials`` dict must be forwarded to the MCPServer model so that
        ``has_client_credentials`` returns True and the proxy auto-fetches tokens."""
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured["server"] = kwargs.get("server")
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="m2m-server",
            url="https://example.com",
            auth_type=MCPAuth.oauth2,
            token_url="https://auth.example.com/token",
            credentials={
                "client_id": "my-id",
                "client_secret": "my-secret",
                "scopes": ["read", "write"],
            },
        )

        result = await rest_endpoints._execute_with_mcp_client(payload, ok_operation)

        assert result["status"] == "ok"
        server = captured["server"]
        assert server.client_id == "my-id"
        assert server.client_secret == "my-secret"
        assert server.token_url == "https://auth.example.com/token"
        assert server.scopes == ["read", "write"]
        assert server.has_client_credentials is True

    @pytest.mark.asyncio
    async def test_m2m_drops_incoming_oauth2_headers(self, monkeypatch):
        """For M2M OAuth servers the incoming Authorization header (which carries
        the litellm API key) must NOT be forwarded as extra_headers — otherwise
        it overwrites the auto-fetched M2M token."""
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured["extra_headers"] = kwargs.get("extra_headers")
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="m2m-server",
            url="https://example.com",
            auth_type=MCPAuth.oauth2,
            token_url="https://auth.example.com/token",
            credentials={
                "client_id": "my-id",
                "client_secret": "my-secret",
            },
        )

        incoming_oauth2 = {"Authorization": "Bearer sk-litellm-api-key"}
        result = await rest_endpoints._execute_with_mcp_client(
            payload,
            ok_operation,
            oauth2_headers=incoming_oauth2,
        )

        assert result["status"] == "ok"
        # The incoming Authorization must be dropped — extra_headers should
        # contain no oauth2 headers (only static_headers, which are None here).
        assert captured["extra_headers"] is None or "Authorization" not in captured["extra_headers"]

    @pytest.mark.asyncio
    async def test_interactive_oauth_resolves_forwarded_token_via_presented_store(self, monkeypatch):
        """Interactive authorization_code preview (oauth2, no client credentials): the forwarded
        just-authorized token is resolved THROUGH the v2 resolver via a one-shot presented store
        (cred_provider), not the caller-override path. The bare token (Bearer stripped) is the
        upstream credential and is not also forwarded in extra_headers."""
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="linear",
            url="https://mcp.linear.app/mcp",
            auth_type=MCPAuth.oauth2,
            authorization_url="https://mcp.linear.app/authorize",
        )

        result = await rest_endpoints._execute_with_mcp_client(
            payload,
            ok_operation,
            oauth2_headers={"Authorization": "Bearer forwarded-user-token"},
        )

        assert result["status"] == "ok"
        # Resolved via the v2 resolver, never the caller-override header
        assert captured["mcp_auth_header"] is None
        provider = captured["cred_provider"]
        assert provider is not None
        token = await provider._oauth_token_store.fetch("u", "s")
        assert token is not None and token.access_token == "forwarded-user-token"
        # The resolver supplies the bearer, so it is not also forwarded as a caller header
        extra_headers = captured.get("extra_headers") or {}
        assert not any(k.lower() == "authorization" for k in extra_headers)

    @pytest.mark.asyncio
    async def test_m2m_does_not_build_presented_store(self, monkeypatch):
        """M2M (client_credentials): to_server_spec returns None, so no presented provider is built;
        the auto-fetch path is unchanged (no cred_provider, the incoming header dropped as before).
        """
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="m2m-server",
            url="https://example.com",
            auth_type=MCPAuth.oauth2,
            token_url="https://auth.example.com/token",
            credentials={"client_id": "my-id", "client_secret": "my-secret"},
        )

        result = await rest_endpoints._execute_with_mcp_client(
            payload,
            ok_operation,
            oauth2_headers={"Authorization": "Bearer sk-litellm-api-key"},
        )

        assert result["status"] == "ok"
        assert captured.get("cred_provider") is None
        assert captured["mcp_auth_header"] is None

    @pytest.mark.asyncio
    async def test_token_exchange_does_not_build_presented_store(self, monkeypatch):
        """OBO / token-exchange (auth_type oauth2_token_exchange, not oauth2): excluded by the
        auth_type == oauth2 guard, so no presented provider is built and the v1 exchange path runs.
        """
        captured: dict = {}

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="obo-server",
            url="https://example.com",
            auth_type=MCPAuth.oauth2_token_exchange,
            token_url="https://auth.example.com/token",
        )

        result = await rest_endpoints._execute_with_mcp_client(
            payload,
            ok_operation,
            oauth2_headers={"Authorization": "Bearer subject-jwt"},
        )

        assert result["status"] == "ok"
        assert captured.get("cred_provider") is None

    @pytest.mark.asyncio
    async def test_catches_exception_group(self, monkeypatch):
        """MCP SDK's anyio TaskGroup raises BaseExceptionGroup which does not
        inherit from Exception.  The handler must catch it and return an error
        dict instead of letting a raw 500 propagate."""

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            raise BaseExceptionGroup("test group", [RuntimeError("Cancelled via cancel scope")])

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_build_stdio_env",
            fake_build_stdio_env,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_create_mcp_client",
            fake_create_client,
            raising=False,
        )

        async def ok_operation(client):
            return {"status": "ok"}

        payload = NewMCPServerRequest(
            server_name="bad-server",
            url="https://example.com",
            auth_type=MCPAuth.none,
        )

        result = await rest_endpoints._execute_with_mcp_client(payload, ok_operation)

        assert result["status"] == "error"
        assert result["error"] is True
        assert "Failed to connect to MCP server" in result["message"]
        # Error message must not leak raw exception details
        assert "cancel scope" not in result["message"]


class TestTestConnection:
    def test_requires_auth_dependency(self):
        route = _get_route("/mcp-rest/test/connection", "POST")
        assert _route_has_dependency(route, user_api_key_auth)


class TestTestToolsList:
    pytestmark = pytest.mark.asyncio

    async def test_forwards_mcp_auth_header(self, monkeypatch):
        """Ensure credential-based auth forwards the auth_value to the MCP client."""

        captured: dict = {}

        async def fake_execute(
            request,
            operation,
            mcp_auth_header=None,
            oauth2_headers=None,
            raw_headers=None,
        ):
            captured["mcp_auth_header"] = mcp_auth_header
            captured["oauth2_headers"] = oauth2_headers
            return {
                "tools": [],
                "error": None,
                "message": "Successfully retrieved tools",
            }

        monkeypatch.setattr(rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False)

        oauth_call_counter = {"count": 0}

        def fake_oauth(headers):
            oauth_call_counter["count"] += 1
            return {"Authorization": "Bearer oauth"}

        monkeypatch.setattr(
            auth_mcp.MCPRequestHandler,
            "_get_oauth2_headers_from_headers",
            staticmethod(fake_oauth),
            raising=False,
        )

        request = _build_request()
        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=MCPAuth.api_key,
            credentials={"auth_value": "secret-key"},
        )

        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        assert result["message"] == "Successfully retrieved tools"
        assert captured["mcp_auth_header"] == "secret-key"
        assert captured["oauth2_headers"] is None
        assert oauth_call_counter["count"] == 0

    async def test_extracts_oauth2_headers(self, monkeypatch):
        """Ensure oauth2 auth type pulls oauth headers and omits MCP auth header."""

        captured: dict = {}

        async def fake_execute(
            request,
            operation,
            mcp_auth_header=None,
            oauth2_headers=None,
            raw_headers=None,
        ):
            captured["mcp_auth_header"] = mcp_auth_header
            captured["oauth2_headers"] = oauth2_headers
            return {
                "tools": [],
                "error": None,
                "message": "Successfully retrieved tools",
            }

        monkeypatch.setattr(rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False)

        oauth_headers = {"Authorization": "Bearer oauth"}
        oauth_call_counter = {"count": 0}

        def fake_oauth(headers):
            oauth_call_counter["count"] += 1
            return oauth_headers

        monkeypatch.setattr(
            auth_mcp.MCPRequestHandler,
            "_get_oauth2_headers_from_headers",
            staticmethod(fake_oauth),
            raising=False,
        )

        request = _build_request({"authorization": "Bearer incoming", "x-litellm-api-key": "sk-admission"})
        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=MCPAuth.oauth2,
        )

        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        assert result["message"] == "Successfully retrieved tools"
        assert captured["mcp_auth_header"] is None
        assert captured["oauth2_headers"] == oauth_headers
        assert oauth_call_counter["count"] == 1

    @pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
    async def test_extracts_oauth2_headers_for_client_forwarded_modes(self, monkeypatch, auth_type):
        """The browser-only authorize flow sends the upstream token as Authorization; the preview
        must thread it through for the client-forwarded token modes so the passthrough arm can
        forward it, instead of probing the upstream unauthenticated."""

        captured: dict = {}

        async def fake_execute(
            request,
            operation,
            mcp_auth_header=None,
            oauth2_headers=None,
            raw_headers=None,
        ):
            captured["mcp_auth_header"] = mcp_auth_header
            captured["oauth2_headers"] = oauth2_headers
            return {
                "tools": [],
                "error": None,
                "message": "Successfully retrieved tools",
            }

        monkeypatch.setattr(rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False)

        oauth_headers = {"Authorization": "Bearer upstream-token"}

        monkeypatch.setattr(
            auth_mcp.MCPRequestHandler,
            "_get_oauth2_headers_from_headers",
            staticmethod(lambda headers: oauth_headers),
            raising=False,
        )

        request = _build_request({"authorization": "Bearer upstream-token", "x-litellm-api-key": "sk-admission"})
        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=auth_type,
        )

        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        assert result["message"] == "Successfully retrieved tools"
        assert captured["mcp_auth_header"] is None
        assert captured["oauth2_headers"] == oauth_headers

    @pytest.mark.parametrize("auth_type", [MCPAuth.oauth2, MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
    async def test_does_not_forward_authorization_that_satisfied_admission(self, monkeypatch, auth_type):
        """Authorization is also the admission fallback: with no x-litellm-api-key on the request,
        the Authorization value is the caller's LiteLLM key, so forwarding it would send the
        admission credential to the upstream."""

        captured: dict = {}

        async def fake_execute(
            request,
            operation,
            mcp_auth_header=None,
            oauth2_headers=None,
            raw_headers=None,
        ):
            captured["oauth2_headers"] = oauth2_headers
            return {
                "tools": [],
                "error": None,
                "message": "Successfully retrieved tools",
            }

        monkeypatch.setattr(rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False)

        request = _build_request({"authorization": "Bearer sk-litellm-admission-key"})
        payload = NewMCPServerRequest(
            server_name="example",
            url="https://example.com",
            auth_type=auth_type,
        )

        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        assert result["message"] == "Successfully retrieved tools"
        assert captured["oauth2_headers"] is None


class TestListToolsRestAPI:
    pytestmark = pytest.mark.asyncio

    async def test_rejects_disallowed_server(self, monkeypatch):
        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return []

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id="server-1",
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert result["tools"] == []
        assert result["error"] == "unexpected_error"
        assert "access_denied" in result["message"]
        assert "server server-1" in result["message"]

    async def test_lists_tools_for_allowed_server(self, monkeypatch):
        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        class StubServer:
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = None
            mcp_info = {"server_name": "stub"}
            available_on_public_internet = True

        stub_server = StubServer()

        captured = {"called": False}

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
            apply_tool_filters=True,
        ):
            captured["called"] = True
            captured["server"] = server
            captured["auth_header"] = server_auth_header
            return ["tool-1"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id="server-1",
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert captured["called"] is True
        assert captured["server"] is stub_server
        assert result["tools"] == ["tool-1"]
        assert result["error"] is None
        assert result["message"] == "Successfully retrieved tools"

    async def test_include_disabled_tools_is_admin_only(self, monkeypatch):
        """include_disabled_tools skips the allowlist filter only for PROXY_ADMIN;
        a non-admin passing it stays filtered so the REST endpoint can't be used
        to enumerate deliberately-disabled tools."""
        from litellm.proxy._types import LitellmUserRoles

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        class StubServer:
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = ["tool1"]
            mcp_info = {"server_name": "stub"}
            available_on_public_internet = True

        stub_server = StubServer()
        captured = {}

        async def fake_get_tools(server, server_auth_header, *args, apply_tool_filters=True, **kwargs):
            captured["apply_tool_filters"] = apply_tool_filters
            return ["tool-1"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")

        await rest_endpoints.list_tool_rest_api(
            request,
            server_id="server-1",
            include_disabled_tools=True,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
        assert captured["apply_tool_filters"] is False

        await rest_endpoints.list_tool_rest_api(
            request,
            server_id="server-1",
            include_disabled_tools=True,
            user_api_key_dict=UserAPIKeyAuth(),
        )
        assert captured["apply_tool_filters"] is True

    @pytest.mark.parametrize("upstream_status", [401, 403])
    async def test_upstream_auth_failure_surfaces_status_and_challenge(self, monkeypatch, upstream_status):
        """A single-server pass-through request whose upstream rejects the token
        must surface the upstream status (401 or 403) plus its WWW-Authenticate
        challenge, not collapse into a 200 ``unexpected_error`` body."""
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )

        class StubServer:
            alias = "server-1"
            server_name = "server-1"
            name = "passthrough"
            allowed_tools = None
            mcp_info = {"server_name": "passthrough"}
            available_on_public_internet = True

        stub_server = StubServer()

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        challenge = 'Bearer resource_metadata="https://upstream/.well-known"'

        async def fake_get_tools(*args, **kwargs):
            raise MCPUpstreamAuthError(
                status_code=upstream_status,
                www_authenticate=challenge,
                server_name="passthrough",
            )

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.list_tool_rest_api(
                request,
                server_id="server-1",
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert exc_info.value.status_code == upstream_status
        assert exc_info.value.headers == {"www-authenticate": challenge}

    async def test_aggregate_list_absorbs_one_server_auth_failure(self, monkeypatch):
        """The multi-server aggregate listing degrades a server whose upstream
        rejects auth to an empty contribution and still returns the healthy
        server's tools with a 200, rather than surfacing a 401."""
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )

        class StubServer:
            def __init__(self, name):
                self.alias = name
                self.server_name = name
                self.name = name
                self.allowed_tools = None
                self.mcp_info = {"server_name": name}
                self.available_on_public_internet = True

        good = StubServer("good")
        bad = StubServer("bad")

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["good", "bad"]

        async def fake_get_tools(server, *args, **kwargs):
            if server.server_name == "bad":
                raise MCPUpstreamAuthError(
                    status_code=401,
                    www_authenticate='Bearer realm="x"',
                    server_name="bad",
                )
            return ["good-tool"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: {"good": good, "bad": bad}.get(server_id),
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id=None,
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert result["tools"] == ["good-tool"]
        assert result["error"] is None

    async def test_name_resolution_finds_server_by_uuid(self, monkeypatch):
        """When server_id is a name string, it should be resolved to its UUID
        and used for the tools lookup when the UUID is in allowed_server_ids."""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        stub_server = MCPServer(
            server_id="uuid-abc-123",
            name="my-server",
            transport=MCPTransport.sse,
        )
        stub_server.alias = "my-server"
        stub_server.server_name = "my-server"
        stub_server.available_on_public_internet = True
        stub_server.allowed_tools = None
        stub_server.mcp_info = {"server_name": "my-server"}

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        # Allowed list contains the UUID, not the name
        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["uuid-abc-123"]

        captured = {"called": False, "server_arg": None}

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
            apply_tool_filters=True,
        ):
            captured["called"] = True
            captured["server_arg"] = server
            return ["tool-x"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_name",
            lambda name: stub_server if name == "my-server" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda sid: stub_server if sid == "uuid-abc-123" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id="my-server",  # pass name, not UUID
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert captured["called"] is True
        assert captured["server_arg"] is stub_server
        assert result["tools"] == ["tool-x"]
        assert result["error"] is None

    async def test_name_not_in_allowed_returns_access_denied(self, monkeypatch):
        """When name resolves to a server whose UUID is NOT in allowed_server_ids,
        the result should be an access_denied error (not a crash or silent pass)."""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        stub_server = MCPServer(
            server_id="uuid-xyz-999",
            name="restricted-server",
            transport=MCPTransport.sse,
        )
        stub_server.available_on_public_internet = True

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        # No allowed servers for this key
        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return []

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_name",
            lambda name: stub_server if name == "restricted-server" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda sid: stub_server if sid == "uuid-xyz-999" else None,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id="restricted-server",
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert result["tools"] == []
        assert result["error"] == "unexpected_error"
        assert "access_denied" in result["message"]

    async def test_mcp_server_name_query_param_resolves_to_server(self, monkeypatch):
        """mcp_server_name is a name-based alias for server_id: it should
        resolve to the matching server and scope the response to it."""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        stub_server = MCPServer(
            server_id="uuid-abc-123",
            name="my-server",
            transport=MCPTransport.sse,
        )
        stub_server.alias = "my-server"
        stub_server.server_name = "my-server"
        stub_server.available_on_public_internet = True
        stub_server.allowed_tools = None
        stub_server.mcp_info = {"server_name": "my-server"}

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["uuid-abc-123"]

        captured = {"called": False, "server_arg": None}

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
            apply_tool_filters=True,
        ):
            captured["called"] = True
            captured["server_arg"] = server
            return ["tool-x"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_name",
            lambda name: stub_server if name == "my-server" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda sid: stub_server if sid == "uuid-abc-123" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id=None,
            mcp_server_name="my-server",
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert captured["called"] is True
        assert captured["server_arg"] is stub_server
        assert result["tools"] == ["tool-x"]
        assert result["error"] is None

    async def test_mcp_server_name_filter_uses_real_catalog_with_tool_search(self, monkeypatch):
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.types.mcp import MCPTransport

        stub_server = MCPServer(
            server_id="uuid-search-123",
            name="search-server",
            transport=MCPTransport.sse,
        )
        stub_server.alias = "search-server"
        stub_server.server_name = "search-server"
        stub_server.available_on_public_internet = True
        stub_server.allowed_tools = None
        stub_server.mcp_info = {"server_name": "search-server"}
        user_api_key_dict = UserAPIKeyAuth(
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="search-scope",
                mcp_tool_search_enabled=True,
                mcp_servers=["uuid-search-123"],
            )
        )

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["uuid-search-123"]

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
            apply_tool_filters=True,
        ):
            return ["scoped-tool"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_name",
            lambda name: stub_server if name == "search-server" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "uuid-search-123" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id=None,
            mcp_server_name="search-server",
            user_api_key_dict=user_api_key_dict,
        )

        assert result["tools"] == ["scoped-tool"]
        assert result["error"] is None

    async def test_toolset_name_query_param_scopes_to_toolset_servers(self, monkeypatch):
        """toolset_name should resolve the toolset, apply its scope to the
        caller's UserAPIKeyAuth via _apply_toolset_scope, and only list tools
        from servers the scoped auth is allowed to see."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        scoped_auth = UserAPIKeyAuth(
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="toolset-scope",
                mcp_tool_search_enabled=True,
                mcp_servers=["toolset-server-1"],
            )
        )

        class StubToolset:
            toolset_id = "toolset-1"

        class StubServer:
            alias = "toolset-server-1"
            server_name = "toolset-server-1"
            name = "toolset-server-1"
            allowed_tools = None
            mcp_info = {"server_name": "toolset-server-1"}
            available_on_public_internet = True

        stub_server = StubServer()

        async def fake_get_toolset_by_name_cached(prisma_client, toolset_name):
            assert toolset_name == "research_tools"
            return StubToolset()

        async def fake_apply_toolset_scope(user_api_key_auth, toolset_id):
            assert toolset_id == "toolset-1"
            return scoped_auth

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(**kwargs):
            assert kwargs["user_api_key_auth"] is scoped_auth
            return ["toolset-server-1"]

        async def fake_get_tools(server, server_auth_header, *args, **kwargs):
            return ["toolset-tool-1"]

        monkeypatch.setattr(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_toolset_by_name_cached",
            fake_get_toolset_by_name_cached,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_apply_toolset_scope",
            fake_apply_toolset_scope,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "toolset-server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id=None,
            toolset_name="research_tools",
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert result["tools"] == ["toolset-tool-1"]
        assert result["error"] is None

    async def test_toolset_name_not_found_returns_error(self, monkeypatch):
        async def fake_get_toolset_by_name_cached(prisma_client, toolset_name):
            return None

        monkeypatch.setattr(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_toolset_by_name_cached",
            fake_get_toolset_by_name_cached,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.list_tool_rest_api(
                request,
                server_id=None,
                toolset_name="does-not-exist",
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert exc_info.value.status_code == 404
        assert "does-not-exist" in str(exc_info.value.detail)

    async def test_oauth2_user_token_injected_for_single_server(self, monkeypatch):
        """For a single-server OAuth2 request, _get_user_oauth_extra_headers is called
        and the returned headers are forwarded to _get_tools_for_single_server."""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        stub_server = MCPServer(
            server_id="oauth-server-id",
            name="oauth-server",
            transport=MCPTransport.sse,
        )
        stub_server.alias = "oauth-server"
        stub_server.server_name = "oauth-server"
        stub_server.available_on_public_internet = True
        stub_server.allowed_tools = None
        stub_server.mcp_info = {"server_name": "oauth-server"}
        stub_server.auth_type = MCPAuth.oauth2

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["oauth-server-id"]

        oauth_headers = {"Authorization": "Bearer user-oauth-token"}

        async def fake_get_user_oauth_extra_headers(server, user_api_key_dict, prefetched_creds=None):
            return oauth_headers

        captured = {}

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
            apply_tool_filters=True,
        ):
            captured["server"] = server
            captured["auth_header"] = server_auth_header
            return ["oauth-tool"]

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda sid: stub_server if sid == "oauth-server-id" else None,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_user_oauth_extra_headers",
            fake_get_user_oauth_extra_headers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "_get_tools_for_single_server",
            fake_get_tools,
            raising=False,
        )

        request = _build_request(path="/mcp-rest/tools/list", method="GET")
        result = await rest_endpoints.list_tool_rest_api(
            request,
            server_id="oauth-server-id",
            user_api_key_dict=UserAPIKeyAuth(user_id="user-123"),
        )

        assert result["tools"] == ["oauth-tool"]
        assert result["error"] is None


class TestCallToolRestAPI:
    pytestmark = pytest.mark.asyncio

    async def test_rejects_disallowed_server(self, monkeypatch):
        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return []

        async def fake_add_litellm_data_to_request(**kwargs):
            return kwargs.get("data", {})

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            fake_add_litellm_data_to_request,
            raising=False,
        )

        mock_server = MagicMock()
        mock_server.server_id = "server-1"

        def fake_get_mcp_server_by_id(server_id):
            return mock_server if server_id == "server-1" else None

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            fake_get_mcp_server_by_id,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_name",
            lambda *args, **kwargs: None,
            raising=False,
        )

        request_payload = {
            "server_id": "server-1",
            "name": "demo-tool",
            "arguments": {"foo": "bar"},
        }
        request = _build_request(
            path="/mcp-rest/tools/call",
            method="POST",
            json_body=request_payload,
        )

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.call_tool_rest_api(
                request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "access_denied"
        assert "server server-1" in exc_info.value.detail["message"]

    async def test_executes_tool_when_allowed(self, monkeypatch):
        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        class StubServer:
            server_id = "server-1"
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = None
            mcp_info = {"server_name": "stub"}
            available_on_public_internet = True
            auth_type = None

        stub_server = StubServer()

        async def fake_add_litellm_data_to_request(**kwargs):
            return kwargs.get("data", {})

        captured = {}

        async def fake_execute_mcp_tool(**kwargs):
            captured.update(kwargs)
            return {"result": "ok"}

        monkeypatch.setattr(
            rest_endpoints,
            "build_effective_auth_contexts",
            fake_contexts,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            fake_add_litellm_data_to_request,
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.proxy_config",
            {},
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints,
            "execute_mcp_tool",
            fake_execute_mcp_tool,
            raising=False,
        )
        fire_logging = AsyncMock(side_effect=RuntimeError("logging failed"))
        monkeypatch.setattr(
            rest_endpoints,
            "_fire_mcp_tool_call_logging",
            fire_logging,
            raising=False,
        )

        request_payload = {
            "server_id": "server-1",
            "name": "demo-tool",
            "arguments": {"foo": "bar"},
        }
        request = _build_request(
            path="/mcp-rest/tools/call",
            method="POST",
            json_body=request_payload,
        )

        result = await rest_endpoints.call_tool_rest_api(
            request,
            user_api_key_dict=UserAPIKeyAuth(),
        )

        assert result == {"result": "ok"}
        assert captured["name"] == "demo-tool"
        assert captured["arguments"] == {"foo": "bar"}
        assert captured["allowed_mcp_servers"] == [stub_server]
        fire_logging.assert_awaited_once()

    @pytest.mark.parametrize("upstream_status", [401, 403])
    async def test_call_tool_rest_relays_upstream_auth_failure(self, monkeypatch, upstream_status):
        """A pass-through call that hits an upstream 401/403 (surfaced by the manager as
        MCPUpstreamAuthError) must reach the REST caller as that status with the upstream
        WWW-Authenticate preserved, so an MCP client can run the upstream OAuth flow, instead of the
        generic 500 the catch-all would otherwise produce."""
        from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        class StubServer:
            server_id = "server-1"
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = None
            mcp_info = {"server_name": "stub"}
            available_on_public_internet = True
            auth_type = None

        stub_server = StubServer()

        async def fake_add_litellm_data_to_request(**kwargs):
            return kwargs.get("data", {})

        challenge = 'Bearer resource_metadata="https://gw.example.com/.well-known/oauth-protected-resource/mcp/stub"'

        async def fake_execute_mcp_tool(**kwargs):
            raise MCPUpstreamAuthError(
                status_code=upstream_status,
                www_authenticate=challenge,
                server_name="stub",
            )

        monkeypatch.setattr(rest_endpoints, "build_effective_auth_contexts", fake_contexts, raising=False)
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: stub_server if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            fake_add_litellm_data_to_request,
            raising=False,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {}, raising=False)
        monkeypatch.setattr(rest_endpoints, "execute_mcp_tool", fake_execute_mcp_tool, raising=False)

        mock_logger = MagicMock()
        monkeypatch.setattr(rest_endpoints, "verbose_logger", mock_logger, raising=False)

        request_payload = {
            "server_id": "server-1",
            "name": "demo-tool",
            "arguments": {"foo": "bar"},
        }
        request = _build_request(
            path="/mcp-rest/tools/call",
            method="POST",
            json_body=request_payload,
        )

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.call_tool_rest_api(
                request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert exc_info.value.status_code == upstream_status
        assert exc_info.value.headers is not None
        assert exc_info.value.headers.get("www-authenticate") == challenge
        # The expected caller-must-reauth signal is logged once, at info, and never at error, so
        # error-rate alerts do not fire on normal pass-through re-authentication.
        error_messages = [str(c.args[0]) for c in mock_logger.error.call_args_list if c.args]
        assert not any("MCP tool call" in m for m in error_messages)
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
        assert sum(str(upstream_status) in m for m in info_messages) == 1

    async def test_local_permission_denial_keeps_error_level_logging(self, monkeypatch):
        """Only the relayed upstream 401 may be demoted to info; a locally generated HTTPException 403
        (tool permission, server access, IP filtering) raised inside the call must stay at error level
        so an authenticated user probing restrictions keeps full monitoring visibility, and must be
        re-raised unchanged (not converted to a re-auth relay)."""

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_get_allowed_mcp_servers(*args, **kwargs):
            return ["server-1"]

        class StubServer:
            server_id = "server-1"
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = None
            mcp_info = {"server_name": "stub"}
            available_on_public_internet = True
            auth_type = None

        async def fake_add_litellm_data_to_request(**kwargs):
            return kwargs.get("data", {})

        async def fake_execute_mcp_tool(**kwargs):
            raise HTTPException(status_code=403, detail="tool not allowed for key")

        monkeypatch.setattr(rest_endpoints, "build_effective_auth_contexts", fake_contexts, raising=False)
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            fake_get_allowed_mcp_servers,
            raising=False,
        )
        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "get_mcp_server_by_id",
            lambda server_id: StubServer() if server_id == "server-1" else None,
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request, raising=False
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {}, raising=False)
        monkeypatch.setattr(rest_endpoints, "execute_mcp_tool", fake_execute_mcp_tool, raising=False)
        mock_logger = MagicMock()
        monkeypatch.setattr(rest_endpoints, "verbose_logger", mock_logger, raising=False)

        request = _build_request(
            path="/mcp-rest/tools/call",
            method="POST",
            json_body={"server_id": "server-1", "name": "demo-tool", "arguments": {}},
        )

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.call_tool_rest_api(request, user_api_key_dict=UserAPIKeyAuth())

        assert exc_info.value.status_code == 403
        error_messages = [str(c.args[0]) for c in mock_logger.error.call_args_list if c.args]
        assert any("HTTPException in MCP tool call" in m for m in error_messages)
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
        assert not any("relaying upstream" in m for m in info_messages)

    async def test_success_logging_cancellation_propagates(self, monkeypatch):
        fire_logging = AsyncMock(side_effect=asyncio.CancelledError())
        monkeypatch.setattr(
            rest_endpoints,
            "_fire_mcp_tool_call_logging",
            fire_logging,
            raising=False,
        )

        with pytest.raises(asyncio.CancelledError):
            await rest_endpoints._safe_fire_mcp_tool_call_logging(
                object(), {"result": "ok"}, datetime.now(), datetime.now()
            )

        fire_logging.assert_awaited_once()

    @pytest.mark.parametrize("upstream_status", [401, 403])
    async def test_virtual_mcp_tool_call_relays_upstream_auth_failure(self, monkeypatch, upstream_status):
        """The virtual mcp_tool_call REST branch reaches execute_mcp_tool via handle_mcp_tool_call
        without the direct branch's relay wrapper, so an MCPUpstreamAuthError from it must be relayed
        by the endpoint-level handler (a real 401/403 + WWW-Authenticate) rather than falling through
        the catch-all into a generic 500."""
        import litellm.proxy._experimental.mcp_server.tool_search as tool_search_mod
        from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        challenge = 'Bearer resource_metadata="https://gw.example.com/.well-known/oauth-protected-resource"'

        async def fake_contexts(user_api_key_auth):
            return [user_api_key_auth]

        async def fake_handle_mcp_tool_call(**kwargs):
            raise MCPUpstreamAuthError(status_code=upstream_status, www_authenticate=challenge, server_name="stub")

        class _FakePreCall:
            def __init__(self, data):
                pass

            async def common_processing_pre_call_logic(self, **kwargs):
                return None, MagicMock()

        monkeypatch.setattr(rest_endpoints, "build_effective_auth_contexts", fake_contexts, raising=False)
        monkeypatch.setattr(tool_search_mod, "handle_mcp_tool_call", fake_handle_mcp_tool_call, raising=False)
        monkeypatch.setattr(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
            _FakePreCall,
            raising=False,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {}, raising=False)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {}, raising=False)

        user_api_key_dict = UserAPIKeyAuth(
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="search-scope",
                mcp_tool_search_enabled=True,
            )
        )
        request = _build_request(
            path="/mcp-rest/tools/call",
            method="POST",
            json_body={"name": "mcp_tool_call", "arguments": {"tool_name": "x", "arguments": {}}},
        )

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.call_tool_rest_api(request, user_api_key_dict=user_api_key_dict)

        assert exc_info.value.status_code == upstream_status
        assert exc_info.value.headers is not None
        assert exc_info.value.headers.get("www-authenticate") == challenge


class TestGetToolsForSingleServer:
    """Test _get_tools_for_single_server with object_permission filtering"""

    pytestmark = pytest.mark.asyncio

    async def test_filters_tools_by_object_permission_mcp_tool_permissions(self, monkeypatch):
        """Test that tools are filtered by user_api_key_auth.object_permission.mcp_tool_permissions"""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.types.mcp import MCPTransport

        # Create mock tools
        class MockTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.inputSchema = {}

        mock_tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
            MockTool("tool3", "Third tool"),
        ]

        # Mock _get_tools_from_server to return all tools
        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        # Create server
        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=None,  # No server-level filtering
        )

        # Create UserAPIKeyAuth with object_permission
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-permission-id",
            mcp_tool_permissions={"test-server-id": ["tool1", "tool3"]},
        )

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            object_permission=object_permission,
        )

        # Call the function
        result = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )

        # Verify only allowed tools are returned
        assert len(result) == 2
        tool_names = [tool.name for tool in result]
        assert "tool1" in tool_names
        assert "tool3" in tool_names
        assert "tool2" not in tool_names

    async def test_no_filtering_when_object_permission_is_none(self, monkeypatch):
        """Test that all tools are returned when object_permission is None"""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        class MockTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.inputSchema = {}

        mock_tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
        ]

        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=None,
        )

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            object_permission=None,
        )

        result = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )

        # All tools should be returned
        assert len(result) == 2

    async def test_no_filtering_when_mcp_tool_permissions_is_none(self, monkeypatch):
        """Test that all tools are returned when mcp_tool_permissions is None"""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.types.mcp import MCPTransport

        class MockTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.inputSchema = {}

        mock_tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
        ]

        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=None,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-permission-id",
            mcp_tool_permissions=None,  # No tool permissions set
        )

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            object_permission=object_permission,
        )

        result = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )

        # All tools should be returned
        assert len(result) == 2

    async def test_no_filtering_when_server_not_in_mcp_tool_permissions(self, monkeypatch):
        """Test that all tools are returned when server is not in mcp_tool_permissions"""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.types.mcp import MCPTransport

        class MockTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.inputSchema = {}

        mock_tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
        ]

        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=None,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-permission-id",
            mcp_tool_permissions={"other-server-id": ["tool1"]},  # Different server
        )

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            object_permission=object_permission,
        )

        result = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )

        # All tools should be returned since server is not in permissions
        assert len(result) == 2

    async def test_combines_server_allowed_tools_and_object_permission_filters(self, monkeypatch):
        """Test that both server.allowed_tools and object_permission.mcp_tool_permissions filters are applied"""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.types.mcp import MCPTransport

        class MockTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.inputSchema = {}

        mock_tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
            MockTool("tool3", "Third tool"),
            MockTool("tool4", "Fourth tool"),
        ]

        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        # Server allows tool1, tool2, tool3
        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=["tool1", "tool2", "tool3"],
        )

        # Object permission allows tool2, tool3, tool4
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-permission-id",
            mcp_tool_permissions={"test-server-id": ["tool2", "tool3", "tool4"]},
        )

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            object_permission=object_permission,
        )

        result = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )

        # Only tools in both lists should be returned (intersection): tool2, tool3
        assert len(result) == 2
        tool_names = [tool.name for tool in result]
        assert "tool2" in tool_names
        assert "tool3" in tool_names
        assert "tool1" not in tool_names
        assert "tool4" not in tool_names

    async def test_apply_tool_filters_false_returns_full_catalog(self, monkeypatch):
        """apply_tool_filters=False returns the raw catalog without the server
        allowed_tools gate, so the config UI can render disabled tools as off."""
        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        class MockTool:
            def __init__(self, name):
                self.name = name
                self.description = name
                self.inputSchema = {}

        mock_tools = [MockTool("tool1"), MockTool("tool2"), MockTool("tool3")]

        async def fake_get_tools_from_server(**kwargs):
            return mock_tools

        monkeypatch.setattr(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            fake_get_tools_from_server,
            raising=False,
        )

        # Server enforces an allowlist of just tool1.
        server = MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport=MCPTransport.sse,
            allowed_tools=["tool1"],
        )
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key", object_permission=None)

        # Runtime default: only the allowed tool comes back.
        filtered = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
        )
        assert [t.name for t in filtered] == ["tool1"]

        # Config view: full catalog, including the disabled tools.
        full = await rest_endpoints._get_tools_for_single_server(
            server=server,
            server_auth_header=None,
            user_api_key_auth=user_api_key_dict,
            apply_tool_filters=False,
        )
        assert {t.name for t in full} == {"tool1", "tool2", "tool3"}


class TestStdioCommandAllowlist:
    """Tests for MCP stdio command allowlist validation."""

    def test_allowed_command_passes_validation(self):
        """npx, uvx, python, etc. should be accepted."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        assert req.command == "npx"

    def test_disallowed_command_raises(self):
        """Arbitrary commands like bash should be rejected."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="bash",
                args=["-c", "echo pwned"],
            )

    def test_sh_command_raises(self):
        """sh should be rejected."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="sh",
                args=["-c", "id > /tmp/output.txt"],
            )

    def test_absolute_path_bypass_blocked(self):
        """/bin/bash should be blocked (basename is 'bash')."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="/bin/bash",
                args=["-c", "echo pwned"],
            )

    def test_absolute_path_to_allowed_command_works(self):
        """/usr/bin/python3 should pass (basename is 'python3')."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="/usr/bin/python3",
            args=["-m", "some_module"],
        )
        assert req.command == "/usr/bin/python3"

    def test_http_transport_ignores_allowlist(self):
        """HTTP/SSE transport should not trigger command validation."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="sse",
            url="https://example.com/mcp",
        )
        assert req.transport == "sse"

    def test_uvx_command_passes(self):
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="uvx",
            args=["mcp-server-sqlite"],
        )
        assert req.command == "uvx"

    def test_node_command_passes(self):
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="node",
            args=["server.js"],
        )
        assert req.command == "node"

    def test_update_request_disallowed_command_raises(self):
        """UpdateMCPServerRequest should also block non-allowlisted commands."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            UpdateMCPServerRequest(
                server_id="some-id",
                transport="stdio",
                command="bash",
                args=["-c", "echo pwned"],
            )


class TestEndpointRoleChecks:
    """Tests for PROXY_ADMIN role checks on MCP test endpoints."""

    def test_test_connection_has_auth_dependency(self):
        route = _get_route("/mcp-rest/test/connection", "POST")
        assert _route_has_dependency(route, user_api_key_auth)

    def test_test_tools_list_has_auth_dependency(self):
        route = _get_route("/mcp-rest/test/tools/list", "POST")
        assert _route_has_dependency(route, user_api_key_auth)

    @pytest.mark.asyncio
    async def test_test_connection_rejects_non_admin(self):
        """Non-admin users should get 403 from test_connection."""
        from litellm.proxy._types import LitellmUserRoles

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="non_admin",
            api_key="sk-test",
        )
        request = _build_request()

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.test_connection(
                request=request,
                new_mcp_server_request=payload,
                user_api_key_dict=user_key,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_test_tools_list_rejects_non_admin(self):
        """Non-admin users should get 403 from test_tools_list."""
        from litellm.proxy._types import LitellmUserRoles

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="non_admin",
            api_key="sk-test",
        )
        request = _build_request()

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.test_tools_list(
                request=request,
                new_mcp_server_request=payload,
                user_api_key_dict=user_key,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_test_connection_allows_admin(self, monkeypatch):
        """PROXY_ADMIN should pass the role check."""
        from litellm.proxy._types import LitellmUserRoles

        async def fake_execute(*args, **kwargs):
            return {"status": "ok"}

        monkeypatch.setattr(
            rest_endpoints,
            "_execute_with_mcp_client",
            fake_execute,
        )

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin",
            api_key="sk-admin",
        )
        request = _build_request()

        result = await rest_endpoints.test_connection(
            request=request,
            new_mcp_server_request=payload,
            user_api_key_dict=user_key,
        )
        assert result["status"] == "ok"


class TestPreviewOpenAPITools:
    """Verify the OpenAPI preview endpoint emits provider-safe tool names.

    Regression: GitHub's OpenAPI spec uses tag-namespaced operationIds like
    `actions/download-job-logs-for-workflow-run` which contain '/'. The
    preview must sanitize so what the dashboard shows matches what gets
    registered (and what makes it past LLM provider tool-name validation).
    """

    pytestmark = pytest.mark.asyncio

    async def test_preview_sanitizes_slash_in_operation_id(self, monkeypatch):
        import re

        async def fake_load_spec(spec_path):  # noqa: ANN001
            return {
                "paths": {
                    "/repos/{owner}/{repo}/actions/jobs/{job_id}/logs": {
                        "get": {
                            "operationId": ("actions/download-job-logs-for-workflow-run"),
                            "summary": "Download job logs",
                        }
                    },
                    "/repos/{owner}/{repo}/pulls/{pull_number}/files": {
                        "get": {
                            "operationId": "pulls/list-files",
                            "summary": "List files",
                        }
                    },
                }
            }

        from litellm.proxy._experimental.mcp_server import (
            openapi_to_mcp_generator,
        )

        monkeypatch.setattr(
            openapi_to_mcp_generator,
            "load_openapi_spec_async",
            fake_load_spec,
            raising=False,
        )

        payload = NewMCPServerRequest(
            server_name="github_openapi_mcp",
            spec_path="https://example.invalid/openapi.json",
            transport="http",
        )
        request = _build_request()

        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        assert result.get("error") is None, result
        names = [t["name"] for t in result["tools"]]
        anthropic_re = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
        for name in names:
            assert anthropic_re.match(name), f"preview tool name {name!r} violates ^[a-zA-Z0-9_-]+$"
        assert "actions_download-job-logs-for-workflow-run" in names
        assert "pulls_list-files" in names

    async def test_preview_method_order_matches_registration(self, monkeypatch):
        """Preview must iterate HTTP methods in the same order as
        register_tools_from_openapi, otherwise collision-disambiguation
        suffixes (_2, _3, ...) get assigned to different operations and the
        dashboard shows names that differ from what's actually registered.
        """
        from litellm.proxy._experimental.mcp_server import (
            openapi_to_mcp_generator,
        )

        spec = {
            "paths": {
                "/items/{id}": {
                    "delete": {
                        "operationId": "items/delete",
                        "summary": "Delete item",
                    },
                    "patch": {
                        "operationId": "items.delete",
                        "summary": "Soft-delete item",
                    },
                }
            }
        }

        async def fake_load_spec(spec_path):  # noqa: ANN001
            return spec

        monkeypatch.setattr(
            openapi_to_mcp_generator,
            "load_openapi_spec_async",
            fake_load_spec,
            raising=False,
        )

        payload = NewMCPServerRequest(
            server_name="collision_openapi_mcp",
            spec_path="https://example.invalid/openapi.json",
            transport="http",
        )
        request = _build_request()
        from litellm.proxy._types import LitellmUserRoles

        result = await rest_endpoints.test_tools_list(
            request,
            payload,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
        assert result.get("error") is None, result
        preview_summary_to_name = {t["description"]: t["name"] for t in result["tools"]}

        registered_summary_to_name: dict = {}

        def fake_create_tool_function(path, method, operation, base_url):  # noqa: ANN001
            def _f():
                return None

            return _f

        monkeypatch.setattr(
            openapi_to_mcp_generator,
            "create_tool_function",
            fake_create_tool_function,
        )

        class _StubRegistry:
            def register_tool(self, name, description, input_schema, handler):  # noqa: ANN001
                registered_summary_to_name[description] = name

        monkeypatch.setattr(
            openapi_to_mcp_generator,
            "global_mcp_tool_registry",
            _StubRegistry(),
        )

        openapi_to_mcp_generator.register_tools_from_openapi(spec, base_url="https://example.invalid")

        assert preview_summary_to_name == registered_summary_to_name, (
            f"preview {preview_summary_to_name} != "
            f"registered {registered_summary_to_name} — method iteration "
            "order is out of sync, so collision suffixes (_2, _3, ...) "
            "land on different operations"
        )


class TestConnectionErrorMessage:
    """The test-connection endpoints turn raw transport errors into messages.

    The message is returned to an admin in an API response, so it must explain
    the failure without echoing the raw header value, which can carry a secret
    (e.g. ``Authorization: Bearer <token>``).
    """

    def test_local_protocol_error_is_actionable_and_redacted(self):
        secret = "Bearer sk-super-secret-token"
        exc = httpx.LocalProtocolError(f"Illegal header value b' {secret}'")

        message = rest_endpoints._connection_error_message(exc)

        assert "header" in message.lower()
        assert secret not in message

    def test_connect_error_points_at_reachability(self):
        message = rest_endpoints._connection_error_message(httpx.ConnectError("All connection attempts failed"))
        assert "unreachable" in message.lower()

    def test_timeout_error_message(self):
        message = rest_endpoints._connection_error_message(httpx.ConnectTimeout("timed out"))
        assert "unreachable" in message.lower()

    def test_http_status_error_includes_status_code(self):
        response = httpx.Response(status_code=503)
        exc = httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("POST", "http://x/"),
            response=response,
        )
        message = rest_endpoints._connection_error_message(exc)
        assert "503" in message

    def test_unknown_error_falls_back_to_generic(self):
        message = rest_endpoints._connection_error_message(RuntimeError("weird"))
        assert "weird" not in message
        assert "proxy logs" in message.lower()


class TestToolResponseMcpInfoEnrichment:
    """The REST tools/list response must expose the user-facing alias and the
    server_id alongside the internal server_name so clients (agent builder UIs)
    can map the internal config key to a friendly name without needing the
    mcp_routes-gated server listing.
    """

    def test_enriches_mcp_info_with_alias_and_server_id(self):
        from mcp.types import Tool as MCPTool

        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        server = MCPServer(
            server_id="a1b2c3d4",
            name="mcpAtlassian",
            alias="atlassian",
            server_name="mcpAtlassian",
            transport=MCPTransport.http,
            mcp_info={"server_name": "mcpAtlassian"},
        )
        tools = [
            MCPTool(
                name="get_issue",
                description="Fetch a Jira issue",
                inputSchema={"type": "object"},
            )
        ]

        result = rest_endpoints._create_tool_response_objects(tools, server)

        assert result[0].mcp_info == {
            "server_name": "mcpAtlassian",
            "server_id": "a1b2c3d4",
            "alias": "atlassian",
        }

    def test_alias_none_is_explicit_in_mcp_info(self):
        from mcp.types import Tool as MCPTool

        from litellm.proxy._experimental.mcp_server.server import MCPServer
        from litellm.types.mcp import MCPTransport

        server = MCPServer(
            server_id="server-uuid",
            name="no_alias_server",
            alias=None,
            server_name="no_alias_server",
            transport=MCPTransport.http,
            mcp_info={"server_name": "no_alias_server"},
        )
        tools = [
            MCPTool(
                name="ping",
                description="Ping",
                inputSchema={"type": "object"},
            )
        ]

        result = rest_endpoints._create_tool_response_objects(tools, server)

        assert result[0].mcp_info == {
            "server_name": "no_alias_server",
            "server_id": "server-uuid",
            "alias": None,
        }

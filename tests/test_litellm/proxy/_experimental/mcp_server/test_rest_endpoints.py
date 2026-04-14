import json
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.auth import (
    user_api_key_auth_mcp as auth_mcp,
)
from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest, UserAPIKeyAuth
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
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in headers.items()
    ]
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
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", set()
        ):
            return route
    raise AssertionError(f"Route {method} {path} not found")


def _route_has_dependency(route, dependency) -> bool:
    if any(
        getattr(dep, "dependency", None) == dependency
        for dep in getattr(route, "dependencies", [])
    ):
        return True
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    return any(
        getattr(dep, "call", None) == dependency for dep in dependant.dependencies
    )


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

        result = await rest_endpoints._execute_with_mcp_client(
            payload, failing_operation
        )

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
        assert (
            captured["extra_headers"] is None
            or "Authorization" not in captured["extra_headers"]
        )

    @pytest.mark.asyncio
    async def test_catches_exception_group(self, monkeypatch):
        """MCP SDK's anyio TaskGroup raises BaseExceptionGroup which does not
        inherit from Exception.  The handler must catch it and return an error
        dict instead of letting a raw 500 propagate."""

        def fake_build_stdio_env(server, raw_headers):
            return None

        async def fake_create_client(*args, **kwargs):
            raise BaseExceptionGroup(
                "test group", [RuntimeError("Cancelled via cancel scope")]
            )

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

        monkeypatch.setattr(
            rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False
        )

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

        monkeypatch.setattr(
            rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False
        )

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

        request = _build_request({"authorization": "Bearer incoming"})
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

        async def fake_get_user_oauth_extra_headers(
            server, user_api_key_dict, prefetched_creds=None
        ):
            return oauth_headers

        captured = {}

        async def fake_get_tools(
            server,
            server_auth_header,
            raw_headers=None,
            user_api_key_auth=None,
            extra_headers=None,
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


class TestGetToolsForSingleServer:
    """Test _get_tools_for_single_server with object_permission filtering"""

    pytestmark = pytest.mark.asyncio

    async def test_filters_tools_by_object_permission_mcp_tool_permissions(
        self, monkeypatch
    ):
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

    async def test_no_filtering_when_server_not_in_mcp_tool_permissions(
        self, monkeypatch
    ):
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

    async def test_combines_server_allowed_tools_and_object_permission_filters(
        self, monkeypatch
    ):
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

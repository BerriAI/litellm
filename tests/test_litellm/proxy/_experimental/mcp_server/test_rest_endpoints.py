import json
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.auth import (
    user_api_key_auth_mcp as auth_mcp,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy._types import NewMCPServerRequest, UserAPIKeyAuth
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
    return any(getattr(dep, "call", None) == dependency for dep in dependant.dependencies)


class TestExecuteWithMcpClient:
    @pytest.mark.asyncio
    async def test_redacts_stack_trace(self, monkeypatch):
        def fake_create_client(*args, **kwargs):
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

        result = await rest_endpoints.test_tools_list(
            request, payload, user_api_key_dict=UserAPIKeyAuth()
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

        result = await rest_endpoints.test_tools_list(
            request, payload, user_api_key_dict=UserAPIKeyAuth()
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

        stub_server = StubServer()

        captured = {"called": False}

        async def fake_get_tools(server, server_auth_header, raw_headers=None):
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
            alias = "server-1"
            server_name = "server-1"
            name = "stub"
            allowed_tools = None
            mcp_info = {"server_name": "stub"}

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

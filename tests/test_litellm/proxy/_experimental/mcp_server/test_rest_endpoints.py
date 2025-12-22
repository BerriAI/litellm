from typing import Dict, Optional

import pytest
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.auth import (
    user_api_key_auth_mcp as auth_mcp,
)
from litellm.proxy._types import NewMCPServerRequest, UserAPIKeyAuth
from litellm.types.mcp import MCPAuth


def _build_request(headers: Optional[Dict[str, str]] = None) -> Request:
    headers = headers or {}
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in headers.items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/mcp-rest/test/tools/list",
        "headers": raw_headers,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_test_tools_list_forwards_mcp_auth_header(monkeypatch):
    """Ensure credential-based auth forwards the auth_value to the MCP client."""

    captured: dict = {}

    async def fake_execute(request, operation, mcp_auth_header=None, oauth2_headers=None):
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


@pytest.mark.asyncio
async def test_test_tools_list_extracts_oauth2_headers(monkeypatch):
    """Ensure oauth2 auth type pulls oauth headers and omits MCP auth header."""

    captured: dict = {}

    async def fake_execute(request, operation, mcp_auth_header=None, oauth2_headers=None):
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

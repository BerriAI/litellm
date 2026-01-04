"""Tests for OpenAPI to MCP generator path handling."""

import pytest

from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    create_tool_function,
)


class _DummyResponse:
    def __init__(self, text: str = "ok"):
        self.text = text


class _DummyAsyncClient:
    """Minimal async client stub that records requests."""

    last_instance = None

    def __init__(self):
        self.requests = []
        _DummyAsyncClient.last_instance = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None, headers=None):
        self.requests.append(("get", url, params, headers))
        return _DummyResponse("dummy-response")


@pytest.mark.asyncio
async def test_should_reject_path_traversal_inputs(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
        lambda *_, **__: _DummyAsyncClient(),
    )
    _DummyAsyncClient.last_instance = None

    operation = {
        "parameters": [
            {
                "name": "filename",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
            }
        ]
    }

    tool_function = create_tool_function(
        path="/files/{filename}",
        method="GET",
        operation=operation,
        base_url="https://example.com",
    )

    response = await tool_function(filename="../admin")

    assert "Invalid path parameter" in response


@pytest.mark.asyncio
async def test_should_encode_and_request_safe_path_parameters(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
        lambda *_, **__: _DummyAsyncClient(),
    )
    _DummyAsyncClient.last_instance = None

    operation = {
        "parameters": [
            {
                "name": "filename",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
            }
        ]
    }

    tool_function = create_tool_function(
        path="/files/{filename}",
        method="GET",
        operation=operation,
        base_url="https://example.com",
    )

    response = await tool_function(filename="report 2024.json")

    assert response == "dummy-response"

    dummy_client = _DummyAsyncClient.last_instance
    assert dummy_client is not None
    method, url, params, headers = dummy_client.requests[0]
    assert method == "get"
    assert url == "https://example.com/files/report%202024.json"
    assert params == {}

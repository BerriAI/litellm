"""
Tests for OpenAPI to MCP generator, focusing on security and edge cases.

This test suite ensures that:
1. Parameter names with invalid Python identifiers are handled safely
2. No exec() is used (security)
3. All edge cases (hyphens, dots, keywords, special chars) work correctly
4. Path traversal attacks are prevented
5. Path parameters are properly URL encoded
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    create_tool_function,
    build_input_schema,
    extract_parameters,
)


GET_ASYNC_CLIENT_TARGET = (
    "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client"
)


def _create_mock_client(method: str, response_text: str) -> AsyncMock:
    """Utility to create a mocked async httpx client for the given method."""
    response = SimpleNamespace(text=response_text)
    client = AsyncMock()
    setattr(client, method, AsyncMock(return_value=response))
    return client


class TestCreateToolFunction:
    """Test create_tool_function with various parameter name edge cases."""

    @pytest.mark.asyncio
    async def test_hyphenated_path_parameter(self):
        """Test function with hyphenated path parameter (e.g., repository-id)."""
        operation = {
            "parameters": [
                {
                    "name": "repository-id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ]
        }

        func = create_tool_function(
            path="/repos/{repository-id}",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        # Should not raise SyntaxError
        assert callable(func)
        assert func.__name__ == "tool_function"

        # Test calling with original parameter name
        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", '{"id": "123"}')
            mock_client.return_value = async_client

            result = await func(**{"repository-id": "test-repo"})
            assert result == '{"id": "123"}'

            # Verify URL was constructed correctly
            call_args = async_client.get.call_args
            assert "repository-id" in str(call_args[0][0]) or "test-repo" in str(
                call_args[0][0]
            )

    @pytest.mark.asyncio
    async def test_leading_digit_parameter(self):
        """Test function with parameter starting with digit (e.g., 2fa-code)."""
        operation = {
            "parameters": [
                {
                    "name": "2fa-code",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                }
            ]
        }

        func = create_tool_function(
            path="/verify",
            method="post",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("post", "verified")
            mock_client.return_value = async_client

            result = await func(**{"2fa-code": "123456"})
            assert result == "verified"

            # Verify query parameter was included
            call_args = async_client.post.call_args
            assert call_args[1]["params"]["2fa-code"] == "123456"

    @pytest.mark.asyncio
    async def test_dot_in_parameter_name(self):
        """Test function with dot in parameter name (e.g., user.name)."""
        operation = {
            "parameters": [
                {
                    "name": "user.name",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                }
            ]
        }

        func = create_tool_function(
            path="/search",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "found")
            mock_client.return_value = async_client

            result = await func(**{"user.name": "john.doe"})
            assert result == "found"

            call_args = async_client.get.call_args
            assert call_args[1]["params"]["user.name"] == "john.doe"

    @pytest.mark.asyncio
    async def test_dollar_sign_parameter(self):
        """Test function with dollar sign parameter (OData style, e.g., $filter)."""
        operation = {
            "parameters": [
                {
                    "name": "$filter",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                }
            ]
        }

        func = create_tool_function(
            path="/entities",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "[]")
            mock_client.return_value = async_client

            result = await func(**{"$filter": "name eq 'test'"})
            assert result == "[]"

            call_args = async_client.get.call_args
            assert call_args[1]["params"]["$filter"] == "name eq 'test'"

    @pytest.mark.asyncio
    async def test_python_keyword_parameter(self):
        """Test function with Python keyword as parameter name (e.g., class)."""
        operation = {
            "parameters": [
                {
                    "name": "class",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                }
            ]
        }

        func = create_tool_function(
            path="/items",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "items")
            mock_client.return_value = async_client

            result = await func(**{"class": "premium"})
            assert result == "items"

            call_args = async_client.get.call_args
            assert call_args[1]["params"]["class"] == "premium"

    @pytest.mark.asyncio
    async def test_multiple_problematic_parameters(self):
        """Test function with multiple problematic parameter names."""
        operation = {
            "parameters": [
                {
                    "name": "repository-id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "2fa-code",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                },
                {
                    "name": "$filter",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                },
            ]
        }

        func = create_tool_function(
            path="/repos/{repository-id}",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "success")
            mock_client.return_value = async_client

            result = await func(
                **{
                    "repository-id": "test-repo",
                    "2fa-code": "123",
                    "$filter": "active",
                }
            )
            assert result == "success"

    @pytest.mark.asyncio
    async def test_request_body_parameter(self):
        """Test function with request body parameter."""
        operation = {
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        }
                    }
                },
            }
        }

        func = create_tool_function(
            path="/create",
            method="post",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("post", "created")
            mock_client.return_value = async_client

            result = await func(**{"body": {"name": "test"}})
            assert result == "created"

            call_args = async_client.post.call_args
            assert call_args[1]["json"] == {"name": "test"}

    @pytest.mark.asyncio
    async def test_no_parameters(self):
        """Test function with no parameters."""
        operation = {}

        func = create_tool_function(
            path="/health",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        assert callable(func)

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "ok")
            mock_client.return_value = async_client

            result = await func()
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_all_http_methods(self):
        """Test all supported HTTP methods."""
        methods = ["get", "post", "put", "delete", "patch"]

        for method in methods:
            operation = {
                "parameters": [
                    {
                        "name": "repository-id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ]
            }

            func = create_tool_function(
                path="/repos/{repository-id}",
                method=method,
                operation=operation,
                base_url="https://api.example.com",
            )

            assert callable(func)

            with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
                async_client = _create_mock_client(method, "success")
                mock_client.return_value = async_client

                result = await func(**{"repository-id": "test"})
                assert result == "success"

    def test_no_exec_usage(self):
        """Verify that create_tool_function does not use exec()."""
        import ast
        import inspect

        # Get the source code of create_tool_function
        source = inspect.getsource(create_tool_function)

        # Parse the AST
        tree = ast.parse(source)

        # Check for exec() calls
        exec_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "exec":
                    exec_calls.append(node)

        # Should have no exec() calls
        assert len(exec_calls) == 0, "create_tool_function should not use exec()"


class TestBuildInputSchema:
    """Test that build_input_schema preserves original parameter names."""

    def test_original_parameter_names_preserved(self):
        """Test that original parameter names are preserved in input schema."""
        operation = {
            "parameters": [
                {
                    "name": "repository-id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "2fa-code",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                },
                {
                    "name": "$filter",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                },
            ]
        }

        schema = build_input_schema(operation)

        # Original names should be in the schema
        assert "repository-id" in schema["properties"]
        assert "2fa-code" in schema["properties"]
        assert "$filter" in schema["properties"]

        # Required should include original names
        assert "repository-id" in schema["required"]


class TestExtractParameters:
    """Test parameter extraction from OpenAPI operations."""

    def test_extract_path_query_body_params(self):
        """Test extraction of different parameter types."""
        operation = {
            "parameters": [
                {"name": "repo-id", "in": "path"},
                {"name": "filter", "in": "query"},
                {"name": "data", "in": "body"},
            ],
            "requestBody": {
                "content": {"application/json": {"schema": {"type": "object"}}}
            },
        }

        path_params, query_params, body_params = extract_parameters(operation)

        assert "repo-id" in path_params
        assert "filter" in query_params
        assert "data" in body_params
        assert "body" in body_params  # From requestBody


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPathSecurity:
    """Test path traversal security and URL encoding."""

    @pytest.mark.asyncio
    async def test_should_reject_path_traversal_inputs(self):
        """Test that path traversal attacks (../admin) are rejected."""
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

        response = await tool_function(**{"filename": "../admin"})

        assert "Invalid path parameter" in response

    @pytest.mark.asyncio
    async def test_should_encode_and_request_safe_path_parameters(self):
        """Test that path parameters are properly URL encoded."""
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

        with patch(GET_ASYNC_CLIENT_TARGET) as mock_client:
            async_client = _create_mock_client("get", "dummy-response")
            mock_client.return_value = async_client

            response = await tool_function(**{"filename": "report 2024.json"})

            assert response == "dummy-response"

            # Verify URL was properly encoded
            call_args = async_client.get.call_args
            url = call_args[0][0]
            assert url == "https://example.com/files/report%202024.json"
